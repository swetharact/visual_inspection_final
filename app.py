import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Realtime Database
cred = credentials.Certificate('fire-836b7-firebase-adminsdk-o14a6-b5aaf1ddc1.json')
firebase_admin.initialize_app(cred, {"databaseURL": "https://fire-836b7-default-rtdb.firebaseio.com/"})
db_ref = db.reference("/helmet_detection")

# Load YOLOv3 model and configuration
yolo_net = cv2.dnn.readNet('yolov3-helmet.weights', 'yolov3-helmet.cfg')
classes = []
with open('helmet.names', 'r') as f:
    classes = [line.strip() for line in f]

# Set up webcam
cap = cv2.VideoCapture(0)

# Non-maximum suppression parameters
nms_threshold = 0.5

while True:
    ret, frame = cap.read()

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Prepare the frame for YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    output_layers =   yolo_net.getUnconnectedOutLayersNames()
    detections = yolo_net.forward(output_layers)

    # Process YOLO output for helmet detection
    boxes = []
    confidences = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.8:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)

    helmet_detected = 0

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Helmet Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            helmet_detected = 1

    # Display the result on the frame
    if not helmet_detected:
        cv2.putText(frame, 'No Helmet Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam Feed', frame)

    # Store the boolean value in Firebase Realtime Database
    db_ref.set({'helmet_detected': helmet_detected})

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()