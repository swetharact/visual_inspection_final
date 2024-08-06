import cv2
import numpy as np
import time

# Load YOLOv3 model for helmet detection
helmet_net = cv2.dnn.readNet('yolov3-helmet.weights', 'yolov3-helmet.cfg')
helmet_classes = []
with open('helmet.names', 'r') as f:
    helmet_classes = [line.strip() for line in f]

# Load YOLOv3 model for object detection (assuming person detection)
object_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
object_classes = []
with open('coco.names', 'r') as f:
    object_classes = [line.strip() for line in f]

# Set up webcam
cap = cv2.VideoCapture(0)

# Non-maximum suppression parameters
nms_threshold = 0.5

# Define restricted area (left half of the frame)
restricted_area = [(0, 0), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))]

def is_inside_restricted_area(x, y, w, h, restricted_area):
    x1, y1 = restricted_area[0]
    x2, y2 = restricted_area[1]
    return x < x2 and x + w > x1 and y < y2 and y + h > y1

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    height, width, _ = frame.shape

    # Detect helmets
    helmet_blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    helmet_net.setInput(helmet_blob)
    helmet_output_layers = helmet_net.getUnconnectedOutLayersNames()
    helmet_detections = helmet_net.forward(helmet_output_layers)

    helmet_boxes = []
    helmet_confidences = []

    for detection in helmet_detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.8:
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                helmet_boxes.append([x, y, w, h])
                helmet_confidences.append(float(confidence))

    helmet_indices = cv2.dnn.NMSBoxes(helmet_boxes, helmet_confidences, 0.5, nms_threshold)

    helmet_detected = False
    helmet_persons = []
    if len(helmet_indices) > 0:
        helmet_detected = True
        for i in helmet_indices.flatten():
            x, y, w, h = helmet_boxes[i]
            helmet_persons.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle
            cv2.putText(frame, 'Helmet Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow text

    # Detect people
    object_blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    object_net.setInput(object_blob)
    object_output_layers = object_net.getUnconnectedOutLayersNames()
    object_detections = object_net.forward(object_output_layers)

    object_boxes = []
    object_confidences = []

    for detection in object_detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and object_classes[class_id] == 'person':
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype("int")
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                object_boxes.append([x, y, w, h])
                object_confidences.append(float(confidence))

    object_indices = cv2.dnn.NMSBoxes(object_boxes, object_confidences, 0.5, nms_threshold)

    object_detected = False
    object_in_restricted_area = False
    if len(object_indices) > 0:
        object_detected = True
        for i in object_indices.flatten():
            x, y, w, h = object_boxes[i]
            wearing_helmet = False
            for hx, hy, hw, hh in helmet_persons:
                if x < hx + hw and x + w > hx and y < hy + hh and y + h > hy:
                    wearing_helmet = True
                    break

            if is_inside_restricted_area(x, y, w, h, restricted_area):
                object_in_restricted_area = True
                color = (0, 0, 255)  # Red for restricted area
                label = 'Person in Restricted Area'
            else:
                color = (0, 255, 0)  # Green for unrestricted area
                label = 'Person Detected'

            if wearing_helmet:
                helmet_status = 'Helmet Detected'
            else:
                helmet_status = 'No Helmet Detected'

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, helmet_status, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)  # Yellow text

    # Add a dividing line between the two areas
    cv2.line(frame, (int(width / 2), 0), (int(width / 2), height), (255, 255, 255), 2)

    # Display the result on the frame
    cv2.putText(frame, 'Restricted Area', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, 'Unrestricted Area', (int(width / 2) + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the annotated frame
    cv2.imshow('Detection Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
