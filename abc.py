import cv2
import numpy as np


# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes


def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, height, width


def draw_labels_and_check_objects(img, outs, height, width, classes):
    boxes = []
    confidences = []
    class_ids = []

    water_bottle_detected_left = False
    water_bottle_detected_right = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':  # Assuming 'person' is in coco.names
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if the person is in left or right partition
                if center_x < width // 2:
                    water_bottle_detected_left = True
                else:
                    water_bottle_detected_right = True

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if x + w // 2 < width // 2:  # Check if the person is on the left
                color = (0, 255, 0)  # Green color
            else:  # Person is on the right
                color = (0, 0, 255)  # Red color
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

    return img, water_bottle_detected_left, water_bottle_detected_right


def start_video_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    net, output_layers, classes = load_yolo()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        outs, height, width = detect_objects(frame, net, output_layers)
        frame, water_bottle_detected_left, water_bottle_detected_right = draw_labels_and_check_objects(frame, outs,
                                                                                                       height, width,
                                                                                                       classes)

        # Draw the partitions
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

        # Display the output status
        if water_bottle_detected_left:
            print("Unrestricted area")
        if water_bottle_detected_right:
            print("Restricted area")

        cv2.imshow("Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_video_capture()
