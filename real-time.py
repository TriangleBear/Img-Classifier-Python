import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("D:\Programming\Python\Img-Classifier-Python\yolov3.weights", "D:\Programming\Python\Img-Classifier-Python\yolov3.cfg")
classes = []
with open("D:\Programming\Python\Img-Classifier-Python\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up camera capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,)


while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    # Process the detection results
    class_ids = []
    confidences = []
    boxes = []
    height, width, _ = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (255, 0, 0)  # BGR color format
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
