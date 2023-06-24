import cv2
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("Python/yolo Folder/yolov3.weights", "Python/yolo Folder/yolov3.cfg")

# Load the class labels
with open("D:/Programming/Python/yolo Folder/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the classes for living things, vehicles, and plants
living_thing_classes = ['person', 'cat', 'dog']
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']
plant_classes = ['pottedplant']

def select_image():
    root = tk.Tk()
    root.withdraw()

    # Prompt the user to select an image file
    file_path = filedialog.askopenfilename()
    return file_path

def perform_object_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    # Resize the image to a desired size while maintaining aspect ratio
    desired_width = 600
    desired_height = int(height * (desired_width / width))
    resized_image = cv2.resize(image, (desired_width, desired_height))

    # Create a blob from the resized image
    blob = cv2.dnn.blobFromImage(resized_image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Define the confidence threshold and NMS threshold
    confidence_threshold = 0.5
    nms_threshold = 0.4

    # Process the output detections
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Get the class label
                class_label = classes[class_id]

                # Check if the class label is a living thing, vehicle, or plant
                if class_label in living_thing_classes or class_label in vehicle_classes or class_label in plant_classes:
                    # Get the bounding box coordinates
                    center_x = int(detection[0] * desired_width)
                    center_y = int(detection[1] * desired_height)
                    bbox_width = int(detection[2] * desired_width)
                    bbox_height = int(detection[3] * desired_height)

                    # Calculate the top-left corner coordinates of the bounding box
                    x = int(center_x - (bbox_width / 2))
                    y = int(center_y - (bbox_height / 2))

                    # Scale the bounding box coordinates to match the original image size
                    x = int(x * (width / desired_width))
                    y = int(y * (height / desired_height))
                    bbox_width = int(bbox_width * (width / desired_width))
                    bbox_height = int(bbox_height * (height / desired_height))

                    # Store the bounding box, confidence, and class ID
                    boxes.append([x, y, bbox_width, bbox_height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Ensure at least one detection exists
    if len(indices) > 0:
        indices = indices.flatten()

        # Draw the filtered bounding boxes and labels on the image
        for i in indices:
            (x, y, w, h) = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]

            # Determine the color based on the class (living thing, vehicle, or plant)
            if classes[class_id] in living_thing_classes:
                color = (0, 255, 0)  # Green color for living things
            elif classes[class_id] in vehicle_classes:
                color = (255, 0, 0)  # Blue color for vehicles
            else:
                color = (0, 0, 255)  # Red color for plants

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Resize the output image to 600x600
    resized_output_image = cv2.resize(image, (600, 600))

    # Display the output image
    cv2.imshow("Object Detection", resized_output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Create the GUI window
    window = tk.Tk()
    window.title("Object Detection")

    def browse_image():
        # Prompt the user to select an image file
        image_path = select_image()

        # Perform object detection on the selected image
        if image_path:
            perform_object_detection(image_path)

    # Create a "Browse" button
    browse_button = tk.Button(window, text="Browse Image", command=browse_image)
    browse_button.pack(pady=10)

    # Start the GUI event loop
    window.mainloop()

if __name__ == "__main__":
    main()
