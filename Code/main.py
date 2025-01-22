import cv2
import numpy as np

# Paths to the model configuration, weights, and labels
model_cfg = 'yolov3.cfg'         # Path to the YOLO configuration file
model_weights = 'yolov3.weights' # Path to the YOLO weights file
coco_names_path = 'coco.names'   # Path to the COCO class labels file

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# Ensure all files are accessible
import os
assert os.path.exists(model_cfg), "Configuration file not found!"
assert os.path.exists(model_weights), "Weights file not found!"
assert os.path.exists(coco_names_path), "COCO names file not found!"

# Read COCO class labels
with open(coco_names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Get YOLO output layer names
layer_names = net.getLayerNames()

# Handle OpenCV version differences for getUnconnectedOutLayers
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the camera or video feed
cap = cv2.VideoCapture(0)  # Use '0' for webcam or replace with video file path
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Processing the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Resize the frame and convert to blob for YOLO
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through YOLO
    detections = net.forward(output_layers)

    # Parsing detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]  # Class confidence scores
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Filter out weak detections
                # Object bounding box
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
