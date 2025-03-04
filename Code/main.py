import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")  # Suppress deprecation warnings

import cv2
import numpy as np
from transformers import pipeline

# Initialize the VQA pipeline with a BLIP model
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# Variables for drawing and detection
drawing = False
points = []
roi = None  # Store the extracted ROI
question_mode = False  # Flag to indicate if we're in question mode

# Mouse callback function for freehand drawing
def draw_freehand(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # Stop drawing, trigger detection

# Initialize camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('Circle an Object and Ask Questions')
cv2.setMouseCallback('Circle an Object and Ask Questions', draw_freehand)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Draw the freehand shape while the user is drawing
    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, np.int32)], False, (255, 0, 0), 2)

    # Extract the ROI when drawing stops and no ROI has been extracted yet
    if not drawing and len(points) > 2 and roi is None:
        # Create a mask for the region of interest (ROI)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, np.int32)], 255)

        # Extract the ROI
        roi = cv2.bitwise_and(frame, frame, mask=mask)

        # Show the ROI (for debugging)
        cv2.imshow('ROI', roi)

        # Save the ROI as an image file (required for the VQA model)
        roi_path = "roi.jpg"
        cv2.imwrite(roi_path, roi)

        # Enter question mode
        question_mode = True

    # If in question mode, allow the user to ask questions
    if question_mode:
        # Ask the user for a question
        question = input("Ask a question about the circled object (or type 'reset' to draw a new region, 'quit' to exit): ")

        if question.lower() == 'quit':
            break
        elif question.lower() == 'reset':
            # Reset the state to allow drawing a new region
            points = []
            roi = None
            question_mode = False
            print("Region reset. Draw a new region.")
            continue

        # Use the VQA model to answer the question
        try:
            answer = vqa_pipeline(roi_path, question)
            print(f"Answer: {answer[0]['answer']}")
        except Exception as e:
            print(f"Error: {e}")

    # Show the frame
    cv2.imshow('Circle an Object and Ask Questions', frame)

    # Controls: Quit ('q')
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()