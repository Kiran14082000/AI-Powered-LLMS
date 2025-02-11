import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from transformers import pipeline

# Disable TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize the VQA pipeline
vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")

# Initialize camera
cap = cv2.VideoCapture(0)
selected_roi = None  # Store ROI for processing

def capture_frame():
    """Captures a frame from the camera."""
    global selected_roi
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def select_roi():
    """Allows the user to select an object (ROI) using OpenCV's selectROI function."""
    global selected_roi
    frame = capture_frame()
    if frame is None:
        return
    roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")
    if roi != (0, 0, 0, 0):  # If a region was selected
        x, y, w, h = roi
        selected_roi = frame[y:y+h, x:x+w]
        display_selected_roi()

def display_selected_roi():
    """Displays the selected ROI in the Tkinter UI."""
    if selected_roi is None:
        return
    img = cv2.cvtColor(selected_roi, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    roi_label.config(image=img_tk)
    roi_label.image = img_tk  # Keep reference to avoid garbage collection

def ask_question():
    """Processes the selected ROI with the user's question."""
    global selected_roi
    if selected_roi is None:
        result_label.config(text="Please select an object first.")
        return
    question = question_entry.get()
    if not question:
        result_label.config(text="Enter a question.")
        return
    
    # Save ROI as an image for processing
    roi_path = "roi.jpg"
    cv2.imwrite(roi_path, selected_roi)
    
    # Run VQA in a separate thread to prevent UI freezing
    def vqa_thread():
        try:
            answer = vqa_pipeline(roi_path, question)
            result_label.config(text=f"Answer: {answer[0]['answer']}")
        except Exception as e:
            result_label.config(text=f"Error: {e}")
    threading.Thread(target=vqa_thread, daemon=True).start()

def reset_selection():
    """Resets the selection and clears the UI."""
    global selected_roi
    selected_roi = None
    roi_label.config(image="")
    result_label.config(text="")
    question_entry.delete(0, tk.END)

def close_app():
    """Closes the application and releases the camera."""
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# Create Tkinter UI
root = tk.Tk()
root.title("AI Object Detection & Question Answering")

# UI Elements
frame_top = tk.Frame(root)
frame_top.pack()

select_button = tk.Button(frame_top, text="Select Object", command=select_roi)
select_button.pack(side=tk.LEFT, padx=10, pady=5)

question_entry = tk.Entry(frame_top, width=50)
question_entry.pack(side=tk.LEFT, padx=10, pady=5)

ask_button = tk.Button(frame_top, text="Ask", command=ask_question)
ask_button.pack(side=tk.LEFT, padx=10, pady=5)

roi_label = tk.Label(root)
roi_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

button_frame = tk.Frame(root)
button_frame.pack()

reset_button = tk.Button(button_frame, text="Reset", command=reset_selection)
reset_button.pack(side=tk.LEFT, padx=10, pady=5)

quit_button = tk.Button(button_frame, text="Quit", command=close_app)
quit_button.pack(side=tk.LEFT, padx=10, pady=5)

# Run Tkinter event loop
root.mainloop()
