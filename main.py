import os
import cv2
import numpy as np
import customtkinter as ctk
from tkinter import messagebox
from transformers import pipeline
from sklearn.cluster import KMeans
import pytesseract
from deepface import DeepFace
import threading
import time
import re
import requests
import string
from html import unescape

# === API Key for Spoonacular ===
SPOONACULAR_API_KEY = "bc1a1d0fd422407baf9adc2d2e2566a0"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Paths ===
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
coco_names_path = "coco.names"
roi_path = "roi.jpg"

# === Load YOLO ===
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

caption_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

selected_object = None
hovered_object = None
stored_caption = None
detected_objects = []
camera_index = 0

# === Helpers ===
def detect_dominant_color(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, n_init=10).fit(image)
    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return f"RGB({int(dominant[0])}, {int(dominant[1])}, {int(dominant[2])})"

def detect_text_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray).strip()

def detect_expression(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception:
        return "No face/expression"

def strip_html_tags(text):
    clean = re.compile('<.*?>')
    return unescape(re.sub(clean, '', text))

def get_recipe_from_ingredient(ingredient):
    try:
        ingredient = ingredient.strip().lower().strip(string.punctuation)
        url = "https://api.spoonacular.com/recipes/findByIngredients"
        params = {
            "ingredients": ingredient,
            "number": 1,
            "ranking": 2,
            "ignorePantry": True,
            "apiKey": SPOONACULAR_API_KEY
        }
        res = requests.get(url, params=params)
        data = res.json()
        if not data:
            return f"Sorry, I couldn’t find a recipe using '{ingredient}'."
        recipe_id = data[0]['id']
        info = requests.get(f"https://api.spoonacular.com/recipes/{recipe_id}/information",
                            params={"apiKey": SPOONACULAR_API_KEY}).json()
        ingredients = [i['original'] for i in info.get('extendedIngredients', [])]
        instructions = strip_html_tags(info.get('instructions', 'No instructions available.'))
        return f"🔍 **Using ingredient: {ingredient}**\n\n🍽️ **{data[0]['title']}**\n\n**Ingredients:**\n" + \
               "\n".join(f"- {i}" for i in ingredients) + f"\n\n**Instructions:**\n{instructions.strip()}"
    except Exception as e:
        return f"❌ Error getting recipe: {str(e)}"

# === Camera Selection ===
def select_camera_index():
    global camera_index
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]: available.append(i)
        cap.release()

    def set_index(idx):
        nonlocal root
        global camera_index
        camera_index = idx
        root.destroy()

    root = ctk.CTk()
    root.geometry("360x300")
    root.title("Select Camera")

    ctk.CTkLabel(root, text="Choose Camera Input", font=("Arial", 16)).pack(pady=15)
    for idx in available:
        ctk.CTkButton(root, text=f"Camera Index {idx}", command=lambda i=idx: set_index(i)).pack(pady=8)

    root.mainloop()

# === Chat UI ===
class ChatBotUI(ctk.CTkToplevel):
    def __init__(self, obj_name, roi_path):
        super().__init__()
        self.title("AI Vision Assistant")
        self.geometry("620x600")
        self.obj_name = obj_name
        self.roi_path = roi_path

        self.chat_log = ctk.CTkTextbox(self, wrap="word", font=("Arial", 13))
        self.chat_log.pack(padx=10, pady=10, fill="both", expand=True)

        self.entry = ctk.CTkEntry(self, placeholder_text="Ask a question...")
        self.entry.pack(padx=10, pady=(0, 5), fill="x")
        self.entry.bind("<Return>", lambda e: self.send_message())

        self.send_btn = ctk.CTkButton(self, text="Send", command=self.send_message)
        self.send_btn.pack(pady=5)

        self.suggestion_frame = ctk.CTkFrame(self)
        self.suggestion_frame.pack(pady=5, fill="x")
        for q in ["Give me a recipe", "What can I make with this?", "What's it used for?"]:
            ctk.CTkButton(self.suggestion_frame, text=q, command=lambda t=q: self.fill_suggestion(t)).pack(pady=2)

        global stored_caption
        if not stored_caption and os.path.exists(self.roi_path):
            stored_caption = caption_model(self.roi_path)[0]['generated_text']
        self.chat_log.insert("end", "🤖 Bot: Hi there! How can I help you today?\n")

    def fill_suggestion(self, text):
        self.entry.delete(0, "end")
        self.entry.insert(0, text)

    def send_message(self):
        msg = self.entry.get()
        if not msg.strip(): return
        self.entry.delete(0, "end")
        self.chat_log.insert("end", f"\n🧑 You: {msg}\n")
        threading.Thread(target=self.get_response, args=(msg,)).start()

    def type_response(self, message):
        self.chat_log.insert("end", "🤖 Bot: ")
        for char in message:
            self.chat_log.insert("end", char)
            self.chat_log.see("end")
            self.update()
            time.sleep(0.01)
        self.chat_log.insert("end", "\n")

    def get_response(self, question):
        global stored_caption
        roi = cv2.imread(self.roi_path) if os.path.exists(self.roi_path) else None
        color = detect_dominant_color(roi) if roi is not None else "Unknown"
        text = detect_text_in_image(roi) if roi is not None else "None"
        expression = detect_expression(roi) if roi is not None else "Unknown"
        context = stored_caption or "No image description available"

        try:
            if "recipe" in question.lower() or "cook" in question.lower():
                banned = {"give", "me", "a", "recipe", "for", "cook", "with", "this", "the", "how", "to", "something", "please", "i", "want"}
                words = question.lower().split()
                food_words = [w for w in words if w not in banned and len(w) > 2 and not w.isdigit()]
                ingredient = food_words[-1] if food_words else self.obj_name or "ingredient"
                reply = get_recipe_from_ingredient(ingredient)
            else:
                prompt = (
                    f"You are looking at an object identified as: '{self.obj_name}'. "
                    f"A visual description of the region shows: {context}. "
                    f"The object's color is: {color}. "
                    f"Visible text (if any): {text}. "
                    f"Facial expression (if any): {expression}. "
                    f"Now answer this question about the object: {question}"
                )
                result = qa_model(prompt, max_new_tokens=100)[0]['generated_text']
                reply = result.strip().capitalize() if result.strip() else "I couldn't find an answer. Try asking differently."
        except Exception as e:
            reply = f"❌ Error generating answer: {str(e)}"
        self.type_response(reply)

# === Core Functions ===
def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    global detected_objects
    detected_objects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detected_objects.append((x, y, w, h, classes[class_id]))

def open_chat(frame):
    global selected_object, stored_caption
    obj_name = None
    if selected_object:
        x, y, w, h, detected_name = selected_object
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            cv2.imwrite(roi_path, roi)
            stored_caption = None
            obj_name = detected_name
        else:
            messagebox.showwarning("Warning", "Invalid region detected.")
    ChatBotUI(obj_name, roi_path).mainloop()

def mouse_event(event, x, y, flags, param):
    global hovered_object
    hovered_object = None
    for obj in detected_objects:
        ox, oy, ow, oh, name = obj
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            hovered_object = obj
            return

# === Launch ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")
select_camera_index()

cap = cv2.VideoCapture(camera_index)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    detect_objects(frame)
    frame_copy = frame.copy()

    for obj in detected_objects:
        x, y, w, h, name = obj
        color = (0, 255, 255) if obj == hovered_object else (255, 0, 0)
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
        if obj == hovered_object:
            roi = frame[y:y+h, x:x+w]
            info = [
                f"Object: {name}",
                f"Color: {detect_dominant_color(roi)}",
                f"Text: {detect_text_in_image(roi)[:30]}",
                f"Emotion: {detect_expression(roi)}"
            ]
            for i, tip in enumerate(info):
                cv2.putText(frame_copy, tip, (x, y - 10 - i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("AI Vision Assistant", frame_copy)
    cv2.setMouseCallback("AI Vision Assistant", mouse_event)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        selected_object = hovered_object
        open_chat(frame)

cap.release()
cv2.destroyAllWindows()
