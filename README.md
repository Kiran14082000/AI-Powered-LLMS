# 🧠 What-sCooken.AI - Chat with Objects

> A multimodal AI application that lets you **hover over real-world objects**, select them with a key press, and **ask questions** about them using powerful image understanding and language models.

## 🔍 Key Features

- 🎯 **Object Detection** with YOLOv3
- 📷 **iPhone Continuity Camera** integration for real-time video
- 🖱️ **Hover & Select** mechanism to choose an object (press `A`)
- 🧠 **Visual Reasoning** using BLIP captioning + FLAN-T5 for Q&A
- 🎨 **Color detection** with KMeans
- 📝 **Text extraction** via Tesseract OCR
- 😊 **Facial Expression Recognition** using DeepFace
- 🍳 **Recipe Generation** using Spoonacular API

## 💻 How It Works

1. **Start the app**
2. **Use your webcam or iPhone camera**
3. **Hover over an object**, then press `A` to select
4. **Chat in a custom UI** to ask about:
   - What the object is
   - What it's used for
   - Recipes using the object (if it's food)
   - Its color, text, emotion, etc.

## 📦 Tech Stack

| Type            | Stack                                  |
|-----------------|----------------------------------------|
| Object Detection| YOLOv3 + OpenCV                        |
| Image Captioning| Salesforce BLIP                        |
| Q&A             | Google FLAN-T5                         |
| Color Analysis  | KMeans Clustering                      |
| OCR             | Tesseract                              |
| Expression      | DeepFace                               |
| UI              | CustomTkinter (desktop) / Streamlit    |
| Recipe API      | [Spoonacular](https://spoonacular.com) |

## 🚀 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/ai-vision-assistant.git
   cd ai-vision-assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   conda create -n vision-assistant python=3.9
   conda activate vision-assistant
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv3 Weights**
   - Download `yolov3.weights` from the official source and place in the project directory.

5. **Add your Spoonacular API Key**
   - Inside `main.py` or `.env`, replace the key with your own:
     ```python
     SPOONACULAR_API_KEY = "your_api_key_here"
     ```

6. **Run the app**
   ```bash
   python main.py
   ```

## 📸 iPhone Continuity Camera

- On macOS, use **your iPhone as a webcam** by selecting it in the camera input window.
- Ensure Bluetooth and Wi-Fi are enabled and both devices are signed in with the same Apple ID.

## 🧪 Demo

> Coming soon — video demo & Hugging Face Spaces live app.
![image](https://github.com/user-attachments/assets/83d0283a-e7b5-4ee8-a683-fa2308e24b62)


## 🙌 Credits

- [YOLOv3](https://pjreddie.com/darknet/yolo/)
- [Hugging Face Transformers](https://huggingface.co)
- [Spoonacular API](https://spoonacular.com/food-api)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Streamlit](https://streamlit.io)

## 📜 License

This project is open-source under the [MIT License](LICENSE).

## 🤖 Built with passion by 
## 👥 Collaborators

- [Kiran Gobi Manivannan](https://github.com/Kiran14082000)
- [Gowtham SR](https://github.com/Gowtham-sr)
- [Skowshi](https://github.com/skowshi)

