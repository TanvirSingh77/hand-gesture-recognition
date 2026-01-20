# 🖐️ Real-Time Hand Gesture Recognition System

## 📌 Overview
This project is a **real-time hand gesture recognition system** built using **Computer Vision and Machine Learning** techniques.  
The system captures live webcam video, detects hand landmarks, extracts geometric features, and performs gesture recognition in real time with low latency.

The project focuses on building an **end-to-end, industry-style pipeline**, optimized for real-time performance and designed with a clean, modular architecture.

---

## 🚀 Features
- Real-time webcam-based hand detection
- Robust hand landmark extraction (21 key points)
- Feature extraction using geometric relationships
- Gesture classification using a lightweight ML model
- Optimized inference using **TensorFlow Lite**
- FPS monitoring for performance analysis
- Modular and extensible project structure

---

## 🧠 Architecture
Webcam Input
↓
Frame Preprocessing (OpenCV)
↓
Hand Detection & Landmark Extraction (MediaPipe)
↓
Feature Extraction (Distances & Angles)
↓
Feature Normalization
↓
Gesture Classification Model
↓
TensorFlow Lite Inference
↓
Real-time Visualization & Output


---

## 🔍 Methodology
- Live video frames are captured using OpenCV
- Hand landmarks are detected using MediaPipe
- A total of **21 hand keypoints** are extracted per frame
- Geometric features such as joint distances and finger angles are computed
- Features are normalized and passed to a trained gesture classification model
- TensorFlow Lite is used for fast, CPU-optimized inference
- Predicted gestures and FPS are displayed in real time

---

## 🛠️ Tech Stack
- **Python 3.10**
- **MediaPipe** – Hand landmark detection
- **OpenCV** – Real-time video capture and visualization
- **TensorFlow / TensorFlow Lite** – Gesture classification and optimized inference
- **NumPy** – Feature computation and numerical operations

---

## 🧪 Example Use Cases
- Touchless user interfaces
- Gesture-controlled applications
- Human–computer interaction (HCI)
- Smart systems and assistive technologies
- Educational demos for computer vision

---

## 📊 Performance
- Designed for **real-time execution on CPU**
- Optimized for low-latency inference
- Achieves stable FPS during live webcam processing
- Accuracy depends on gesture set and training data

---

## ⚠️ Limitations
- Currently supports **single-hand detection**
- Limited set of predefined gestures
- Static gestures only (no temporal modeling yet)

---

## 🔮 Future Improvements
- Add support for dynamic gestures using temporal models (LSTM)
- Improve accuracy with larger datasets
- Multi-hand gesture recognition
- Map gestures to real-world actions (volume control, slides, etc.)
- Deploy as a desktop or web application

---

## 🎓 Learning Outcomes
This project helped me understand:
- Real-time computer vision pipelines
- Hand landmark-based gesture representation
- Feature engineering for gesture recognition
- Model optimization using TensorFlow Lite
- Writing clean, modular, and testable ML code

---

## 👤 Author
**Tanvir Singh**

---

## 📜 Disclaimer
This project is built for **learning, experimentation, and portfolio purposes** and demonstrates practical implementation of real-time hand gesture recognition systems.
