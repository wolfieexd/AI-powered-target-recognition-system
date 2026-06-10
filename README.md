# 🎯 AI-Powered Target Recognition System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLOv8-Powered-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#license)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Jetson%20Nano-brightgreen)]()

An **AI-powered real-time surveillance system** that detects and recognizes potential threats using YOLO-based deep learning models. Designed for use in smart security systems, drones, and defense setups.

---

## 🎥 Sample Detection (JPG Preview)

<p align="center">
  <img src="https://github.com/wolfieexd/AI-powered-target-recognition-system/blob/main/Picture1.jpg?raw=true" alt="Picture1" />
</p>

> *Above: The system detects and highlights objects like people or weapons in real-time.*

---

## 🔥 Features

- 🚀 Real-time object detection using YOLOv8 and custom YOLO11L
- 📸 Webcam and video file input support
- 🔍 Heatmap and motion tracking capabilities *(customizable)*
- 🧠 Pre-trained & custom-trained model support
- 💾 Local SQLite logging of detections
- 📂 Clean modular folder structure

---

## 🗂 Folder Structure

```
AI-Powered Target Recognition System/
├── detections.db # SQLite database
├── error.log # Log file
├── Requirements.txt # Dependencies
├── models/
|   ├── yolo11l.pt # Custom YOLOv11 model
|   └── yolov8n.pt # YOLOv8 nano model
├── main_dashboard.py # Core interactive UI and dashboard
├── file_weapon_detector.py # File analysis and active suspects tracking
└── Documentations/
    ├── *.pptx # Project presentation
    └── *.pdf # Final documentation
```

---

## ⚙️ Installation

> 🐍 Requires **Python 3.10+**

Clone the repository:
```bash
git clone https://github.com/wolfieexd/AI-powered-target-recognition-system
cd AI-powered-target-recognition-system
```

Install the required Python packages:
```bash
pip install -r Requirements.txt
```

Run the detection system dashboard:
```bash
python main_dashboard.py
```
## 🧠 Models
yolov8n.pt — Lightweight YOLOv8 model  
yolo11l.pt — Custom-trained model for high-accuracy object detection

✅ You can swap out models in the script to suit performance/accuracy needs.

## 🧰 Tech Stack
Python 3.10  
OpenCV for video processing   
YOLOv8 / Custom YOLO11L (via PyTorch)  
SQLite for threat logging  
Ultralytics for model support  

## 📄 Documentation
Find all reports, presentation decks, and research materials in the /Documentations folder.

## 👨‍💻 Author

Sujan S  
🎓 SRM Institute of Science and Technology  
📧 [sujans1411@gmail.com]  
🔗 [Portfolio](https://wolfieexd.github.io/portfolio/)

## 📌 License
This project is intended for academic and research purposes. For commercial use, please contact the author.

## ⚠️ Ensure your system has a GPU for better performance during detection. This system is built for prototype and educational demonstrations.
