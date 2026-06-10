-import cv2  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # For displaying OpenCV frames in Tkinter
import sqlite3
from datetime import datetime
import logging  # Import logging for error handling
import threading
import queue

# Set up logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

# Function to log detection
def log_detection(conn, camera_index, object_class, confidence, weapon_detected):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c = conn.cursor()
    c.execute('''INSERT INTO detections 
                 (timestamp, camera_index, object_class, confidence, weapon_detected)
                 VALUES (?, ?, ?, ?, ?)''',
              (timestamp, camera_index, object_class, confidence, int(weapon_detected)))
    conn.commit()

# Load YOLO model on GPU
try:
    model = YOLO("models/yolov5x6u.pt").to("cuda")  # Load the new YOLOv5 model on GPU
except Exception as e:
    logging.error(f"Model loading error: {e}")
    messagebox.showerror("Model Error", "Failed to load the YOLO model.")

# Thread-safe queue for frames
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# Function to draw bounding boxes
def draw_detections(frame, results):
    global detection_count, weapon_count
    weapon_detected = False
    detection_count_local = 0
    weapon_count_local = 0
    
    logging.debug(f"Number of results: {len(results)}")
    
    for result in results:
        # Access the boxes from the result
        boxes = result.boxes
        logging.debug(f"Number of boxes: {len(boxes)}")
        for box in boxes:
            # Get the confidence score and move it to CPU
            conf = box.conf.cpu().item()  # Get the confidence as a Python scalar
            logging.debug(f"Box confidence: {conf}, Threshold: {threshold_slider.get()}")
            if conf > threshold_slider.get():  # Use the threshold slider value
                detection_count_local += 1
                # Get the bounding box coordinates and move them to CPU
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Ensure to access the first element

                object_class = result.names[int(box.cls)]
                label = f"{object_class} ({conf:.2f})"  # Get the class name and confidence

                # Check for weapons and highlight them
                if object_class in ["knife", "gun"]:
                    color = (0, 0, 255)  # Red for weapons
                    weapon_detected = True
                    weapon_count_local += 1
                    # Log weapon detection instead of blocking messagebox
                    logging.warning(f"Weapon detected: {object_class} with confidence {conf:.2f}")
                else:
                    color = (0, 255, 0)  # Green for other objects

                # Log the detection
                log_detection(db_conn, current_camera_index, object_class, conf, weapon_detected)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    detection_count = detection_count_local
    weapon_count = weapon_count_local

    return frame, weapon_detected

# Function to apply pseudo-thermal effect
def apply_thermal_effect(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a thermal-like colormap (e.g., COLORMAP_JET or COLORMAP_HOT)
    thermal_frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return thermal_frame

# Function to start/stop recording
def toggle_recording():
    global recording, out
    if not recording:
        # Start recording
        recording = True
        record_button.config(text="Stop Recording")
        # Initialize video writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            f"output/recording_{time.strftime('%Y%m%d_%H%M%S')}.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (frame_width, frame_height),
        )
    else:
        # Stop recording
        recording = False
        record_button.config(text="Start Recording")
        if out is not None:
            out.release()
            out = None

def detection_thread_func():
    global cap, recording, out, thermal_mode
    while True:
        try:
            frame = frame_queue.get()
            if frame is None:
                break  # Exit signal

            # Resize the frame
            frame_resized = cv2.resize(frame, (640, 480))

            # Apply pseudo-thermal effect if enabled
            if thermal_mode:
                frame_resized = apply_thermal_effect(frame_resized)

            # Perform object detection
            results = model(frame_resized)

            # Draw detections on the frame
            frame_processed, weapon_detected = draw_detections(frame_resized, results)

            # Save the frame if recording
            if recording and out is not None:
                out.write(frame_processed)

            # Put the processed frame and detection info in result queue
            if not result_queue.full():
                result_queue.put((frame_processed, weapon_detected, results))
            else:
                logging.debug("Result queue full, skipping frame")
        except Exception as e:
            logging.error(f"Error in detection thread: {e}")

# Function to update the video feed in the Tkinter GUI
def update_video_feed():
    global cap

    try:
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Error: Could not read frame from camera at index {current_camera_index}.")
            messagebox.showerror("Camera Error", f"Could not read frame from camera at index {current_camera_index}.")
            return

        # Put the frame into the frame queue for detection thread
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get the processed frame from the result queue
        if not result_queue.empty():
            frame_processed, weapon_detected, results = result_queue.get()

            # Convert the frame to RGB format for Tkinter
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label with the new image
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            # Update detection statistics
            update_statistics(results)
        else:
            # If no processed frame available, show raw frame as fallback
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        video_label.after(10, update_video_feed)  # Update every 10 ms
    except Exception as e:
        logging.error(f"Error in update_video_feed: {e}")

# Database setup
def init_db():
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT NOT NULL,
                      camera_index INTEGER NOT NULL,
                      object_class TEXT NOT NULL,
                      confidence REAL NOT NULL,
                      weapon_detected INTEGER NOT NULL)''')
        conn.commit()
        return conn
    except Exception as e:
        logging.error(f"Database initialization error: {e}")
        messagebox.showerror("Database Error", "Failed to initialize the database.")

# Initialize database
db_conn = init_db()

# Initialize video capture for multiple cameras
cameras = [
    0,  # Index of the built-in webcam as primary camera
    "https://192.168.0.100:8080/video"  # Mobile camera stream URL as secondary camera
]
current_camera_index = 0  # Start with the built-in webcam
try:
    cap = cv2.VideoCapture(cameras[current_camera_index])
    if not cap.isOpened():
        raise Exception(f"Could not open camera at index {current_camera_index}.")
except Exception as e:
    logging.error(f"Camera initialization error: {e}")
    messagebox.showerror("Camera Error", f"Could not open camera at index {current_camera_index}. The application will continue without video feed.")
    cap = None  # Set cap to None to avoid further errors

# Initialize motion detection using a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# Global variables
thermal_mode = False  # Default to normal mode
recording = False  # Flag to indicate if recording is active
out = None  # Video writer object

# Function to toggle thermal mode
def toggle_thermal_mode():
    global thermal_mode
    thermal_mode = not thermal_mode
    thermal_button.config(text="Switch to Normal Mode" if thermal_mode else "Switch to Thermal Mode")

# Function to switch cameras
def switch_camera():
    global current_camera_index, cap
    current_camera_index = (current_camera_index + 1) % len(cameras)  # Cycle through cameras
    cap.release()  # Release the current camera
    cap = cv2.VideoCapture(cameras[current_camera_index])  # Open the new camera
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {current_camera_index}.")
        return

# Create the main Tkinter window
root = tk.Tk()
root.title("Video Stream with YOLO Detection")

# Create a label to display the video feed
video_label = ttk.Label(root)
video_label.pack()

# Create a menu bar and status bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create a frame for statistics
stats_frame = ttk.Frame(root)
stats_frame.pack(pady=10)

# Create labels for detection statistics
detection_label = ttk.Label(stats_frame, text="Detections: 0")
detection_label.pack(side=tk.LEFT, padx=5)

weapon_label = ttk.Label(stats_frame, text="Weapons: 0")
weapon_label.pack(side=tk.LEFT, padx=5)

confidence_label = ttk.Label(stats_frame, text="Avg Confidence: 0.00")
confidence_label.pack(side=tk.LEFT, padx=5)

# Create a frame for interactive controls
control_frame = ttk.Frame(root)
control_frame.pack(pady=10)

# Create a slider for detection threshold
threshold_label = ttk.Label(control_frame, text="Detection Threshold:")
threshold_label.pack(side=tk.LEFT, padx=5)

threshold_slider = ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, length=200)
threshold_slider.set(0.5)  # Default threshold value
threshold_slider.pack(side=tk.LEFT, padx=5)

# Function to update detection statistics
def update_statistics(results):
    detection_label.config(text=f"Detections: {detection_count}")
    weapon_label.config(text=f"Weapons: {weapon_count}")
    if detection_count > 0:
        avg_confidence = sum([box.conf.cpu().item() for box in results[0].boxes]) / detection_count
    else:
        avg_confidence = 0
    confidence_label.config(text=f"Avg Confidence: {avg_confidence:.2f}")
    root.after(1000, lambda: update_statistics(results))  # Update every second

# Create settings menu
settings_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=settings_menu)
settings_menu.add_command(label="Camera Settings", command=lambda: messagebox.showinfo("Info", "Camera settings not implemented yet"))
settings_menu.add_command(label="Detection Threshold", command=lambda: messagebox.showinfo("Info", "Detection threshold settings not implemented yet"))

# Create help menu
help_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "AI-Powered Target Recognition System\nVersion 1.0"))

# Create status bar
status_bar = ttk.Label(root, text="FPS: 0 | Detections: 0 | Weapons: 0", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Performance metrics variables
fps = 0
detection_count = 0
weapon_count = 0
last_time = time.time()  # Initialize last_time to the current time

# Function to update performance metrics
def update_metrics():
    global fps, detection_count, weapon_count, last_time
    current_time = time.time()
    elapsed_time = current_time - last_time
    if elapsed_time > 0:  # Ensure no division by zero
        fps = 1 / elapsed_time
    else:
        fps = 0
    last_time = current_time
    status_bar.config(text=f"FPS: {fps:.1f} | Detections: {detection_count} | Weapons: {weapon_count}")
    root.after(1000, update_metrics)

# Create buttons for toggling thermal mode, switching cameras, and recording
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

thermal_button = ttk.Button(button_frame, text="Switch to Thermal Mode", command=toggle_thermal_mode)
thermal_button.pack(side=tk.LEFT, padx=5)

switch_button = ttk.Button(button_frame, text="Switch Camera", command=switch_camera)
switch_button.pack(side=tk.LEFT, padx=5)

record_button = ttk.Button(button_frame, text="Start Recording", command=toggle_recording)
record_button.pack(side=tk.LEFT, padx=5)

# Start detection thread
detection_thread = threading.Thread(target=detection_thread_func, daemon=True)
detection_thread.start()

# Start performance metrics update
update_metrics()

# Start the video feed
update_video_feed()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when done
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# Stop detection thread
frame_queue.put(None)
detection_thread.join()

# Function to toggle thermal mode
def toggle_thermal_mode():
    global thermal_mode
    thermal_mode = not thermal_mode
    thermal_button.config(text="Switch to Normal Mode" if thermal_mode else "Switch to Thermal Mode")

# Function to switch cameras
def switch_camera():
    global current_camera_index, cap
    current_camera_index = (current_camera_index + 1) % len(cameras)  # Cycle through cameras
    cap.release()  # Release the current camera
    cap = cv2.VideoCapture(cameras[current_camera_index])  # Open the new camera
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {current_camera_index}.")
        return

# Create the main Tkinter window
root = tk.Tk()
root.title("Video Stream with YOLO Detection")

# Create a label to display the video feed
video_label = ttk.Label(root)
video_label.pack()

# Create a menu bar and status bar
menubar = tk.Menu(root)
root.config(menu=menubar)

# Create a frame for statistics
stats_frame = ttk.Frame(root)
stats_frame.pack(pady=10)

# Create labels for detection statistics
detection_label = ttk.Label(stats_frame, text="Detections: 0")
detection_label.pack(side=tk.LEFT, padx=5)

weapon_label = ttk.Label(stats_frame, text="Weapons: 0")
weapon_label.pack(side=tk.LEFT, padx=5)

confidence_label = ttk.Label(stats_frame, text="Avg Confidence: 0.00")
confidence_label.pack(side=tk.LEFT, padx=5)

# Create a frame for interactive controls
control_frame = ttk.Frame(root)
control_frame.pack(pady=10)

# Create a slider for detection threshold
threshold_label = ttk.Label(control_frame, text="Detection Threshold:")
threshold_label.pack(side=tk.LEFT, padx=5)

threshold_slider = ttk.Scale(control_frame, from_=0, to=1, orient=tk.HORIZONTAL, length=200)
threshold_slider.set(0.5)  # Default threshold value
threshold_slider.pack(side=tk.LEFT, padx=5)

# Function to update detection statistics
def update_statistics(results):
    detection_label.config(text=f"Detections: {detection_count}")
    weapon_label.config(text=f"Weapons: {weapon_count}")
    if detection_count > 0:
        avg_confidence = sum([box.conf.cpu().item() for box in results[0].boxes]) / detection_count
    else:
        avg_confidence = 0
    confidence_label.config(text=f"Avg Confidence: {avg_confidence:.2f}")
    root.after(1000, lambda: update_statistics(results))  # Update every second

# Create settings menu
settings_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=settings_menu)
settings_menu.add_command(label="Camera Settings", command=lambda: messagebox.showinfo("Info", "Camera settings not implemented yet"))
settings_menu.add_command(label="Detection Threshold", command=lambda: messagebox.showinfo("Info", "Detection threshold settings not implemented yet"))

# Create help menu
help_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "AI-Powered Target Recognition System\nVersion 1.0"))

# Create status bar
status_bar = ttk.Label(root, text="FPS: 0 | Detections: 0 | Weapons: 0", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Performance metrics variables
fps = 0
detection_count = 0
weapon_count = 0
last_time = time.time()  # Initialize last_time to the current time

# Function to update performance metrics
def update_metrics():
    global fps, detection_count, weapon_count, last_time
    current_time = time.time()
    elapsed_time = current_time - last_time
    if elapsed_time > 0:  # Ensure no division by zero
        fps = 1 / elapsed_time
    else:
        fps = 0
    last_time = current_time
    status_bar.config(text=f"FPS: {fps:.1f} | Detections: {detection_count} | Weapons: {weapon_count}")
    root.after(1000, update_metrics)

# Create buttons for toggling thermal mode, switching cameras, and recording
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

thermal_button = ttk.Button(button_frame, text="Switch to Thermal Mode", command=toggle_thermal_mode)
thermal_button.pack(side=tk.LEFT, padx=5)

switch_button = ttk.Button(button_frame, text="Switch Camera", command=switch_camera)
switch_button.pack(side=tk.LEFT, padx=5)

record_button = ttk.Button(button_frame, text="Start Recording", command=toggle_recording)
record_button.pack(side=tk.LEFT, padx=5)

# Start performance metrics update
update_metrics()

# Start the video feed
update_video_feed()

# Start the Tkinter main loop
root.mainloop()

# Release the video capture when done
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
