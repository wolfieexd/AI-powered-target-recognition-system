import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import time

app = ctk.CTk()
app.geometry("800x600")

live_frame = ctk.CTkFrame(app)
live_frame.pack(fill=tk.BOTH, expand=True)

live_frame.columnconfigure(0, weight=1)
live_frame.rowconfigure(0, weight=1)

main_frame = ttk.Frame(live_frame)
main_frame.grid(row=0, column=0, sticky="nsew")
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed")
video_frame.grid(row=0, column=0, sticky="nsew")
video_frame.columnconfigure(0, weight=1)
video_frame.rowconfigure(0, weight=1)

video_label = ttk.Label(video_frame, background='black')
video_label.pack(fill=tk.BOTH, expand=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def update():
    ret, frame = cap.read()
    if ret:
        print("Got frame!")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.configure(image=imgtk)
        video_label.image = imgtk
    else:
        print("No frame")
    app.after(33, update)

update()
app.after(3000, app.destroy)
app.mainloop()
