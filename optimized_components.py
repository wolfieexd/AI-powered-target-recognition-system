import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
from datetime import datetime
import logging
import time
import os
import sqlite3
import csv
from collections import deque
import threading
import winsound  # For Windows sound alerts
import subprocess
import platform
import webbrowser
import urllib.parse

class SystemComponents:
    """
    Manages all core components and state of the surveillance system.
    This includes the GUI, video capture from cameras, and database connections.
    """

    def __init__(self, master=None, config=None, shared_model=None):
        self.shared_model = shared_model
        # If config is None, show config dialog
        if config is None:
            config = self.show_config_dialog()
        
        if not config:  # User cancelled the dialog
            raise SystemExit("Configuration cancelled by user")
            
        self.config = config
        self.db_conn = self.init_db()

        # Enhanced camera management - backward compatible
        self.cameras = {}
        self.active_camera_index = None
        self.next_camera_id = 0

        # Legacy support - keep existing attributes
        self.device_cap = None
        self.ip_cap = None
        self.out = None
        self.recording = False
        self.thermal_mode = False
        self.detection_stats = {'detections': 0, 'weapons': 0, 'avg_confidence': 0.0}
        self.last_time = time.time()

        # Alert system variables
        self.alert_threshold = self.config.get('alert_threshold', 0.6)
        self.alerts_enabled = self.config.get('alerts_enabled', True)
        self.whatsapp_enabled = self.config.get('whatsapp_enabled', False)
        self.whatsapp_number = self.config.get('whatsapp_number', '+1234567890')
        self.last_alert_time = 0
        self.alert_cooldown = 10  # seconds between alerts

        # Performance optimizations
        self.frame_buffer = deque(maxlen=2)
        self.last_gui_update = time.time()
        self.gui_update_interval = 0.033

        # Threading locks
        self.stats_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.db_lock = threading.Lock()

        # Initialize main window and widgets
        if master is None:
            self.root = tk.Tk()
            self.root.title("AI-Powered Surveillance System - Enhanced")
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.geometry("900x700")
        else:
            self.root = master
        # Hold a reference to the current PhotoImage to prevent garbage collection
        self.current_frame_imgtk = None
        self._create_widgets()

    def show_config_dialog(self):
        """Show a configuration dialog for user input."""
        dialog = tk.Tk()
        dialog.title("System Configuration")
        dialog.geometry("450x700")
        dialog.resizable(False, False)

        # Model path
        tk.Label(dialog, text="Model Path:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        model_var = tk.StringVar(value="models/weapon_model.pt")  # Use weapon model by default
        tk.Entry(dialog, textvariable=model_var, width=40).pack(padx=10, pady=5)

        # Detection threshold
        tk.Label(dialog, text="Detection Threshold (0-1):", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        threshold_var = tk.DoubleVar(value=0.4)  # Default threshold - balanced detection
        tk.Entry(dialog, textvariable=threshold_var, width=10).pack(padx=10, pady=5)

        # Camera sources
        tk.Label(dialog, text="Camera Sources (comma separated):", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        cameras_var = tk.StringVar(value="0")
        tk.Entry(dialog, textvariable=cameras_var, width=40).pack(padx=10, pady=5)

        # Database path
        tk.Label(dialog, text="Database Path:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        db_var = tk.StringVar(value="detections.db")  # Match file_weapon_detector.py database
        tk.Entry(dialog, textvariable=db_var, width=40).pack(padx=10, pady=5)

        # Output directory
        tk.Label(dialog, text="Output Directory:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        out_var = tk.StringVar(value="output")
        tk.Entry(dialog, textvariable=out_var, width=40).pack(padx=10, pady=5)

        # Log file
        tk.Label(dialog, text="Log File Path:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        log_var = tk.StringVar(value="logs/weapon_surveillance.log")
        tk.Entry(dialog, textvariable=log_var, width=40).pack(padx=10, pady=5)

        # Weapon classes
        tk.Label(dialog, text="Weapon Classes (comma separated):", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10,0))
        weapons_var = tk.StringVar(value="guns,knife")  # Match trained model classes exactly
        tk.Entry(dialog, textvariable=weapons_var, width=40).pack(padx=10, pady=5)

        # Alert settings frame
        alert_frame = tk.LabelFrame(dialog, text="Alert Settings", padx=10, pady=10, font=('Arial', 10, 'bold'))
        alert_frame.pack(fill='x', padx=10, pady=(15,0))

        # Alert threshold
        tk.Label(alert_frame, text="High Confidence Alert Threshold (0-1):").pack(anchor='w')
        alert_threshold_var = tk.DoubleVar(value=0.6)
        tk.Entry(alert_frame, textvariable=alert_threshold_var, width=10).pack(anchor='w', pady=3)

        # Enable alerts
        alerts_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(alert_frame, text="Enable Screen Alerts", variable=alerts_enabled_var).pack(anchor='w', pady=3)

        # WhatsApp settings
        whatsapp_enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(alert_frame, text="Enable WhatsApp Notifications", variable=whatsapp_enabled_var).pack(anchor='w', pady=3)

        tk.Label(alert_frame, text="WhatsApp Phone Number (with country code):").pack(anchor='w', pady=(5,0))
        whatsapp_number_var = tk.StringVar(value="+1234567890")
        tk.Entry(alert_frame, textvariable=whatsapp_number_var, width=20).pack(anchor='w', pady=3)

        # Confirm button
        result = {}
        def confirm():
            try:
                threshold = threshold_var.get()
                alert_thresh = alert_threshold_var.get()
                
                if not (0 <= threshold <= 1):
                    messagebox.showerror("Invalid Input", "Detection threshold must be between 0 and 1")
                    return
                
                if not (0 <= alert_thresh <= 1):
                    messagebox.showerror("Invalid Input", "Alert threshold must be between 0 and 1")
                    return
                
                result['model_path'] = model_var.get().strip()
                result['detection_threshold'] = threshold
                result['camera_sources'] = [s.strip() for s in cameras_var.get().split(',') if s.strip()]
                result['database_path'] = db_var.get().strip()
                result['output_directory'] = out_var.get().strip()
                result['log_path'] = log_var.get().strip()
                result['weapon_classes'] = [w.strip().lower() for w in weapons_var.get().split(',') if w.strip()]
                result['alert_threshold'] = alert_thresh
                result['alerts_enabled'] = alerts_enabled_var.get()
                result['whatsapp_enabled'] = whatsapp_enabled_var.get()
                result['whatsapp_number'] = whatsapp_number_var.get().strip()
                result['process_every_n_frames'] = 2  # Default: Balanced mode
                
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please check your input values: {e}")

        tk.Button(dialog, text="Start System", command=confirm, bg='#4CAF50', fg='white', 
                 font=('Arial', 11, 'bold'), padx=20, pady=10).pack(pady=20)
        
        dialog.mainloop()
        return result if result else None

    def _init_database(self):
        """Initialize SQLite database for logging detections."""
        try:
            db_path = self.config.get('database_path', 'detections.db')
            
            # Create directory if it doesn't exist
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
            cursor = self.db_conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS detections
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT NOT NULL,
                         camera_index INTEGER NOT NULL,
                         object_class TEXT NOT NULL,
                         confidence REAL NOT NULL,
                         weapon_detected INTEGER NOT NULL,
                         track_id INTEGER DEFAULT 0)''')
            
            # Migration for existing DB
            try:
                cursor.execute("ALTER TABLE detections ADD COLUMN track_id INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass # Column already exists
                
            self.db_conn.commit()
            logging.info(f"Database initialized successfully at {db_path}")
            return self.db_conn
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
            messagebox.showerror("Database Error", f"Failed to initialize the database: {e}")
            return None

    def init_db(self):
        return self._init_database()

    def open_dynamic_config(self):
        """Open a dynamic configuration dialog that updates settings in real-time."""
        config_window = tk.Toplevel(self.root)
        config_window.title("System Configuration - Live Settings")
        config_window.geometry("500x650")
        config_window.resizable(False, False)
        config_window.transient(self.root)  # Keep on top of main window
        
        # Main frame with scrollbar
        main_canvas = tk.Canvas(config_window)
        scrollbar = ttk.Scrollbar(config_window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # === Detection Settings ===
        detection_frame = tk.LabelFrame(scrollable_frame, text="Detection Settings", 
                                       padx=15, pady=15, font=('Arial', 10, 'bold'))
        detection_frame.pack(fill='x', padx=10, pady=10)
        
        # Detection threshold with real-time slider
        tk.Label(detection_frame, text="Detection Threshold:", 
                font=('Arial', 9)).pack(anchor='w', pady=(0,5))
        
        threshold_display = tk.Label(detection_frame, 
                                    text=f"Current: {self.threshold_slider.get():.2f}",
                                    font=('Arial', 9, 'bold'), fg='blue')
        threshold_display.pack(anchor='w')
        
        threshold_var = tk.DoubleVar(value=self.threshold_slider.get())
        
        def update_threshold(val):
            """Update detection threshold in real-time"""
            new_val = float(val)
            self.threshold_slider.set(new_val)
            self.config['detection_threshold'] = new_val
            threshold_display.config(text=f"Current: {new_val:.2f}")
            logging.info(f"Detection threshold updated to {new_val:.2f}")
        
        threshold_scale = tk.Scale(detection_frame, from_=0.1, to=0.95, resolution=0.05,
                                  orient='horizontal', variable=threshold_var,
                                  command=update_threshold, length=350)
        threshold_scale.pack(fill='x', pady=5)
        
        # === Alert Settings ===
        alert_frame = tk.LabelFrame(scrollable_frame, text="Alert Settings", 
                                   padx=15, pady=15, font=('Arial', 10, 'bold'))
        alert_frame.pack(fill='x', padx=10, pady=10)
        
        # Alert threshold
        tk.Label(alert_frame, text="High Confidence Alert Threshold:", 
                font=('Arial', 9)).pack(anchor='w', pady=(0,5))
        
        alert_display = tk.Label(alert_frame, 
                               text=f"Current: {self.alert_threshold:.2f}",
                               font=('Arial', 9, 'bold'), fg='red')
        alert_display.pack(anchor='w')
        
        alert_threshold_var = tk.DoubleVar(value=self.alert_threshold)
        
        def update_alert_threshold(val):
            """Update alert threshold in real-time"""
            new_val = float(val)
            self.alert_threshold = new_val
            self.config['alert_threshold'] = new_val
            alert_display.config(text=f"Current: {new_val:.2f}")
            logging.info(f"Alert threshold updated to {new_val:.2f}")
        
        alert_scale = tk.Scale(alert_frame, from_=0.5, to=0.95, resolution=0.05,
                              orient='horizontal', variable=alert_threshold_var,
                              command=update_alert_threshold, length=350)
        alert_scale.pack(fill='x', pady=5)
        
        # Enable screen alerts
        alerts_enabled_var = tk.BooleanVar(value=self.alerts_enabled)
        
        def toggle_alerts():
            """Toggle screen alerts"""
            self.alerts_enabled = alerts_enabled_var.get()
            self.config['alerts_enabled'] = self.alerts_enabled
            status = "enabled" if self.alerts_enabled else "disabled"
            logging.info(f"Screen alerts {status}")
            self.update_status_bar(f"Screen alerts {status}")
        
        tk.Checkbutton(alert_frame, text="Enable Screen Alerts", 
                      variable=alerts_enabled_var, 
                      command=toggle_alerts,
                      font=('Arial', 9)).pack(anchor='w', pady=5)
        
        # === WhatsApp Settings ===
        whatsapp_frame = tk.LabelFrame(scrollable_frame, text="WhatsApp Notifications", 
                                      padx=15, pady=15, font=('Arial', 10, 'bold'))
        whatsapp_frame.pack(fill='x', padx=10, pady=10)
        
        # Enable WhatsApp
        whatsapp_enabled_var = tk.BooleanVar(value=self.whatsapp_enabled)
        
        def toggle_whatsapp():
            """Toggle WhatsApp notifications"""
            self.whatsapp_enabled = whatsapp_enabled_var.get()
            self.config['whatsapp_enabled'] = self.whatsapp_enabled
            status = "enabled" if self.whatsapp_enabled else "disabled"
            logging.info(f"WhatsApp notifications {status}")
            self.update_status_bar(f"WhatsApp notifications {status}")
            # Enable/disable phone number entry
            phone_entry.config(state='normal' if self.whatsapp_enabled else 'disabled')
        
        tk.Checkbutton(whatsapp_frame, text="Enable WhatsApp Notifications", 
                      variable=whatsapp_enabled_var, 
                      command=toggle_whatsapp,
                      font=('Arial', 9)).pack(anchor='w', pady=5)
        
        # WhatsApp phone number
        tk.Label(whatsapp_frame, text="Phone Number (with country code):", 
                font=('Arial', 9)).pack(anchor='w', pady=(10,5))
        
        phone_var = tk.StringVar(value=self.whatsapp_number)
        phone_entry = tk.Entry(whatsapp_frame, textvariable=phone_var, width=25,
                              state='normal' if self.whatsapp_enabled else 'disabled')
        phone_entry.pack(anchor='w', pady=5)
        
        def update_phone():
            """Update WhatsApp phone number"""
            new_phone = phone_var.get().strip()
            if new_phone:
                self.whatsapp_number = new_phone
                self.config['whatsapp_number'] = new_phone
                logging.info(f"WhatsApp number updated to {new_phone}")
                messagebox.showinfo("Updated", f"WhatsApp number updated to:\n{new_phone}")
        
        tk.Button(whatsapp_frame, text="Update Phone Number", command=update_phone,
                 bg='#2196F3', fg='white', font=('Arial', 9)).pack(anchor='w', pady=10)
        
        # === Performance Settings ===
        perf_frame = tk.LabelFrame(scrollable_frame, text="Performance Settings", 
                                  padx=15, pady=15, font=('Arial', 10, 'bold'))
        perf_frame.pack(fill='x', padx=10, pady=10)
        
        # Frame processing rate
        tk.Label(perf_frame, text="Processing Mode:", 
                font=('Arial', 9)).pack(anchor='w', pady=(0,5))
        
        process_mode_var = tk.StringVar(value="Balanced (Every 2nd frame)")
        
        def update_processing_mode():
            """Update frame processing mode"""
            mode = process_mode_var.get()
            # This will be read by the detection worker
            if "Every frame" in mode:
                self.config['process_every_n_frames'] = 1
                logging.info("Processing mode: Maximum (every frame)")
            elif "3rd frame" in mode:
                self.config['process_every_n_frames'] = 3
                logging.info("Processing mode: Performance (every 3rd frame)")
            else:  # Balanced
                self.config['process_every_n_frames'] = 2
                logging.info("Processing mode: Balanced (every 2nd frame)")
            
            messagebox.showinfo("Processing Mode", 
                              f"Mode updated to: {mode}\n\n"
                              "Note: Restart detection worker for changes to take full effect.")
        
        for mode in ["Maximum (Every frame)", "Balanced (Every 2nd frame)", "Performance (Every 3rd frame)"]:
            tk.Radiobutton(perf_frame, text=mode, variable=process_mode_var, 
                         value=mode, command=update_processing_mode,
                         font=('Arial', 9)).pack(anchor='w', pady=2)
        
        # === Info Section ===
        info_frame = tk.LabelFrame(scrollable_frame, text="System Information", 
                                  padx=15, pady=15, font=('Arial', 10, 'bold'))
        info_frame.pack(fill='x', padx=10, pady=10)
        
        import torch
        gpu_info = "GPU Available" if torch.cuda.is_available() else "CPU Only"
        if torch.cuda.is_available():
            gpu_info += f"\n{torch.cuda.get_device_name(0)}"
        
        tk.Label(info_frame, text=f"Model: {self.config.get('model_path', 'N/A')}", 
                font=('Arial', 8), anchor='w').pack(fill='x', pady=2)
        tk.Label(info_frame, text=f"Database: {self.config.get('database_path', 'N/A')}", 
                font=('Arial', 8), anchor='w').pack(fill='x', pady=2)
        tk.Label(info_frame, text=f"Hardware: {gpu_info}", 
                font=('Arial', 8), anchor='w').pack(fill='x', pady=2)
        
        # === Action Buttons ===
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=10, pady=20)
        
        def close_config():
            """Close configuration window"""
            config_window.destroy()
            self.update_status_bar("Configuration updated successfully")
        
        tk.Button(button_frame, text="Close", command=close_config,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                 padx=30, pady=10).pack(side='right', padx=5)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Center the window
        config_window.update_idletasks()
        x = (config_window.winfo_screenwidth() // 2) - (config_window.winfo_width() // 2)
        y = (config_window.winfo_screenheight() // 2) - (config_window.winfo_height() // 2)
        config_window.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        """Creates and arranges GUI widgets with camera management."""
        # Create menu bar
        self.create_menu_bar()
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Camera controls frame
        camera_control_frame = ttk.LabelFrame(main_frame, text="Camera Controls", padding="10")
        camera_control_frame.grid(row=0, column=0, sticky="we", pady=(0, 10))
        camera_control_frame.columnconfigure(1, weight=1)

        # Active camera display
        self.active_camera_label = ttk.Label(
            camera_control_frame, 
            text="Active Camera: No cameras configured",
            font=('Arial', 12, 'bold'),
            foreground='blue'
        )
        self.active_camera_label.grid(row=0, column=0, sticky=tk.W, padx=10)

        # Camera control buttons
        button_frame = ttk.Frame(camera_control_frame)
        button_frame.grid(row=0, column=1, sticky=tk.E)

        ttk.Button(button_frame, text="Add Camera", command=self.quick_add_camera).pack(side=tk.RIGHT, padx=5)
        
        self.switch_camera_button = ttk.Button(
            button_frame,
            text="Switch Camera",
            command=self.switch_camera,
            state="disabled"
        )
        self.switch_camera_button.pack(side=tk.RIGHT, padx=5)

        # Content container for Video + Suspects
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        content_frame.columnconfigure(0, weight=3) # Video
        content_frame.columnconfigure(1, weight=1) # Suspects
        content_frame.rowconfigure(0, weight=1)

        # Video Frame - Single display
        video_frame = ttk.LabelFrame(content_frame, text="Live Video Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        # Single video label
        self.video_label = ttk.Label(
            video_frame, 
            compound='center',
            anchor='center',
            font=('Arial', 14),
            foreground='gray'
        )
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_label.bind("<Button-1>", self.on_video_click)

        # Legacy labels for backward compatibility
        self.device_video_label = self.video_label
        self.ip_video_label = self.video_label

        # Cinematic Suspects Panel
        self.suspects_frame = tk.LabelFrame(content_frame, text="ACTIVE SUSPECTS", padx=5, pady=5, font=('Courier', 12, 'bold'), fg='red', bg='#111111')
        self.suspects_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self.suspects_canvas = tk.Canvas(self.suspects_frame, bg="#111111", highlightthickness=0, width=250)
        self.suspects_scrollbar = ttk.Scrollbar(self.suspects_frame, orient="vertical", command=self.suspects_canvas.yview)
        self.suspects_inner_frame = tk.Frame(self.suspects_canvas, bg="#111111")
        
        self.suspects_inner_frame.bind(
            "<Configure>",
            lambda e: self.suspects_canvas.configure(
                scrollregion=self.suspects_canvas.bbox("all")
            )
        )
        
        self.suspect_window = self.suspects_canvas.create_window((0, 0), window=self.suspects_inner_frame, anchor="nw", width=250)
        self.suspects_canvas.configure(yscrollcommand=self.suspects_scrollbar.set)
        
        self.suspects_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.suspects_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.suspect_count = 0
        self.suspect_images_refs = []  # Keep references to avoid GC

        # Statistics Frame
        stats_frame = ttk.LabelFrame(main_frame, text="Detection Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky="we", pady=(0, 10))

        stats_inner = ttk.Frame(stats_frame)
        stats_inner.pack(fill=tk.X)

        self.detection_label = ttk.Label(stats_inner, text="Detections: 0", font=('Arial', 11, 'bold'))
        self.detection_label.pack(side=tk.LEFT, padx=15)

        self.weapon_label = ttk.Label(stats_inner, text="Weapons: 0", font=('Arial', 11, 'bold'), foreground='red')
        self.weapon_label.pack(side=tk.LEFT, padx=15)

        self.confidence_label = ttk.Label(stats_inner, text="Avg Confidence: 0.00", font=('Arial', 11))
        self.confidence_label.pack(side=tk.LEFT, padx=15)

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="System Controls", padding="10")
        controls_frame.grid(row=3, column=0, sticky="we", pady=(0, 10))

        # Threshold slider
        threshold_frame = ttk.Frame(controls_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        ttk.Label(threshold_frame, text="Detection Threshold:").pack(side=tk.LEFT)
        self.threshold_slider = tk.Scale(
            threshold_frame, 
            from_=0, to=1, resolution=0.01,
            orient='horizontal', 
            length=250,
            showvalue=True
        )
        self.threshold_slider.set(self.config.get("detection_threshold", 0.4))
        self.threshold_slider.pack(side=tk.LEFT, padx=10)

        # Control buttons
        button_frame2 = ttk.Frame(controls_frame)
        button_frame2.pack(fill=tk.X, pady=5)

        self.thermal_button = ttk.Button(
            button_frame2, 
            text="Enable Thermal Mode", 
            command=self.toggle_thermal_mode
        )
        self.thermal_button.pack(side=tk.LEFT, padx=5)

        self.clear_zone_btn = ttk.Button(
            button_frame2,
            text="Clear Intrusion Zone",
            command=self.clear_intrusion_zone
        )
        self.clear_zone_btn.pack(side=tk.LEFT, padx=5)
        
        self._continue_create_widgets(main_frame, button_frame2)

    def clear_intrusion_zone(self):
        self.config['intrusion_zone'] = []
        self.update_status_bar("Intrusion zone cleared.")

    def on_video_click(self, event):
        if 'intrusion_zone' not in self.config:
            self.config['intrusion_zone'] = []
            
        # The image is 640x480 and centered in the label. We need to calculate the offset.
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        
        offset_x = max(0, (label_w - 640) // 2)
        offset_y = max(0, (label_h - 480) // 2)
        
        x = event.x - offset_x
        y = event.y - offset_y
        
        # Ignore clicks outside the actual image boundary
        if x < 0 or x > 640 or y < 0 or y > 480:
            return
            
        if len(self.config['intrusion_zone']) >= 4:
            self.config['intrusion_zone'] = [] # Reset on 5th click
            
        self.config['intrusion_zone'].append([x, y])
        self.update_status_bar(f"Added point {len(self.config['intrusion_zone'])}/4 to intrusion zone.")

        # END of on_video_click. The following lines belong to _create_widgets

    def _continue_create_widgets(self, main_frame, button_frame2):
        self.record_button = ttk.Button(
            button_frame2, 
            text="Start Recording", 
            command=self.toggle_recording,
            state="disabled"
        )
        self.record_button.pack(side=tk.LEFT, padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Status: Ready - Add cameras to begin surveillance")
        self.status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=('Arial', 10)
        )
        self.status_bar.grid(row=4, column=0, sticky="we")

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        try:
            self.root.config(menu=menubar)
        except AttributeError:
            pass # CTkFrame doesn't support native menus, we'll skip the menu in dashboard mode

        # Camera menu
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Cameras", menu=camera_menu)
        camera_menu.add_command(label="Add Camera...", command=self.quick_add_camera)
        camera_menu.add_separator()
        camera_menu.add_command(label="Refresh Cameras", command=self.refresh_all_cameras)

        # Database menu
        database_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Database", menu=database_menu)
        database_menu.add_command(label="View Detections", command=self.show_database_viewer)
        database_menu.add_command(label="Export Data", command=self.export_database)
        database_menu.add_separator()
        database_menu.add_command(label="Clear Database", command=self.clear_database)

        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="System Configuration", command=self.open_dynamic_config)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Camera Setup Guide", command=self.show_camera_guide)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        """Show about dialog."""
        about_text = """AI-Powered Target Recognition System
Version 2.0

Real-time weapon detection and surveillance system
using YOLOv8 deep learning architecture.

Features:
• Multi-camera support
• Real-time weapon detection
• Intelligent alert system
• Evidence preservation
• Database logging

© 2024 - All Rights Reserved"""
        
        messagebox.showinfo("About", about_text)

    def initialize_cameras(self):
        """Initialize cameras from config."""
        camera_sources = self.config.get('camera_sources', [0])
        
        for i, source in enumerate(camera_sources):
            name = f"Camera {i}"
            if i == 0:
                name = "Primary Camera"
            elif isinstance(source, str) and ('http' in source or 'rtsp' in source):
                name = f"IP Camera {i}"
            
            success = self.add_camera_source(source, name)
            if success:
                logging.info(f"Initialized camera: {name}")

        if not self.cameras:
            logging.warning("No cameras could be initialized from config")
            messagebox.showwarning("Camera Warning", "No cameras could be initialized. Please add cameras manually.")
        else:
            self.switch_camera_button.config(state="normal")
            self.record_button.config(state="normal")

    def quick_add_camera(self):
        """Quick add camera dialog."""
        source = simpledialog.askstring(
            "Add Camera", 
            "Enter camera source:\n\nExamples:\n• USB: 0, 1, 2\n• IP Camera: http://192.168.1.100:8080/video\n• RTSP: rtsp://192.168.1.100:554/stream"
        )
        
        if source:
            name = simpledialog.askstring("Camera Name", "Enter camera name (optional):")
            if not name:
                name = f"Camera {len(self.cameras)}"
            
            success = self.add_camera_source(source.strip(), name.strip())
            if success:
                messagebox.showinfo("Success", f"Camera '{name}' added successfully!")
            else:
                messagebox.showerror("Error", "Failed to add camera. Please check the source and try again.")

    def add_camera_source(self, source, name):
        """Add a new camera source with IP camera support."""
        try:
            # Convert to int if it's a number (USB camera)
            try:
                camera_source = int(source)
                is_ip_camera = False
                logging.info(f"Adding USB camera: {camera_source}")
            except ValueError:
                camera_source = source
                is_ip_camera = ('http' in camera_source.lower() or 'rtsp' in camera_source.lower())
                logging.info(f"Adding {'IP/RTSP' if is_ip_camera else 'file/other'} camera: {camera_source}")
            
            # Create VideoCapture with appropriate backend
            if is_ip_camera:
                # For IP cameras, use CAP_FFMPEG backend explicitly
                logging.info(f"Opening IP camera with FFMPEG backend: {camera_source}")
                cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                
                # Set IP camera specific settings with error handling
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for low latency
                except Exception as e:
                    logging.warning(f"Could not set buffer size: {e}")
                
                # Try hardware acceleration if available (skip if not supported)
                try:
                    if hasattr(cv2, 'VIDEO_ACCELERATION_ANY'):
                        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                except Exception as e:
                    logging.info(f"Hardware acceleration not available: {e}")
            else:
                # USB camera or file
                import os
                if isinstance(camera_source, int) and os.name == 'nt':
                    logging.info(f"Using DirectShow backend for USB camera: {camera_source}")
                    cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(camera_source)
                
                # Optimize USB camera settings
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except Exception as e:
                    logging.warning(f"Could not optimize USB camera settings: {e}")
            
            # Test camera connection
            if not cap.isOpened():
                cap.release()
                error_msg = f"Failed to open camera source: {source}"
                logging.error(error_msg)
                
                if is_ip_camera:
                    messagebox.showerror("IP Camera Error", 
                        f"Failed to connect to IP camera:\n{source}\n\n"
                        "Common issues:\n"
                        "• Check network connection\n"
                        "• Verify IP address and port\n"
                        "• Ensure camera is streaming\n"
                        "• Check firewall settings\n"
                        "• Try adding username:password if required\n"
                        "  Example: http://user:pass@192.168.1.100:8080/video")
                else:
                    messagebox.showerror("Camera Error", 
                        f"Failed to open camera:\n{source}\n\n"
                        "Please check:\n"
                        "• Camera is connected and powered on\n"
                        "• No other app is using the camera\n"
                        "• Correct device index (0, 1, 2...)")
                return False
            
            # Try to read a test frame with timeout
            logging.info(f"Testing camera stream: {source}")
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                error_msg = f"Camera opened but failed to read frame: {source}"
                logging.error(error_msg)
                
                if is_ip_camera:
                    messagebox.showerror("IP Camera Streaming Error", 
                        f"Camera connected but not streaming:\n{source}\n\n"
                        "Possible causes:\n"
                        "• Camera is not actively streaming\n"
                        "• Wrong video path/endpoint\n"
                        "• Codec not supported\n"
                        "• Network bandwidth issue\n\n"
                        "Try different endpoints:\n"
                        "• /video\n"
                        "• /videostream\n"
                        "• /mjpeg\n"
                        "• /stream")
                else:
                    messagebox.showerror("Camera Error", 
                        f"Camera connected but not producing frames:\n{source}\n\n"
                        "Please check camera settings and try again.")
                return False
            
            logging.info(f"Successfully read test frame from camera: {source} (shape: {test_frame.shape})")
            
            # Add to cameras dictionary
            camera_id = self.next_camera_id
            self.cameras[camera_id] = {
                'capture': cap,
                'source': camera_source,
                'name': name,
                'is_ip': is_ip_camera
            }
            
            # Set as active if it's the first camera
            if len(self.cameras) == 1:
                self.active_camera_index = camera_id
                self.switch_camera_button.config(state="normal")
                self.record_button.config(state="normal")
                
                # Update legacy attributes for backward compatibility
                self.device_cap = cap
            
            self.next_camera_id += 1
            self._update_active_camera_display()
            
            camera_type = "IP/RTSP" if is_ip_camera else "USB"
            logging.info(f"✅ {camera_type} camera added successfully: {name} (ID: {camera_id}, Source: {source})")
            return True
            
        except cv2.error as e:
            error_msg = f"OpenCV error adding camera: {e}"
            logging.error(error_msg)
            messagebox.showerror("Camera Error", 
                f"Failed to add camera due to OpenCV error:\n\n{str(e)}\n\n"
                "This may be due to:\n"
                "• Unsupported video codec\n"
                "• Missing FFMPEG libraries\n"
                "• Invalid camera URL format")
            return False
        except Exception as e:
            error_msg = f"Unexpected error adding camera: {e}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Error", f"Failed to add camera:\n\n{str(e)}")
            return False

    def get_active_camera(self):
        """Returns the active camera capture object."""
        if self.active_camera_index is not None and self.active_camera_index in self.cameras:
            return self.cameras[self.active_camera_index]['capture']
        return None

    def get_active_camera_name(self):
        """Returns the active camera name."""
        if self.active_camera_index is not None and self.active_camera_index in self.cameras:
            return self.cameras[self.active_camera_index]['name']
        return "No Camera"

    def switch_camera(self):
        """Switch to the next available camera."""
        if len(self.cameras) <= 1:
            messagebox.showinfo("Camera Switch", "Only one camera available.")
            return
        
        with self.camera_lock:
            camera_ids = list(self.cameras.keys())
            try:
                current_pos = camera_ids.index(self.active_camera_index)
                next_pos = (current_pos + 1) % len(camera_ids)
                self.active_camera_index = camera_ids[next_pos]
            except (ValueError, IndexError):
                self.active_camera_index = camera_ids[0] if camera_ids else None
            
            # Update legacy attributes for backward compatibility
            active_cap = self.get_active_camera()
            if self.active_camera_index == 0:
                self.device_cap = active_cap
                self.ip_cap = None
            else:
                self.ip_cap = active_cap
                self.device_cap = None
            
            self._update_active_camera_display()
            
            camera_name = self.get_active_camera_name()
            self.update_status_bar(f"Switched to {camera_name}")
            logging.info(f"Camera switched to: {camera_name}")

    def _update_active_camera_display(self):
        """Update the active camera label."""
        if self.cameras:
            camera_name = self.get_active_camera_name()
            self.active_camera_label.config(
                text=f"Active Camera: {camera_name} (ID: {self.active_camera_index})"
            )
        else:
            self.active_camera_label.config(text="Active Camera: No cameras configured")
            self.video_label.config(
                text="No camera active\n\nClick 'Add Camera' to get started"
            )

    def refresh_all_cameras(self):
        """Refresh all camera connections."""
        disconnected_cameras = []
        
        for camera_id, camera_data in self.cameras.items():
            if not camera_data['capture'].isOpened():
                disconnected_cameras.append((camera_id, camera_data['name']))
        
        if disconnected_cameras:
            camera_list = "\n".join([f"• {name}" for _, name in disconnected_cameras])
            messagebox.showwarning(
                "Disconnected Cameras", 
                f"The following cameras are disconnected:\n\n{camera_list}\n\nPlease check connections and restart cameras."
            )
        else:
            messagebox.showinfo("Camera Status", "All cameras are connected and working properly.")

    def show_camera_guide(self):
        """Show camera setup guide."""
        guide_text = """Camera Setup Guide

USB CAMERAS:
• Use device index: 0, 1, 2, etc.
• 0 is usually the built-in webcam
• 1, 2, 3... for external USB cameras

IP CAMERAS:
• HTTP format: http://192.168.1.100:8080/video
• RTSP format: rtsp://192.168.1.100:554/stream
• Include username/password if required:
  rtsp://user:pass@192.168.1.100:554/stream

MOBILE CAMERAS:
• IP Webcam app: http://phone-ip:8080/video
• DroidCam: http://phone-ip:4747/video
• Ensure phone and PC are on same network

TROUBLESHOOTING:
• Close other apps using the camera
• Check network connectivity for IP cameras
• Try different ports (8080, 8081, 554)
• Verify camera permissions in Windows settings
• Restart the application if camera freezes
        """
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Camera Setup Guide")
        guide_window.geometry("550x450")
        guide_window.resizable(False, False)
        
        text_widget = tk.Text(guide_window, wrap=tk.WORD, padx=15, pady=15, font=('Arial', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, guide_text)
        text_widget.config(state=tk.DISABLED)
        
        close_btn = ttk.Button(guide_window, text="Close", command=guide_window.destroy)
        close_btn.pack(pady=10)

    def toggle_thermal_mode(self):
        """Switches between normal and thermal modes."""
        self.thermal_mode = not self.thermal_mode
        mode = "Thermal" if self.thermal_mode else "Normal"
        button_text = "Disable Thermal Mode" if self.thermal_mode else "Enable Thermal Mode"
        
        self.thermal_button.config(text=button_text)
        self.update_status_bar(f"Switched to {mode} mode")
        logging.info(f"Thermal mode: {self.thermal_mode}")

    def toggle_recording(self):
        """Starts or stops recording from the active camera."""
        if not self.recording:
            try:
                active_cap = self.get_active_camera()
                if not active_cap:
                    messagebox.showerror("Recording Error", "No active camera available for recording.")
                    return
                
                # Create output directory
                output_dir = self.config.get('output_directory', 'output')
                os.makedirs(output_dir, exist_ok=True)
                
                # Get dimensions from active camera
                frame_width = int(active_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(active_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = 20
                
                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                camera_name = self.get_active_camera_name().replace(' ', '_')
                filename = f"recording_{camera_name}_{timestamp}.avi"
                filepath = os.path.join(output_dir, filename)
                
                # Initialize video writer
                fourcc = cv2.VideoWriter.fourcc(*'XVID')
                self.out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))
                
                if self.out.isOpened():
                    self.recording = True
                    self.record_button.config(text="Stop Recording")
                    self.update_status_bar(f"Recording started: {filename}")
                    logging.info(f"Recording started: {filepath}")
                else:
                    raise Exception("Failed to initialize video writer")
                    
            except Exception as e:
                logging.error(f"Recording error: {e}")
                messagebox.showerror("Recording Error", f"Could not start recording:\n{e}")
        else:
            self.recording = False
            self.record_button.config(text="Start Recording")
            if self.out:
                self.out.release()
                self.out = None
            self.update_status_bar("Recording stopped")
            logging.info("Recording stopped")

    def update_status_bar(self, text):
        """Update status bar text safely from any thread."""
        try:
            if hasattr(self, 'status_var'):
                # Schedule update on the main GUI thread to prevent Tkinter corruption
                if hasattr(self, 'root') and self.root:
                    self.root.after(0, lambda t=text: self.status_var.set(t))
                else:
                    self.status_var.set(text)
        except Exception as e:
            logging.error(f"Error updating status bar: {e}")

    def update_statistics_display(self):
        """Refreshes detection statistics."""
        with self.stats_lock:
            try:
                self.detection_label.config(
                    text=f"Detections: {self.detection_stats['detections']}"
                )
                self.weapon_label.config(
                    text=f"Weapons: {self.detection_stats['weapons']}"
                )
                self.confidence_label.config(
                    text=f"Avg Confidence: {self.detection_stats['avg_confidence']:.2f}"
                )
            except Exception as e:
                logging.error(f"Error updating statistics display: {e}")

    def update_video_feed_smooth(self, frame):
        """Update video feed with smooth frame display - optimized to prevent flicker."""
        try:
            current_time = time.time()
            # Throttle GUI updates to 30 FPS max (prevents flicker)
            if current_time - self.last_gui_update < 0.033:  # ~30 FPS
                return
            
            self.last_gui_update = current_time
            
            # Convert frame to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
            
            # Draw intrusion zone on the display frame
            if self.config.get('intrusion_zone') and len(self.config['intrusion_zone']) > 0:
                import numpy as np
                pts = np.array(self.config['intrusion_zone'], np.int32)
                
                # Draw points and intermediate lines
                for pt in pts:
                    cv2.circle(frame_resized, tuple(pt), 5, (0, 0, 255), -1)
                if len(pts) == 2:
                    cv2.line(frame_resized, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 2)
                elif len(pts) >= 3:
                    # Use convex hull to ensure a clean polygon without crossed lines
                    hull = cv2.convexHull(pts)
                    cv2.polylines(frame_resized, [hull], True, (255, 0, 0), 2)
                    
                    # Semi-transparent overlay
                    overlay = frame_resized.copy()
                    cv2.fillPoly(overlay, [hull], (255, 0, 0))
                    cv2.addWeighted(overlay, 0.2, frame_resized, 0.8, 0, frame_resized)
                        
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Keep a reference to prevent garbage collection
            self.current_frame_imgtk = imgtk
            # Update label without triggering multiple redraws
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk  # Additional reference
            
        except Exception as e:
            logging.error(f"Error updating video feed: {e}")

    def show_database_viewer(self):
        """Show database viewer window."""
        viewer = tk.Toplevel(self.root)
        viewer.title("Detection Database Viewer")
        viewer.geometry("900x500")
        
        # Create treeview
        tree_frame = ttk.Frame(viewer)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        tree = ttk.Treeview(tree_frame, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        
        # Configure columns
        tree["columns"] = ("ID", "Timestamp", "Camera", "Object", "Confidence", "Weapon")
        tree.column("#0", width=0, stretch=tk.NO)
        tree.column("ID", anchor=tk.W, width=50)
        tree.column("Timestamp", anchor=tk.W, width=150)
        tree.column("Camera", anchor=tk.W, width=100)
        tree.column("Object", anchor=tk.W, width=150)
        tree.column("Confidence", anchor=tk.W, width=100)
        tree.column("Weapon", anchor=tk.W, width=100)
        
        # Create headings
        tree.heading("ID", text="ID", anchor=tk.W)
        tree.heading("Timestamp", text="Timestamp", anchor=tk.W)
        tree.heading("Camera", text="Camera", anchor=tk.W)
        tree.heading("Object", text="Object Class", anchor=tk.W)
        tree.heading("Confidence", text="Confidence", anchor=tk.W)
        tree.heading("Weapon", text="Weapon Detected", anchor=tk.W)
        
        # Pack treeview and scrollbars
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Load data
        try:
            if not self.db_conn:
                raise Exception("Database connection not available")
            
            with self.db_lock:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1000")
                rows = cursor.fetchall()
            
            for row in rows:
                weapon_status = "Yes" if row[5] == 1 else "No"
                tree.insert("", tk.END, values=(
                    row[0], row[1], f"Camera {row[2]}", row[3], 
                    f"{row[4]:.2f}", weapon_status
                ))
            
            # Add count label
            count_label = ttk.Label(viewer, text=f"Total records: {len(rows)} (showing latest 1000)", 
                                   font=('Arial', 10))
            count_label.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load data: {e}")
            logging.error(f"Database viewer error: {e}")
        
        # Buttons
        button_frame = ttk.Frame(viewer)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Export to CSV", 
                  command=lambda: self.export_database()).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Refresh", 
                  command=lambda: self.show_database_viewer()).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=viewer.destroy).pack(side=tk.LEFT, padx=5)

    def export_database(self):
        """Export database to CSV file."""
        try:
            # Ask for save location
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            
            if not filepath:
                return
            
            if not self.db_conn:
                raise Exception("Database connection not available")
            
            # Query all data
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT * FROM detections ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            
            # Write to CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['ID', 'Timestamp', 'Camera Index', 'Object Class', 
                               'Confidence', 'Weapon Detected'])
                
                for row in rows:
                    weapon_status = "Yes" if row[5] == 1 else "No"
                    writer.writerow([row[0], row[1], row[2], row[3], row[4], weapon_status])
            
            messagebox.showinfo("Export Successful", 
                              f"Exported {len(rows)} records to:\n{filepath}")
            logging.info(f"Database exported to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export database: {e}")
            logging.error(f"Database export error: {e}")

    def clear_database(self):
        """Clear all detections from database."""
        result = messagebox.askyesno(
            "Clear Database",
            "Are you sure you want to delete ALL detection records?\n\nThis action cannot be undone!"
        )
        
        if result:
            try:
                if not self.db_conn:
                    raise Exception("Database connection not available")
                
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM detections")
                self.db_conn.commit()
                
                messagebox.showinfo("Database Cleared", "All detection records have been deleted.")
                logging.info("Database cleared by user")
                
                # Reset statistics
                with self.stats_lock:
                    self.detection_stats = {'detections': 0, 'weapons': 0, 'avg_confidence': 0.0}
                self.update_statistics_display()
                
            except Exception as e:
                messagebox.showerror("Database Error", f"Failed to clear database: {e}")
                logging.error(f"Database clear error: {e}")

    def trigger_high_confidence_alert(self, object_class, confidence, frame):
        """Trigger alerts for high confidence detections."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        
        # Screen alert
        if self.alerts_enabled:
            self.show_screen_alert(object_class, confidence)
        
        # Sound alert
        try:
            frequency = 2500  # Hz
            duration = 500  # ms
            winsound.Beep(frequency, duration)
        except Exception as e:
            logging.warning(f"Sound alert failed: {e}")
        
        # Save detection image
        image_path = self.save_detection_image(frame, object_class, confidence)
        
        # WhatsApp alert
        if self.whatsapp_enabled and image_path:
            threading.Thread(target=self.send_whatsapp_alert, 
                           args=(object_class, confidence, image_path), 
                           daemon=True).start()

    def show_screen_alert(self, object_class, confidence):
        """Show screen alert window."""
        if hasattr(self, 'root') and self.root:
            self.root.after(0, lambda: self._create_screen_alert(object_class, confidence))
        else:
            self._create_screen_alert(object_class, confidence)

    def _create_screen_alert(self, object_class, confidence):
        alert_window = tk.Toplevel(self.root)
        alert_window.title("⚠️ HIGH CONFIDENCE DETECTION ⚠️")
        alert_window.geometry("400x200")
        alert_window.configure(bg='red')
        
        # Make it always on top
        alert_window.attributes('-topmost', True)
        
        # Alert message
        message = f"WEAPON DETECTED!\n\n{object_class.upper()}\n\nConfidence: {confidence*100:.1f}%"
        
        label = tk.Label(
            alert_window, 
            text=message,
            font=('Arial', 18, 'bold'),
            bg='red',
            fg='white'
        )
        label.pack(expand=True)
        
        # Auto close after 5 seconds
        alert_window.after(5000, alert_window.destroy)
        
        # Flash effect
        def flash():
            try:
                current_bg = alert_window.cget('bg')
                new_bg = 'yellow' if current_bg == 'red' else 'red'
                alert_window.configure(bg=new_bg)
                label.configure(bg=new_bg)
                if alert_window.winfo_exists():
                    alert_window.after(500, flash)
            except:
                pass
        
        flash()

    def save_detection_image(self, frame, object_class, confidence):
        """Save detection image as evidence."""
        try:
            # Create evidence directory
            evidence_dir = os.path.join(self.config.get('output_directory', 'output'), 'evidence')
            os.makedirs(evidence_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            camera_name = self.get_active_camera_name().replace(' ', '_')
            filename = f"detection_{camera_name}_{object_class}_{timestamp}.jpg"
            filepath = os.path.join(evidence_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, frame)
            logging.info(f"Detection image saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            logging.error(f"Failed to save detection image: {e}")
            return None

    def send_whatsapp_alert(self, object_class, confidence, image_path):
        """Send WhatsApp alert with image."""
        try:
            # Format message
            message = f"⚠️ WEAPON DETECTION ALERT ⚠️%0A%0AObject: {object_class.upper()}%0AConfidence: {confidence*100:.1f}%%0ATime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}%0ACamera: {self.get_active_camera_name()}"
            
            # Open WhatsApp Web
            phone_number = self.whatsapp_number.replace('+', '').replace('-', '').replace(' ', '')
            url = f"https://web.whatsapp.com/send?phone={phone_number}&text={message}"
            
            webbrowser.open(url)
            
            logging.info(f"WhatsApp alert sent to {self.whatsapp_number}")
            
            # Note: Automatic image sending requires additional setup
            # User will need to manually attach the image from: {image_path}
            
        except Exception as e:
            logging.error(f"WhatsApp alert failed: {e}")

    def add_suspect_to_ui(self, suspect_crop, weapon_crop, timestamp, weapon_type="Weapon"):
        """Adds a suspect profile card to the active suspects UI panel."""
        import cv2
        from PIL import Image, ImageTk
        import datetime
        
        self.suspect_count += 1
        
        # Format images
        def format_img(cv_img, size=(120, 120)):
            if cv_img.size == 0:
                return None
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            cv_img = cv2.resize(cv_img, size)
            img = Image.fromarray(cv_img)
            return ImageTk.PhotoImage(image=img)
            
        p_img = format_img(suspect_crop, size=(100, 140))
        w_img = format_img(weapon_crop, size=(100, 100))
        
        if p_img and w_img:
            # Keep refs
            self.suspect_images_refs.extend([p_img, w_img])
            
            # Create Card Frame
            card = tk.Frame(self.suspects_inner_frame, bg="#222222", bd=1, relief=tk.RIDGE)
            card.pack(fill=tk.X, padx=5, pady=5)
            
            # Header
            header = tk.Label(card, text=f"SUSPECT_{self.suspect_count:02d}", bg="#880000", fg="white", font=('Courier', 10, 'bold'))
            header.pack(fill=tk.X)
            
            # Images container
            img_container = tk.Frame(card, bg="#222222")
            img_container.pack(fill=tk.X, pady=2)
            
            p_label = tk.Label(img_container, image=p_img, bg="#000000", bd=1)
            p_label.pack(side=tk.LEFT, padx=2)
            
            w_label = tk.Label(img_container, image=w_img, bg="#000000", bd=1)
            w_label.pack(side=tk.LEFT, padx=2)
            
            # Info container
            info_frame = tk.Frame(card, bg="#222222")
            info_frame.pack(fill=tk.X, padx=2, pady=2)
            
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            tk.Label(info_frame, text=f"TYPE: {weapon_type.upper()}", bg="#222222", fg="#ffaa00", font=('Courier', 8, 'bold')).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"TIME: {time_str}", bg="#222222", fg="#AAAAAA", font=('Courier', 8)).pack(anchor=tk.W)
            
            # Auto scroll to bottom
            self.suspects_canvas.update_idletasks()
            self.suspects_canvas.yview_moveto(1.0)
            
            # Limit the number of cards to prevent memory leak
            if len(self.suspects_inner_frame.winfo_children()) > 10:
                self.suspects_inner_frame.winfo_children()[0].destroy()
                # Also prune image refs (roughly)
                if len(self.suspect_images_refs) > 40:
                    self.suspect_images_refs = self.suspect_images_refs[-20:]

    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit the surveillance system?"):
            logging.info("Application closing...")
            
            # Stop recording if active
            if self.recording:
                self.toggle_recording()
            
            # Release all cameras
            for camera_data in self.cameras.values():
                try:
                    camera_data['capture'].release()
                except:
                    pass
            
            # Close database
            if self.db_conn:
                try:
                    self.db_conn.close()
                except:
                    pass
            
            self.root.destroy()

    def cleanup(self):
        """Cleanup resources."""
        try:
            # Release video writer
            if self.out:
                self.out.release()
            
            # Release all cameras
            for camera_data in self.cameras.values():
                try:
                    camera_data['capture'].release()
                except:
                    pass
            
            # Close database
            if self.db_conn:
                self.db_conn.close()
                
            logging.info("Resources cleaned up successfully")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
