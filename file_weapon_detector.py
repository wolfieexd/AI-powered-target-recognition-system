"""
File-based Weapon Detection Tool
Supports uploading images and videos for weapon detection analysis
Enhanced with preprocessing, tiled inference, and TTA
"""

# Fix FFmpeg threading issue and enable optimizations
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
# Enable OpenCV optimizations
cv2.setNumThreads(4)  # Use 4 CPU threads for parallel processing
cv2.setUseOptimized(True)  # Enable optimized code paths

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
import logging
from ultralytics import YOLO
import torch

# Import preprocessing utilities
from preprocessing_utils import ImagePreprocessor, TiledInference, TestTimeAugmentation

# Weapon classes to detect
# These match the classes from the trained weapon detection model (Weapon-2-2 dataset)
# Model classes: {0: 'guns', 1: 'knife'}
WEAPON_CLASSES = ['guns', 'knife']

# For testing: These are classes YOLOv8n CAN detect from COCO dataset
# Uncomment the line below to test with detectable objects
# WEAPON_CLASSES = ['person', 'car', 'dog', 'cat']  # Just for testing detection works

class FileWeaponDetector:
    def __init__(self, root, shared_model=None, person_model=None):
        self.root = root
        if hasattr(self.root, 'title'):
            self.root.title("🔍 File-Based Weapon Detection - Enhanced")
            self.root.geometry("1400x800")
        try:
            self.root.configure(bg='#2b2b2b')
        except:
            pass
            
        self.shared_model = shared_model
        self.person_model_shared = person_model
        self.rotation_angle = 0
        
        self.model = None
        self.device = 'cpu'  # Default to CPU, will be set to GPU if available
        self.current_file = None
        self.is_video = False
        self.video_cap = None
        self.playing = False
        self.detection_results = []
        self.total_frames = 0
        self.current_frame_num = 0
        self.fps = 30
        self.seeking = False
        self.skip_frames = 1  # Process EVERY frame for maximum detection intensity
        self.video_lock = threading.Lock()  # Thread lock for video operations
        
        # Dynamic configuration (like optimized_surveillance_system.py)
        self.config = {
            'detection_threshold': 0.20,  # Lower default for better small weapon detection
            'alert_threshold': 0.80,
            'process_every_n_frames': 2,  # Balanced mode
            'preprocessing_enabled': False,  # DISABLED: CLAHE+Gamma ruins normal YOLO detection
            'tiled_inference': False,  # NEW: Enable for large images
            'tta_enabled': False,  # NEW: Test-time augmentation
            'multi_scale': False,  # NEW: Multi-scale detection
        }
        
        # Initialize enhancement modules
        self.preprocessor = ImagePreprocessor(self.config)
        self.tiled_inference = TiledInference(tile_size=640, overlap=0.2)
        self.tta = TestTimeAugmentation(use_flips=True, use_scales=True)
        
        # Performance optimizations
        self.detection_queue = []  # Queue of frames waiting for detection
        self.max_queue_size = 1  # Minimal queue for instant processing
        self.last_detected_frame = None  # Cache last detection result
        self.detection_in_progress = False  # Flag to prevent multiple detections
        self.display_size = (800, 600)  # Larger display for better quality
        self.frame_buffer = None  # Pre-allocate frame buffer
        self.photo_cache = None  # Cache photo object
        
        # Async processing for maximum speed
        self.async_display = True  # Enable async frame display
        self.drop_frames = True  # Drop frames if rendering is slow
        
        self.setup_ui()
        self.load_model()
    
    def create_menu_bar(self):
        """Create menu bar with Settings option (like optimized_components.py)"""
        menubar = tk.Menu(self.root)
        try:
            self.root.config(menu=menubar)
        except AttributeError:
            pass # CTkFrame doesn't support native menus
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Detection Configuration", command=self.open_settings_dialog)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def open_settings_dialog(self):
        """Open settings dialog for dynamic configuration with preprocessing options"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Detection Settings - Enhanced")
        settings_window.geometry("500x600")
        settings_window.resizable(False, False)
        settings_window.transient(self.root)
        settings_window.configure(bg='#3c3c3c')
        
        # Create scrollable frame
        canvas = tk.Canvas(settings_window, bg='#3c3c3c', highlightthickness=0)
        scrollbar = tk.Scrollbar(settings_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#3c3c3c')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Header
        tk.Label(scrollable_frame, text="Detection Settings", 
                font=('Arial', 12, 'bold'), bg='#3c3c3c', fg='white').pack(pady=15)
        
        # 1. Detection Threshold
        threshold_frame = tk.LabelFrame(scrollable_frame, text="Detection Threshold",
                                       bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'),
                                       padx=15, pady=10)
        threshold_frame.pack(fill='x', padx=20, pady=10)
        
        threshold_var = tk.DoubleVar(value=self.threshold_slider.get())
        
        def update_threshold(val):
            self.threshold_slider.set(float(val))
            self.config['detection_threshold'] = float(val)
            threshold_display.config(text=f"Current: {float(val):.2f}")
        
        threshold_display = tk.Label(threshold_frame, text=f"Current: {threshold_var.get():.2f}",
                                     bg='#3c3c3c', fg='#00ff00', font=('Arial', 10, 'bold'))
        threshold_display.pack()
        
        threshold_scale = tk.Scale(threshold_frame, from_=0.1, to=0.95, resolution=0.05,
                                  orient='horizontal', variable=threshold_var,
                                  command=update_threshold, length=400,
                                  bg='#4a4a4a', fg='white', highlightthickness=0,
                                  troughcolor='#2b2b2b', activebackground='#0078d4')
        threshold_scale.pack(pady=5)
        
        # 2. Preprocessing Options (NEW)
        preprocess_frame = tk.LabelFrame(scrollable_frame, text="Image Enhancement",
                                        bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'),
                                        padx=15, pady=10)
        preprocess_frame.pack(fill='x', padx=20, pady=10)
        
        preprocess_var = tk.BooleanVar(value=self.config['preprocessing_enabled'])
        def toggle_preprocessing():
            self.config['preprocessing_enabled'] = preprocess_var.get()
            logging.info(f"Preprocessing: {'Enabled' if preprocess_var.get() else 'Disabled'}")
        
        tk.Checkbutton(preprocess_frame, text="Enable Auto-Enhancement (CLAHE + Gamma)",
                      variable=preprocess_var, command=toggle_preprocessing,
                      bg='#3c3c3c', fg='white', selectcolor='#2b2b2b',
                      activebackground='#3c3c3c', activeforeground='white',
                      font=('Arial', 9)).pack(anchor='w')
        
        tk.Label(preprocess_frame, text="Improves detection in low-light/dark images",
                bg='#3c3c3c', fg='#aaaaaa', font=('Arial', 8)).pack(anchor='w')
        
        # 3. Advanced Detection Options (NEW)
        advanced_frame = tk.LabelFrame(scrollable_frame, text="Advanced Detection",
                                      bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'),
                                      padx=15, pady=10)
        advanced_frame.pack(fill='x', padx=20, pady=10)
        
        tiled_var = tk.BooleanVar(value=self.config['tiled_inference'])
        def toggle_tiled():
            self.config['tiled_inference'] = tiled_var.get()
            logging.info(f"Tiled Inference: {'Enabled' if tiled_var.get() else 'Disabled'}")
        
        tk.Checkbutton(advanced_frame, text="Tiled Inference (for small weapons)",
                      variable=tiled_var, command=toggle_tiled,
                      bg='#3c3c3c', fg='white', selectcolor='#2b2b2b',
                      activebackground='#3c3c3c', activeforeground='white',
                      font=('Arial', 9)).pack(anchor='w', pady=2)
        
        tta_var = tk.BooleanVar(value=self.config['tta_enabled'])
        def toggle_tta():
            self.config['tta_enabled'] = tta_var.get()
            logging.info(f"Test-Time Augmentation: {'Enabled' if tta_var.get() else 'Disabled'}")
        
        tk.Checkbutton(advanced_frame, text="Test-Time Augmentation (TTA)",
                      variable=tta_var, command=toggle_tta,
                      bg='#3c3c3c', fg='white', selectcolor='#2b2b2b',
                      activebackground='#3c3c3c', activeforeground='white',
                      font=('Arial', 9)).pack(anchor='w', pady=2)
        
        multiscale_var = tk.BooleanVar(value=self.config['multi_scale'])
        def toggle_multiscale():
            self.config['multi_scale'] = multiscale_var.get()
            logging.info(f"Multi-scale Detection: {'Enabled' if multiscale_var.get() else 'Disabled'}")
        
        tk.Checkbutton(advanced_frame, text="Multi-scale Detection (slower, more accurate)",
                      variable=multiscale_var, command=toggle_multiscale,
                      bg='#3c3c3c', fg='white', selectcolor='#2b2b2b',
                      activebackground='#3c3c3c', activeforeground='white',
                      font=('Arial', 9)).pack(anchor='w', pady=2)
        
        tk.Label(advanced_frame, text="⚠️ Priority: Tiled > Multi-scale > TTA",
                bg='#3c3c3c', fg='#ffaa00', font=('Arial', 8, 'bold')).pack(anchor='w')
        tk.Label(advanced_frame, text="Only the highest priority enabled mode will be used",
                bg='#3c3c3c', fg='#aaaaaa', font=('Arial', 8)).pack(anchor='w', pady=(0, 5))
        
        # 4. Threshold Guide
        info_frame = tk.LabelFrame(scrollable_frame, text="Threshold Guide",
                                  bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'),
                                  padx=15, pady=10)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """0.20-0.30: Very Sensitive (more detections, may have false positives)
0.30-0.40: Sensitive (good for small/distant weapons)
0.45-0.55: Balanced (recommended for general use)
0.60-0.75: Strict (fewer false positives)
0.80-0.95: Very Strict (only obvious weapons)"""
        
        tk.Label(info_frame, text=info_text, bg='#3c3c3c', fg='white',
                font=('Arial', 8), justify='left').pack()
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(settings_window, text="✓ Apply & Close", command=settings_window.destroy,
                 bg='#0078d4', fg='white', font=('Arial', 10, 'bold'),
                 padx=30, pady=10).pack(pady=20)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About",
                          "File-Based Weapon Detection Tool\n\n"
                          "Uses YOLOv8 AI for real-time weapon detection\n"
                          "in images and videos.\n\n"
                          "Features:\n"
                          "• Dynamic threshold adjustment\n"
                          "• GPU acceleration\n"
                          "• Real-time processing\n\n"
                          "Version: 2.0 - Dynamic Config")
        
    def setup_ui(self):
        # Create menu bar
        self.create_menu_bar()
        
        # Top Control Panel
        control_frame = tk.Frame(self.root, bg='#3c3c3c', height=80)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        control_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(control_frame, text="🎯 Weapon Detection - Upload File", 
                              font=('Arial', 16, 'bold'), bg='#3c3c3c', fg='white')
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg='#3c3c3c')
        btn_frame.pack(side=tk.RIGHT, padx=20)
        
        self.upload_btn = tk.Button(btn_frame, text="📁 Upload Image", command=self.upload_image,
                                    bg='#0078d4', fg='white', font=('Arial', 11, 'bold'),
                                    width=15, height=2, cursor='hand2')
        self.upload_btn.grid(row=0, column=0, padx=5)
        
        self.upload_vid_btn = tk.Button(btn_frame, text="🎬 Upload Video", command=self.upload_video,
                                        bg='#0078d4', fg='white', font=('Arial', 11, 'bold'),
                                        width=15, height=2, cursor='hand2')
        self.upload_vid_btn.grid(row=0, column=1, padx=5)
        
        self.play_btn = tk.Button(btn_frame, text="▶️ Play", command=self.toggle_play,
                                  bg='#107c10', fg='white', font=('Arial', 11, 'bold'),
                                  width=10, height=2, state=tk.DISABLED, cursor='hand2')
        self.play_btn.grid(row=0, column=2, padx=5)
        
        self.detect_btn = tk.Button(btn_frame, text="🔍 Detect", command=self.detect_current_frame,
                                    bg='#ff8c00', fg='white', font=('Arial', 11, 'bold'),
                                    width=10, height=2, state=tk.DISABLED, cursor='hand2')
        self.detect_btn.grid(row=0, column=3, padx=5)
        
        self.rotate_btn = tk.Button(btn_frame, text="↻ Rotate 90°", command=self.rotate_current_image,
                                    bg='#0078d4', fg='white', font=('Arial', 11, 'bold'),
                                    width=10, height=2, state=tk.DISABLED, cursor='hand2')
        self.rotate_btn.grid(row=0, column=4, padx=5)
        
        # Create tooltip-like label for Detect button
        tk.Label(btn_frame, text="(Re-run with current threshold)", 
                bg='#3c3c3c', fg='#aaaaaa', font=('Arial', 8)).grid(row=1, column=3)
        
        self.save_btn = tk.Button(btn_frame, text="💾 Save Result", command=self.save_result,
                                  bg='#8a3ffc', fg='white', font=('Arial', 11, 'bold'),
                                  width=12, height=2, state=tk.DISABLED, cursor='hand2')
        self.save_btn.grid(row=0, column=5, padx=5)
        

        # Main Content Area
        content_frame = tk.Frame(self.root, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left: Display Area
        display_frame = tk.LabelFrame(content_frame, text="📺 Detection View", 
                                     bg='#3c3c3c', fg='white', font=('Arial', 11, 'bold'))
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.canvas = tk.Canvas(display_frame, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        # Video Controls Frame (timeline + info)
        video_controls_frame = tk.Frame(display_frame, bg='#3c3c3c', height=60)
        video_controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        video_controls_frame.pack_propagate(False)
        
        # Timeline/Seekbar
        timeline_frame = tk.Frame(video_controls_frame, bg='#3c3c3c')
        timeline_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.timeline_scale = tk.Scale(timeline_frame, from_=0, to=100, 
                                       orient=tk.HORIZONTAL, command=self.seek_video,
                                       bg='#4a4a4a', fg='white', highlightthickness=0,
                                       troughcolor='#2b2b2b', activebackground='#0078d4',
                                       state=tk.DISABLED, showvalue=False)
        self.timeline_scale.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        # Video info label
        self.video_info_label = tk.Label(video_controls_frame, text="", 
                                         bg='#3c3c3c', fg='#aaaaaa', font=('Arial', 9))
        self.video_info_label.pack(pady=(0, 5))
        
        # Placeholder text
        self.placeholder_text = self.canvas.create_text(
            400, 300, text="Upload an image or video to start detection",
            fill='#666666', font=('Arial', 14)
        )
        
        # Right: Results Panel
        results_frame = tk.LabelFrame(content_frame, text="📊 Detection Results", 
                                     bg='#3c3c3c', fg='white', font=('Arial', 11, 'bold'),
                                     width=400)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        results_frame.pack_propagate(False)
        
        # Status
        self.status_label = tk.Label(results_frame, text="Status: Ready", 
                                    bg='#3c3c3c', fg='#00ff00', font=('Arial', 10, 'bold'))
        self.status_label.pack(pady=5)
        
        # Detection Threshold Slider (Moved from top bar)
        threshold_frame = tk.Frame(results_frame, bg='#3c3c3c')
        threshold_frame.pack(fill=tk.X, padx=10, pady=0)
        
        tk.Label(threshold_frame, text="Detection Threshold:", bg='#3c3c3c', fg='white',
                font=('Arial', 10)).pack(anchor='center')
        
        self.threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.1, to=0.95, resolution=0.05,
            orient='horizontal',
            length=250,
            bg='#4a4a4a', fg='white',
            highlightthickness=0,
            troughcolor='#2b2b2b',
            activebackground='#0078d4',
            showvalue=True
        )
        self.threshold_slider.set(self.config['detection_threshold'])
        self.threshold_slider.pack(anchor='center')
        
        # Results Tree
        tree_frame = tk.Frame(results_frame, bg='#3c3c3c')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        
        columns = ('Object', 'Confidence', 'Weapon?')
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                                        yscrollcommand=vsb.set, height=6)
        vsb.config(command=self.results_tree.yview)
        
        self.results_tree.heading('Object', text='Object Class')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.heading('Weapon?', text='Weapon?')
        
        self.results_tree.column('Object', width=150)
        self.results_tree.column('Confidence', width=100)
        self.results_tree.column('Weapon?', width=100)
        
        self.results_tree.tag_configure('weapon', background='#ff4444', foreground='white')
        self.results_tree.tag_configure('normal', background='#3c3c3c', foreground='white')
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Suspects Panel (bottom half)
        suspects_outer = tk.LabelFrame(results_frame, text="[!] ACTIVE SUSPECTS", bg='#1a1a1a', fg='#ff4444', font=('Courier', 10, 'bold'))
        suspects_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.suspects_canvas = tk.Canvas(suspects_outer, bg='#1a1a1a', highlightthickness=0)
        self.suspects_scrollbar = ttk.Scrollbar(suspects_outer, orient="vertical", command=self.suspects_canvas.yview)
        
        self.suspects_inner_frame = tk.Frame(self.suspects_canvas, bg='#1a1a1a')
        
        self.suspects_inner_frame.bind(
            "<Configure>",
            lambda e: self.suspects_canvas.configure(
                scrollregion=self.suspects_canvas.bbox("all")
            )
        )
        
        self.suspects_canvas.create_window((0, 0), window=self.suspects_inner_frame, anchor="nw", width=350)
        self.suspects_canvas.configure(yscrollcommand=self.suspects_scrollbar.set)
        
        # Make canvas scrollable with mouse wheel
        def _on_mousewheel(event):
            self.suspects_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.suspects_canvas.bind("<Enter>", lambda e: self.suspects_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.suspects_canvas.bind("<Leave>", lambda e: self.suspects_canvas.unbind_all("<MouseWheel>"))
        
        self.suspects_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.suspects_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.suspect_count = 0
        self.suspect_images_refs = []
        self.suspects_logged_track_ids = set()
        
        # Summary
        self.summary_label = tk.Label(results_frame, text="No detections yet", 
                                     bg='#3c3c3c', fg='white', font=('Arial', 9),
                                     wraplength=350, justify=tk.LEFT)
        self.summary_label.pack(pady=10, padx=10)
        
    def load_model(self):
        """Load YOLOv8 model(s) - supports ensemble mode with both weapon and generic models"""
        try:
            self.status_label.config(text="Status: Loading AI Model...", fg='#ffaa00')
            self.root.update()
            
            # Check for available models
            weapon_model_path = Path('models/weapon_model.pt')
            generic_model_path = Path('models/yolov8n.pt')
            
            self.model = None
            self.generic_model = None
            
            if hasattr(self, 'shared_model') and self.shared_model is not None:
                self.model = self.shared_model
                logging.info("✅ Using shared YOLO model from Dashboard to save RAM/VRAM!")
                if hasattr(self, 'person_model_shared') and self.person_model_shared is not None:
                    self.generic_model = self.person_model_shared
                    logging.info("✅ Using shared YOLO generic model for person tracking!")
                else:
                    if generic_model_path.exists():
                        self.generic_model = self._load_single_model(generic_model_path, "Generic Model")
            else:
                # Try to load weapon-specific model
                if weapon_model_path.exists():
                    logging.info("Loading weapon-specific model...")
                    self.model = self._load_single_model(weapon_model_path, "Weapon Model")
                    
                # Try to load generic model for ensemble
                if generic_model_path.exists():
                    logging.info("Loading generic YOLOv8 model for ensemble detection...")
                    self.generic_model = self._load_single_model(generic_model_path, "Generic Model")
            
            # Check if at least one model loaded
            if self.model is None and self.generic_model is None:
                messagebox.showerror("Error", 
                    "No model found!\n\n"
                    "Please download a model:\n"
                    "1. Weapon model: save as models/weapon_model.pt\n"
                    "2. Or generic YOLOv8: save as models/yolov8n.pt")
                return
            
            # Use weapon model as primary if available
            if self.model is None:
                self.model = self.generic_model
                self.generic_model = None
                logging.warning("Using generic YOLOv8 only - weapons may not be detected!")
            
            # Log ensemble status
            if self.generic_model is not None:
                logging.info("✅ ENSEMBLE MODE: Both weapon and generic models loaded!")
                logging.info("   This will detect more objects and improve accuracy")
            
            # Configure GPU
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logging.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f}GB)")
                
                # Prevent OOM on 4GB cards like RTX 3050 by disabling ensemble
                # BUT if we are using shared models from the Dashboard, they are already loaded in VRAM anyway!
                if gpu_mem <= 4.5 and self.model is not None and self.generic_model is not None:
                    if hasattr(self, 'person_model_shared') and self.person_model_shared is not None:
                        logging.info("Low VRAM detected, but keeping generic model since it's shared from dashboard.")
                    else:
                        logging.warning(f"⚠️ Low VRAM detected ({gpu_mem:.1f}GB)!")
                        logging.warning("Disabling generic YOLOv8 ensemble to prevent Out of Memory (OOM) crashes.")
                        self.generic_model = None
                
                logging.info("Model will use GPU for faster detection")
                
                # Enable GPU optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.set_device(0)
                
                # Warm up GPU
                import numpy as np
                logging.info("Warming up GPU with FP16 precision...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                for _ in range(5):
                    self.model(dummy_frame, device=self.device, imgsz=640, half=True, verbose=False)
                if self.generic_model:
                    for _ in range(5):
                        self.generic_model(dummy_frame, device=self.device, imgsz=640, half=True, verbose=False)
                
                torch.cuda.empty_cache()
                logging.info("GPU optimizations enabled: FP16, TF32, CUDNN benchmarking")
                logging.info("GPU warmed up and ready")
            else:
                self.device = 'cpu'
                logging.info("GPU not available, using CPU")
            
            # Log model classes
            model_classes = self.model.names
            logging.info(f"📋 Model classes: {model_classes}")
            
            # Check weapon classes
            has_weapons = any(cls.lower() in ['gun', 'guns', 'knife', 'pistol', 'rifle', 'weapon'] 
                            for cls in model_classes.values())
            
            if has_weapons:
                logging.info("✅ Weapon classes detected in model - ready for weapon detection!")
            else:
                logging.warning("⚠️ No weapon classes found in primary model!")
            
            status_text = f"Status: Models Loaded ✓"
            if self.device == 'cuda:0':
                status_text += " (GPU)"
            if self.generic_model:
                status_text += " + Ensemble"
            self.status_label.config(text=status_text, fg='#00ff00')
            logging.info(f"YOLOv8 model loaded successfully: weapon_model.pt on {self.device}")
            
        except Exception as e:
            self.status_label.config(text=f"Status: Error loading model", fg='#ff0000')
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}")
            logging.error(f"Model loading error: {e}")
    
    def _load_single_model(self, model_path, model_name):
        """Load a single YOLO model"""
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Temporarily set torch.load to use weights_only=False
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        try:
            model = YOLO(str(model_path))
            logging.info(f"✓ Loaded {model_name}: {model_path.name}")
            return model
        except Exception as e:
            logging.error(f"Failed to load {model_name}: {e}")
            return None
        finally:
            torch.load = original_load
    
    def upload_image(self):
        """Upload and analyze an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.is_video = False
        self.playing = False
        self.rotation_angle = 0
        self.play_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.NORMAL)  # Enable detect button for re-detection
        self.rotate_btn.config(state=tk.NORMAL)  # Enable rotate button
        
        # Process image in thread
        threading.Thread(target=self.process_image, args=(file_path,), daemon=True).start()
    
    def upload_video(self):
        """Upload and analyze a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.is_video = True
        self.playing = False
        
        # Process first frame
        threading.Thread(target=self.process_video, args=(file_path,), daemon=True).start()
    
    def process_image(self, file_path):
        """Process single image for weapon detection with advanced preprocessing"""
        try:
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded. Please restart the application.")
                return
            
            self.status_label.config(text="Status: Processing image...", fg='#ffaa00')
            
            # Read image and fix EXIF orientation
            from PIL import Image, ImageOps
            import numpy as np
            try:
                pil_image = Image.open(file_path)
                pil_image = ImageOps.exif_transpose(pil_image)
                # Convert to RGB if it's not (e.g. RGBA or P)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Apply manual rotation if requested
                if self.rotation_angle != 0:
                    # PIL rotate: counter-clockwise, so we use -self.rotation_angle for clockwise
                    pil_image = pil_image.rotate(-self.rotation_angle, expand=True)
                
                # Convert to OpenCV BGR
                open_cv_image = np.array(pil_image)
                original_frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logging.warning(f"Failed to read/rotate image with PIL: {e}. Falling back to cv2.imread.")
                original_frame = cv2.imread(file_path)
                
                # Apply manual rotation to cv2 fallback
                if self.rotation_angle != 0:
                    # Rotate 90 clockwise = cv2.ROTATE_90_CLOCKWISE
                    # Handle multiple 90 degree rotations
                    for _ in range((self.rotation_angle % 360) // 90):
                        original_frame = cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE)
                
            if original_frame is None:
                messagebox.showerror("Error", "Failed to read image file")
                return
            
            frame = original_frame.copy()
            
            # Use dynamic threshold from slider
            dynamic_threshold = self.threshold_slider.get()
            logging.info(f"🔍 Running detection with threshold: {dynamic_threshold:.2f}")
            
            # ===== STEP 1: Preprocessing (if enabled) =====
            if self.config.get('preprocessing_enabled', True):
                logging.info("📸 Applying image preprocessing (CLAHE + Gamma)...")
                frame = self.preprocessor.preprocess(frame)
            
            # ===== STEP 2: Choose detection strategy =====
            # Priority: Tiled > Multi-scale > TTA > Standard
            # Only ONE advanced mode should be active at a time for best results
            all_detections = []
            
            # Determine which strategy to use (priority order)
            use_tiled = self.config.get('tiled_inference', False)
            use_multiscale = self.config.get('multi_scale', False) and not use_tiled
            use_tta = self.config.get('tta_enabled', False) and not use_tiled and not use_multiscale
            
            # Option A: Tiled Inference (HIGHEST PRIORITY - best for small weapons)
            if use_tiled:
                logging.info("🔲 Using Tiled Inference for small object detection...")
                tiles = self.tiled_inference.split_image_to_tiles(frame)
                logging.info(f"  Split into {len(tiles)} tiles")
                
                for tile_img, x_offset, y_offset in tiles:  # Fixed: unpack 3 values, not 2
                    results = self.model(
                        tile_img,
                        conf=dynamic_threshold * 0.8,  # Lower threshold for tiles to catch more
                        iou=0.45,
                        imgsz=640,
                        device=self.device,
                        half=torch.cuda.is_available(),
                        verbose=False,
                        agnostic_nms=False,
                        max_det=100
                    )
                    
                    # Log each tile (even if empty)
                    tile_det_count = len(results[0].boxes) if results and results[0].boxes is not None else 0
                    if tile_det_count > 0:
                        logging.info(f"  Tile at ({x_offset}, {y_offset}): {tile_det_count} detections")
                    
                    # Adjust coordinates back to full image
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf)
                                cls_name = result.names[int(box.cls)]
                                logging.info(f"    - {cls_name} @ ({x1+x_offset},{y1+y_offset}) conf={conf:.3f}")
                                all_detections.append({
                                    'xyxy': [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset],
                                    'conf': conf,
                                    'cls': int(box.cls),
                                    'class_name': cls_name
                                })
                
                logging.info(f"  Total detections before NMS: {len(all_detections)}")
                logging.info(f"  Detection threshold used: {dynamic_threshold * 0.8:.3f} (80% of slider value for tiles)")
                
                # Merge overlapping detections from tiles
                all_detections = self.tiled_inference.merge_tile_detections(
                    all_detections, 
                    image_shape=frame.shape[:2],
                    iou_threshold=0.3  # Lower threshold to keep more detections
                )
                logging.info(f"  After NMS merge: {len(all_detections)} detections")
            
            # Option B: Multi-scale Detection (2nd priority)
            elif use_multiscale:
                logging.info("📏 Using Multi-scale Detection...")
                scales = [0.6, 0.8, 1.0, 1.2, 1.4]  # More scales including smaller (0.6) and larger (1.4)
                
                for scale in scales:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    scaled_frame = cv2.resize(frame, (new_w, new_h))
                    
                    logging.info(f"  Processing at scale {scale:.1f}x ({new_w}x{new_h})...")
                    
                    results = self.model(
                        scaled_frame,
                        conf=0.10,  # Very low fixed threshold to catch everything
                        iou=0.45,
                        imgsz=640,
                        device=self.device,
                        half=torch.cuda.is_available(),
                        verbose=False,
                        agnostic_nms=False,
                        max_det=300  # Allow many detections
                    )
                    
                    # Scale coordinates back
                    scale_det_count = 0
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf)
                                cls_name = result.names[int(box.cls)]
                                
                                # Scale back to original size
                                orig_x1, orig_y1 = int(x1/scale), int(y1/scale)
                                orig_x2, orig_y2 = int(x2/scale), int(y2/scale)
                                
                                # Log ALL detections to see what's being detected
                                logging.info(f"    - {cls_name} @ ({orig_x1},{orig_y1}) conf={conf:.3f}")
                                
                                all_detections.append({
                                    'xyxy': [orig_x1, orig_y1, orig_x2, orig_y2],
                                    'conf': conf,
                                    'cls': int(box.cls),
                                    'class_name': cls_name
                                })
                                scale_det_count += 1
                    
                    if scale_det_count > 0:
                        logging.info(f"    Found {scale_det_count} detections at this scale")
                
                logging.info(f"  Total detections before NMS: {len(all_detections)}")
                
                # Apply NMS to merged multi-scale results
                all_detections = self.tiled_inference.merge_tile_detections(
                    all_detections,
                    image_shape=frame.shape[:2],
                    iou_threshold=0.3  # Lower to keep more detections
                )
                logging.info(f"  After NMS merge: {len(all_detections)} detections")
            
            # Option C: Test-Time Augmentation (3rd priority - currently disabled due to complexity)
            elif use_tta:
                logging.warning("⚠️ TTA mode detected - using standard detection instead")
                logging.warning("   (TTA integration needs refinement, falling back to standard)")
                use_tta = False  # Force standard detection
            
            # Option D: Standard single-pass detection (default or fallback)
            # This is the primary detection pipeline used when Tiled/Multiscale are disabled.
            # It passes the entire raw image frame into the YOLOv8 neural network.
            if not use_tiled and not use_multiscale:
                logging.info("⚡ Using Standard Detection...")
                
                # Core Detection Inference Step
                # The model() call handles everything: resizing to 640x640, normalizing pixels,
                # passing through the network layers, and extracting bounding boxes.
                results = self.model(
                    frame,
                    conf=dynamic_threshold,  # Minimum confidence threshold (e.g. 0.45 = 45% certainty)
                    iou=0.30,                # Intersection over Union threshold for Non-Max Suppression (NMS) to merge overlapping duplicate boxes

                    imgsz=640,
                    device=self.device,
                    half=torch.cuda.is_available(),
                    verbose=False,
                    agnostic_nms=True,
                    max_det=100,
                    augment=False
                )
                
                # Convert to unified format
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            all_detections.append({
                                'xyxy': [x1, y1, x2, y2],
                                'conf': float(box.conf),
                                'cls': int(box.cls),
                                'class_name': result.names[int(box.cls)]
                            })
            
            # ===== STEP 3: Process unified detections =====
            self.detection_results = []
            weapons_found = 0
            total_detections = len(all_detections)
            
            logging.info(f"✓ Found {total_detections} detection(s)")
            
            # Draw detections on original frame for display
            display_frame = original_frame.copy()
            
            # Clear previous suspects from the UI
            for widget in self.suspects_inner_frame.winfo_children():
                widget.destroy()
            self.suspect_count = 0
            self.suspect_images_refs.clear()
            
            for detection in all_detections:
                x1, y1, x2, y2 = detection['xyxy']
                conf = detection['conf']
                obj_class = detection['class_name']
                
                logging.info(f"  - Detected: {obj_class} (confidence: {conf:.3f})")
                
                # Basic Classification Step
                # The YOLO network outputs an integer class ID (e.g., 0 for person, 1 for guns).
                # We map this ID to a string name via `result.names`, yielding `obj_class`.
                # We then strictly classify if this detected object represents a localized threat 
                # by checking it against our known WEAPON_CLASSES list.
                is_weapon = obj_class.lower() in WEAPON_CLASSES
                
                if is_weapon:
                    logging.info(f"    ⚠️ WEAPON DETECTED: {obj_class}")
                    weapons_found += 1

                    
                    # Cinematic Suspects Tracking Logic
                    # If a weapon is detected, we need to find out WHO is holding it.
                    # We run a secondary generic YOLO model specifically to find "person" bounding boxes (class 0).
                    if hasattr(self, 'generic_model') and self.generic_model is not None:
                        person_results = self.generic_model.predict(original_frame, conf=0.2, classes=[0], verbose=False)
                        h, w = original_frame.shape[:2]
                        suspect_found = False
                        for p_res in person_results:
                            if p_res.boxes is not None and len(p_res.boxes) > 0:
                                for p_box in p_res.boxes:
                                    # Extract coordinates for the detected person
                                    px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
                                    
                                    # Heuristic Spatial Intersection 
                                    # People often hold weapons outside the strict boundaries of their torso bounding box.
                                    # We artificially expand the person's bounding box outward by a 100-pixel margin
                                    # to reliably encapsulate hands and extended arms holding the weapon.
                                    margin = 100
                                    epx1, epy1 = px1 - margin, py1 - margin
                                    epx2, epy2 = px2 + margin, py2 + margin

                                    
                                    wx_c, wy_c = (x1 + x2) / 2, (y1 + y2) / 2
                                    if (epx1 <= wx_c <= epx2) and (epy1 <= wy_c <= epy2) or (
                                        x1 < epx2 and x2 > epx1 and y1 < epy2 and y2 > epy1):
                                        # Crop the original person, not the expanded box
                                        suspect_crop = original_frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)].copy()
                                        weapon_crop = original_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                                        
                                        self.root.after(0, lambda p=suspect_crop, w_crop=weapon_crop, t=time.time(), wt=obj_class: self.add_suspect_to_ui(p, w_crop, t, wt))
                                        suspect_found = True
                                        break
                            if suspect_found:
                                break
                else:
                    logging.info(f"    ℹ️ Non-weapon: {obj_class}")
                
                # Draw on frame
                color = (0, 0, 255) if is_weapon else (0, 255, 0)
                thickness = 4 if is_weapon else 2
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                label = f"{obj_class} ({conf:.2f})"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add to results
                self.detection_results.append({
                    'class': obj_class,
                    'confidence': conf,
                    'is_weapon': is_weapon
                })
            
            # Log summary
            if total_detections == 0:
                logging.warning(f"⚠️ NO DETECTIONS at threshold {dynamic_threshold:.2f}")
                logging.info(f"💡 Try lowering the threshold (currently {dynamic_threshold:.2f})")
                logging.info(f"💡 Recommended: Try 0.20 - 0.35 for more sensitive detection")
                if not self.config.get('preprocessing_enabled', True):
                    logging.info(f"💡 Try enabling preprocessing in Settings for low-light images")
            else:
                logging.info(f"✓ Total detections: {total_detections}, Weapons: {weapons_found}")
            
            # Add warning if weapons found
            if weapons_found > 0:
                cv2.putText(display_frame, f"[!] {weapons_found} WEAPON(S) DETECTED", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Add info text if no detections
            if total_detections == 0:
                cv2.putText(display_frame, f"No detections at threshold {dynamic_threshold:.2f}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                cv2.putText(display_frame, "Try lowering the threshold slider", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Store processed frame
            self.processed_frame = display_frame
            
            # Display the frame with bounding boxes drawn
            self.display_frame(display_frame)
            self.update_results_panel()
            
            # Update status with detection count
            if total_detections == 0:
                self.status_label.config(text=f"Status: No detections (threshold: {dynamic_threshold:.2f})", fg='#ffaa00')
            else:
                self.status_label.config(text=f"Status: Detection Complete ✓ ({total_detections} found)", fg='#00ff00')
            
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            import traceback
            self.status_label.config(text="Status: Error", fg='#ff0000')
            error_details = traceback.format_exc()
            messagebox.showerror("Processing Error", f"Error processing image:\n{str(e)}\n\nSee console for details")
            logging.error(f"Image processing error: {e}")
            logging.error(f"Full traceback:\n{error_details}")
    
    def process_video(self, file_path):
        """Process video for weapon detection"""
        try:
            self.status_label.config(text="Status: Loading video...", fg='#ffaa00')
            
            # Stop any playing video first
            self.playing = False
            time.sleep(0.05)  # Reduced wait time
            
            # Open video with optimized settings for speed
            with self.video_lock:
                if self.video_cap:
                    self.video_cap.release()
                
                # Use CAP_FFMPEG with optimizations for maximum speed
                self.video_cap = cv2.VideoCapture(file_path, cv2.CAP_FFMPEG)
                
                # Optimize video reading for speed
                self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for low latency
                self.video_cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)  # Hardware decoding
                
                if not self.video_cap.isOpened():
                    messagebox.showerror("Error", "Failed to open video file")
                    return
                
                # Get video properties
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = int(self.video_cap.get(cv2.CAP_PROP_FPS)) or 30
                self.current_frame_num = 0
            
            # Enable timeline
            self.timeline_scale.config(to=self.total_frames - 1, state=tk.NORMAL)
            
            # Process first frame
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame.copy()  # Store current frame
                self.display_frame(frame)
                self.play_btn.config(state=tk.NORMAL)
                self.detect_btn.config(state=tk.NORMAL)
                self.update_video_info()
                self.status_label.config(text="Status: Video loaded. Click Play ▶️ or Detect 🔍", fg='#00ff00')
            
        except Exception as e:
            self.status_label.config(text="Status: Error", fg='#ff0000')
            messagebox.showerror("Video Error", f"Error loading video:\n{str(e)}")
    
    def process_video_frame(self, frame):
        """Process a single video frame with maximum speed and precision"""
        if self.model is None or self.detection_in_progress:
            return
        
        try:
            self.detection_in_progress = True
            
            # Use dynamic threshold from slider (like optimized_surveillance_system.py)
            dynamic_threshold = self.threshold_slider.get()
            
            # GPU-optimized detection with dynamic threshold
            results = self.model(
                frame,
                conf=dynamic_threshold,  # Dynamic threshold from slider
                device=self.device,
                imgsz=640,
                verbose=False,
                half=torch.cuda.is_available()  # FP16 for 2x speedup on GPU
            )
            
            # Clear previous results
            self.detection_results = []
            weapons_found = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        obj_class = result.names[int(box.cls)]
                        
                        is_weapon = obj_class.lower() in WEAPON_CLASSES
                        
                        color = (0, 0, 255) if is_weapon else (0, 255, 0)
                        thickness = 4 if is_weapon else 2
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        label = f"{obj_class} ({conf:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        self.detection_results.append({
                            'class': obj_class,
                            'confidence': conf,
                            'is_weapon': is_weapon
                        })
                        
                        if is_weapon:
                            weapons_found += 1
                            
                            # Cinematic Suspects Tracking Logic
                            if hasattr(self, 'generic_model') and self.generic_model is not None:
                                person_results = self.generic_model.predict(frame, conf=0.2, classes=[0], verbose=False)
                                h, w = frame.shape[:2]
                                suspect_found = False
                                for p_res in person_results:
                                    if p_res.boxes is not None and len(p_res.boxes) > 0:
                                        for p_box in p_res.boxes:
                                            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
                                            # Expand person box by 100 pixels in all directions to catch weapons held in hands
                                            margin = 100
                                            epx1, epy1 = px1 - margin, py1 - margin
                                            epx2, epy2 = px2 + margin, py2 + margin
                                            
                                            wx_c, wy_c = (x1 + x2) / 2, (y1 + y2) / 2
                                            if (epx1 <= wx_c <= epx2) and (epy1 <= wy_c <= epy2) or (
                                                x1 < epx2 and x2 > epx1 and y1 < epy2 and y2 > epy1):
                                                
                                                # Use pseudo track_id based on rounded coordinates to prevent spamming the UI in videos
                                                pseudo_track_id = hash((px1//50, py1//50, px2//50, py2//50))
                                                if pseudo_track_id not in self.suspects_logged_track_ids:
                                                    suspect_crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)].copy()
                                                    weapon_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                                                    
                                                    self.root.after(0, lambda p=suspect_crop, w_crop=weapon_crop, t=time.time(), wt=obj_class: self.add_suspect_to_ui(p, w_crop, t, wt))
                                                    self.suspects_logged_track_ids.add(pseudo_track_id)
                                                suspect_found = True
                                                break
                                    if suspect_found:
                                        break
            
            if weapons_found > 0:
                cv2.putText(frame, f"[!] {weapons_found} WEAPON(S) DETECTED", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Cache the detected frame
            self.last_detected_frame = frame.copy()
            
        finally:
            self.detection_in_progress = False
        
        self.processed_frame = frame
        self.display_frame(frame)
        self.update_results_panel()
    
    def detect_current_frame(self):
        """Run detection on current paused frame OR re-detect loaded image"""
        # If it's an image, re-run detection on the original image file
        if not self.is_video and hasattr(self, 'current_file') and self.current_file:
            self.status_label.config(text="Status: Re-running detection...", fg='#ffaa00')
            logging.info(f"Re-detecting image with new threshold: {self.threshold_slider.get():.2f}")
            threading.Thread(target=self.process_image, args=(self.current_file,), daemon=True).start()
        # If it's a video, detect on current paused frame
        elif hasattr(self, 'current_frame') and self.current_frame is not None:
            self.status_label.config(text="Status: Running detection...", fg='#ffaa00')
            threading.Thread(target=self.process_video_frame, args=(self.current_frame.copy(),), daemon=True).start()
        else:
            messagebox.showinfo("Info", "Please upload an image or pause the video first")
            
    def rotate_current_image(self):
        """Rotate the currently loaded image by 90 degrees clockwise and re-detect"""
        if not self.is_video and hasattr(self, 'current_file') and self.current_file:
            self.rotation_angle = (self.rotation_angle + 90) % 360
            self.detect_current_frame()
        elif self.is_video:
            messagebox.showinfo("Info", "Rotation is only supported for static images.")
        else:
            messagebox.showinfo("Info", "Please upload an image first.")
    
    def toggle_play(self):
        """Play/pause video"""
        self.playing = not self.playing
        
        if self.playing:
            self.play_btn.config(text="⏸️ Pause")
            threading.Thread(target=self.play_video, daemon=True).start()
        else:
            self.play_btn.config(text="▶️ Play")
    
    def play_video(self):
        """Ultra-fast video playback with frame dropping for real-time performance"""
        frame_count = 0
        last_display_time = time.time()
        target_frame_time = 1.0 / self.fps if self.fps > 0 else 0.033  # Target time per frame
        
        while self.playing and self.video_cap and self.video_cap.isOpened():
            loop_start = time.time()
            
            # Skip reading if currently seeking
            if self.seeking:
                time.sleep(0.001)
                continue
            
            # Read frame WITHOUT lock for maximum speed
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame_num = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            if not ret:
                # Video ended, reset to beginning
                with self.video_lock:
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_num = 0
                self.playing = False
                self.play_btn.config(text="▶️ Play")
                continue
            
            # Store current frame for on-demand detection
            self.current_frame = frame.copy()
            
            # MAXIMUM DETECTION INTENSITY: Launch detection on EVERY frame (completely non-blocking)
            if not self.detection_in_progress:
                threading.Thread(target=self.process_video_frame, args=(frame.copy(),), daemon=True).start()
            
            # Check if we should skip display to maintain real-time playback
            current_time = time.time()
            elapsed_since_last_display = current_time - last_display_time
            
            # Always display or drop frames to maintain FPS
            if elapsed_since_last_display >= target_frame_time or not self.drop_frames:
                self.display_frame(frame)
                last_display_time = current_time
            # else: skip this frame to catch up
            
            # Update timeline less frequently
            if frame_count % 15 == 0:
                self.timeline_scale.set(self.current_frame_num)
                self.update_video_info()
            
            frame_count += 1
            
            # Aggressive RAM cleanup for limited memory systems
            if frame_count % 60 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Sleep only if we're ahead of schedule
            elapsed = time.time() - loop_start
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
        
        if not self.playing:
            self.status_label.config(text="Status: Video paused", fg='#ffaa00')
    
    def seek_video(self, value):
        """Seek to specific frame in video - thread-safe"""
        if not self.video_cap or not self.video_cap.isOpened():
            return
        
        # Pause playback during seeking to prevent concurrent access
        was_playing = self.playing
        self.playing = False
        time.sleep(0.05)  # Allow play thread to finish
        
        self.seeking = True
        frame_num = int(float(value))
        self.current_frame_num = frame_num
        
        # Use lock to prevent concurrent video access
        with self.video_lock:
            try:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.video_cap.read()
                
                if ret:
                    # Store current frame for on-demand detection
                    self.current_frame = frame.copy()
                    # Just display the frame, no automatic detection
                    self.root.after(0, lambda: self.display_frame(frame))
                    self.root.after(10, self.update_video_info)
            except Exception as e:
                logging.error(f"Seek error: {e}")
        
        self.seeking = False
        
        # Resume playback if it was playing before
        if was_playing:
            self.playing = True
            threading.Thread(target=self.play_video, daemon=True).start()
    
    def update_video_info(self):
        """Update video timeline information"""
        if self.total_frames > 0:
            current_time = self.current_frame_num / self.fps
            total_time = self.total_frames / self.fps
            
            info_text = f"Frame: {self.current_frame_num}/{self.total_frames} | " \
                       f"Time: {int(current_time//60):02d}:{int(current_time%60):02d} / " \
                       f"{int(total_time//60):02d}:{int(total_time%60):02d} | " \
                       f"FPS: {self.fps}"
            
            self.video_info_label.config(text=info_text)
    
    def display_frame(self, frame):
        """Display frame on canvas - extreme speed optimization"""
        # Convert to RGB and resize in one optimized operation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.display_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Fast canvas update - delete and recreate in single operation
        if self.placeholder_text:
            self.canvas.delete(self.placeholder_text)
            self.placeholder_text = None
        else:
            self.canvas.delete("all")
        
        # Center and display
        canvas_width = self.canvas.winfo_width() or 800
        canvas_height = self.canvas.winfo_height() or 600
        x = (canvas_width - self.display_size[0]) // 2
        y = (canvas_height - self.display_size[1]) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        y = (canvas_height - self.display_size[1]) // 2
        
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
    
    def update_results_panel(self):
        """Update the results tree and summary"""
        # Clear tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Count weapons
        weapons = sum(1 for r in self.detection_results if r['is_weapon'])
        total = len(self.detection_results)
        
        # Add results
        for result in self.detection_results:
            tag = 'weapon' if result['is_weapon'] else 'normal'
            weapon_text = "⚠️ YES" if result['is_weapon'] else "No"
            
            self.results_tree.insert('', 'end', 
                                    values=(result['class'], 
                                           f"{result['confidence']:.2%}",
                                           weapon_text),
                                    tags=(tag,))
        
        # Update summary
        if total == 0:
            summary = "No objects detected in current frame"
        else:
            summary = f"Total Detections: {total}\n"
            if weapons > 0:
                summary += f"⚠️ WEAPONS FOUND: {weapons}\n"
                summary += f"Non-weapons: {total - weapons}"
            else:
                summary += "✓ No weapons detected"
        
        self.summary_label.config(text=summary)
    
    def save_result(self):
        """Save processed image/frame"""
        if not hasattr(self, 'processed_frame'):
            messagebox.showwarning("Warning", "No processed frame to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("PNG Image", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_frame)
            messagebox.showinfo("Success", f"Result saved to:\n{file_path}")
    
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

    def __del__(self):
        """Cleanup"""
        if self.video_cap:
            self.video_cap.release()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    app = FileWeaponDetector(root)
    root.mainloop()
