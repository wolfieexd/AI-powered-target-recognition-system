import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import torch
from pathlib import Path
from ultralytics import YOLO
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Set CustomTkinter Appearance
ctk.set_appearance_mode("Dark")  # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
ctk.deactivate_automatic_dpi_awareness()
ctk.set_widget_scaling(1.0) # Prevent scaling Tkinter crashes
ctk.set_window_scaling(1.0)

class MainDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI-Powered Target Recognition System - Pro")
        self.geometry("1400x900")
        
        # Load Configuration
        self.config_path = "config.json"
        self.config = self.load_config()

        # Load Shared Models Once to save RAM/VRAM
        self.shared_model = None
        self.load_shared_models()

        # Configure grid layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Create Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Surveillance", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_live = ctk.CTkButton(self.sidebar_frame, text="🔴 Live Surveillance", command=self.show_live_mode)
        self.btn_live.grid(row=1, column=0, padx=20, pady=10)

        self.btn_file = ctk.CTkButton(self.sidebar_frame, text="📁 File Analysis", command=self.show_file_mode)
        self.btn_file.grid(row=2, column=0, padx=20, pady=10)
        
        self.btn_settings = ctk.CTkButton(self.sidebar_frame, text="⚙️ Global Settings", command=self.show_settings)
        self.btn_settings.grid(row=3, column=0, padx=20, pady=10)

        self.btn_analytics = ctk.CTkButton(self.sidebar_frame, text="📊 Analytics & Logs", command=self.show_analytics)
        self.btn_analytics.grid(row=4, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))
        self.appearance_mode_optionemenu.set("Dark")

        # Create Main Content Area
        self.main_content = ctk.CTkFrame(self, corner_radius=10)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        # Welcome Screen
        self.welcome_label = ctk.CTkLabel(self.main_content, text="Welcome to AI-Powered Target Recognition\n\nPlease select a mode from the sidebar.",
                                         font=ctk.CTkFont(size=24, weight="bold"))
        self.welcome_label.grid(row=0, column=0)
        
        # We will hold references to the app frames so they aren't destroyed when hidden
        self.live_frame = None
        self.file_frame = None
        self.settings_frame = None
        self.analytics_frame = None
        # self.person_model is already initialized by load_shared_models
        
    def load_shared_models(self):
        """Load the heavy YOLO models once into GPU memory to be shared."""
        try:
            logging.info("Loading shared YOLO models...")
            weapon_model_path = Path('models/weapon_model.pt')
            generic_model_path = Path('models/yolov8n.pt')
            
            if weapon_model_path.exists():
                self.shared_model = YOLO(str(weapon_model_path))
                logging.info("Shared weapon model loaded.")
            else:
                logging.warning("Weapon model not found.")
                
            if generic_model_path.exists():
                self.person_model = YOLO(str(generic_model_path))
                logging.info("Shared generic model (for persons) loaded.")
            else:
                logging.warning("Generic model not found.")
                
        except Exception as e:
            logging.error(f"Failed to load shared models: {e}")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        
    def clear_content(self):
        for widget in self.main_content.winfo_children():
            widget.pack_forget()
            widget.grid_forget()

    def load_config(self):
        default_config = {
            'model_path': 'models/weapon_model.pt',
            'detection_threshold': 0.75,
            'camera_sources': ['0', '1', '2'],
            'database_path': 'detections.db',
            'output_directory': 'output',
            'log_path': 'logs/weapon_surveillance.log',
            'weapon_classes': ['guns', 'knife'],
            'alert_threshold': 0.6,
            'alerts_enabled': True,
            'whatsapp_enabled': False,
            'whatsapp_number': '+1234567890',
            'process_every_n_frames': 2,
            'intrusion_zone': []
        }
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        return default_config

    def save_config(self):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

    def show_live_mode(self):
        self.clear_content()
        if not self.live_frame:
            from optimized_surveillance_system import start_live_surveillance
            self.live_frame = ctk.CTkFrame(self.main_content)
            self.live_app = start_live_surveillance(master=self.live_frame, shared_model=self.shared_model, config=self.config, person_model=self.person_model)
            
        self.live_frame.pack(fill=tk.BOTH, expand=True)

    def show_file_mode(self):
        self.clear_content()
        if not self.file_frame:
            from file_weapon_detector import FileWeaponDetector
            self.file_frame = ctk.CTkFrame(self.main_content)
            self.file_app = FileWeaponDetector(root=self.file_frame, shared_model=self.shared_model, person_model=self.person_model)
            
        self.file_frame.pack(fill=tk.BOTH, expand=True)

    def show_settings(self):
        self.clear_content()
        if not self.settings_frame:
            from settings_dashboard import SettingsDashboard
            self.settings_frame = ctk.CTkFrame(self.main_content)
            self.settings_app = SettingsDashboard(parent_frame=self.settings_frame, main_app=self)
            
        self.settings_frame.pack(fill=tk.BOTH, expand=True)

    def show_analytics(self):
        self.clear_content()
        if not hasattr(self, 'analytics_frame') or not self.analytics_frame:
            from analytics_dashboard import AnalyticsDashboard
            self.analytics_frame = ctk.CTkFrame(self.main_content)
            self.analytics_app = AnalyticsDashboard(parent_frame=self.analytics_frame, config=self.config)
            
        # Force refresh data when shown
        if hasattr(self, 'analytics_app'):
            self.analytics_app.load_data()
            
        self.analytics_frame.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = MainDashboard()
    app.mainloop()
