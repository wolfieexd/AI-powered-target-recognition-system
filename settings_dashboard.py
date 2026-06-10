import customtkinter as ctk
import tkinter as tk

class SettingsDashboard:
    def __init__(self, parent_frame, main_app):
        self.parent = parent_frame
        self.main_app = main_app
        self.config = main_app.config

        self.setup_ui()

    def setup_ui(self):
        # Create a scrollable frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self.parent, width=600, height=800)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(self.scrollable_frame, text="Global Settings", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20), anchor="w")

        # 1. Detection Settings
        det_frame = ctk.CTkFrame(self.scrollable_frame)
        det_frame.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        ctk.CTkLabel(det_frame, text="Detection Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10, padx=10, anchor="w")
        
        self.det_thresh_label = ctk.CTkLabel(det_frame, text=f"Detection Threshold: {self.config.get('detection_threshold', 0.75):.2f}")
        self.det_thresh_label.pack(padx=20, anchor="w")
        self.det_thresh_slider = ctk.CTkSlider(det_frame, from_=0.1, to=0.99, command=self.update_det_label)
        self.det_thresh_slider.set(self.config.get('detection_threshold', 0.75))
        self.det_thresh_slider.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.alert_thresh_label = ctk.CTkLabel(det_frame, text=f"Alert Threshold: {self.config.get('alert_threshold', 0.6):.2f}")
        self.alert_thresh_label.pack(padx=20, anchor="w")
        self.alert_thresh_slider = ctk.CTkSlider(det_frame, from_=0.1, to=0.99, command=self.update_alert_label)
        self.alert_thresh_slider.set(self.config.get('alert_threshold', 0.6))
        self.alert_thresh_slider.pack(fill=tk.X, padx=20, pady=(0, 20))

        # 2. Alert & Notification Settings
        alert_frame = ctk.CTkFrame(self.scrollable_frame)
        alert_frame.pack(fill=tk.X, pady=(0, 20), padx=10)
        
        ctk.CTkLabel(alert_frame, text="Alerts & Notifications", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10, padx=10, anchor="w")
        
        self.sound_alert_var = ctk.BooleanVar(value=self.config.get('alerts_enabled', True))
        ctk.CTkSwitch(alert_frame, text="Enable Sound & Screen Alerts", variable=self.sound_alert_var).pack(padx=20, pady=10, anchor="w")

        # 3. Save Button
        save_btn = ctk.CTkButton(self.scrollable_frame, text="Save Settings", command=self.save_settings, fg_color="green", hover_color="darkgreen")
        save_btn.pack(pady=20)

    def update_det_label(self, value):
        self.det_thresh_label.configure(text=f"Detection Threshold: {value:.2f}")

    def update_alert_label(self, value):
        self.alert_thresh_label.configure(text=f"Alert Threshold: {value:.2f}")

    def save_settings(self):
        self.config['detection_threshold'] = self.det_thresh_slider.get()
        self.config['alert_threshold'] = self.alert_thresh_slider.get()
        self.config['alerts_enabled'] = self.sound_alert_var.get()
        
        # Save to disk via main_dashboard
        self.main_app.config = self.config
        self.main_app.save_config()
        
        # Optional: update live app if it's running
        if getattr(self.main_app, 'live_app', None):
            self.main_app.live_app.config = self.config
            
        tk.messagebox.showinfo("Settings Saved", "Global configuration has been updated successfully!")
