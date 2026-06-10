import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import sqlite3
import os

class AnalyticsDashboard:
    def __init__(self, parent_frame, config):
        self.parent = parent_frame
        self.config = config
        self.db_path = config.get('database_path', 'detections.db')
        
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        ctk.CTkLabel(self.main_frame, text="Incident Analytics & Logs", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0, 20), anchor="w")

        # Stats bar
        self.stats_frame = ctk.CTkFrame(self.main_frame)
        self.stats_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.total_lbl = ctk.CTkLabel(self.stats_frame, text="Total Threats Logged: 0", font=ctk.CTkFont(size=16, weight="bold"))
        self.total_lbl.pack(side=tk.LEFT, padx=20, pady=10)

        # Treeview for logs
        columns = ("id", "timestamp", "camera", "class", "confidence", "track_id")
        self.tree = ttk.Treeview(self.main_frame, columns=columns, show="headings", height=20)
        
        self.tree.heading("id", text="ID")
        self.tree.heading("timestamp", text="Timestamp")
        self.tree.heading("camera", text="Camera")
        self.tree.heading("class", text="Class")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("track_id", text="Track ID")

        self.tree.column("id", width=50)
        self.tree.column("timestamp", width=150)
        self.tree.column("camera", width=80)
        self.tree.column("class", width=100)
        self.tree.column("confidence", width=100)
        self.tree.column("track_id", width=100)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Refresh button
        refresh_btn = ctk.CTkButton(self.parent, text="Refresh Data", command=self.load_data)
        refresh_btn.pack(pady=10)

    def load_data(self):
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if not os.path.exists(self.db_path):
            self.total_lbl.configure(text="Database not found.")
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if track_id column exists, if not query without it
            cursor.execute("PRAGMA table_info(detections)")
            columns = [col[1] for col in cursor.fetchall()]
            
            has_track = 'track_id' in columns
            query = "SELECT id, timestamp, camera_index, object_class, confidence"
            if has_track:
                query += ", track_id"
            query += " FROM detections ORDER BY id DESC LIMIT 100"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                if not has_track:
                    row = list(row) + ["N/A"]
                
                # Format confidence
                row_list = list(row)
                row_list[4] = f"{row_list[4]:.2f}"
                self.tree.insert("", tk.END, values=row_list)
                
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM detections")
            total = cursor.fetchone()[0]
            self.total_lbl.configure(text=f"Total Threats Logged: {total}")
            
            conn.close()
        except Exception as e:
            self.total_lbl.configure(text=f"Error loading database: {e}")
