import cv2
# Enable OpenCV optimizations for better performance
cv2.setNumThreads(4)  # Use 4 CPU threads
cv2.setUseOptimized(True)  # Enable optimized code paths

import argparse
import logging
import threading
import torch
import os
import time
import gc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from queue import Queue

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, DFL, Bottleneck, SPPF
from ultralytics.nn.modules.head import Detect
from torch.nn import Sequential
import torch.nn

from optimized_components import SystemComponents

# Global variables
is_running = True

# Weapon classes to detect (matches file_weapon_detector.py)
WEAPON_CLASSES = ['guns', 'knife']

@contextmanager
def allow_yolo_model_loading():
    """Context manager for safe YOLO model loading."""
    safe_classes = [
        DetectionModel, Sequential, Conv, C2f, DFL, Bottleneck, SPPF, Concat,
        torch.nn.Conv2d, torch.nn.modules.batchnorm.BatchNorm2d,
        torch.nn.modules.activation.SiLU, torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.upsampling.Upsample, Detect,
        torch.nn.modules.container.Sequential, torch.nn.modules.container.ModuleList,
    ]
    torch.serialization.add_safe_globals(safe_classes)
    yield

def setup_logging(log_path):
    """Configure logging."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def multi_camera_detection_worker(model, components, frame_queue, person_model=None):
    """Multi-camera detection worker optimized for smooth video feed and good detection."""
    global is_running
    
    frame_counter = 0
    fps_counter = 0
    fps_start_time = time.time()
    last_detection_frame = None
    
    # Pre-allocate CUDA tensors for faster processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable TF32 for faster computation on Ampere GPUs (RTX 30 series)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set CUDA to blocking mode for better GPU utilization
        torch.cuda.set_device(0)
        logging.info("GPU optimizations enabled: TF32, CUDNN benchmarking")
    
    logging.info("Weapon detection worker started - DYNAMIC CONFIG MODE")
    
    while is_running:
        loop_start = time.time()
        try:
            active_cap = components.get_active_camera()
            if not active_cap or not active_cap.isOpened():
                time.sleep(0.1)
                continue
            
            ret, frame = active_cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from camera {components.active_camera_index}")
                time.sleep(0.05)
                continue
            
            frame_counter += 1
            fps_counter += 1
            
            # Calculate FPS
            if fps_counter >= 30:
                fps_elapsed = time.time() - fps_start_time
                current_fps = fps_counter / fps_elapsed if fps_elapsed > 0 else 0
                camera_name = components.get_active_camera_name()
                # Show GPU utilization if available
                if torch.cuda.is_available():
                    gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                    components.update_status_bar(f"{camera_name} - {current_fps:.1f} FPS | GPU: {gpu_util:.0f}%")
                else:
                    components.update_status_bar(f"{camera_name} - {current_fps:.1f} FPS")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Use the currently active camera's frame for detection
            display_frame = frame.copy()
            
            # Apply thermal effect if enabled
            if components.thermal_mode:
                display_frame = cv2.applyColorMap(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY),
                    cv2.COLORMAP_JET
                )
            
            # AI Processing - Read processing mode from config dynamically
            process_every_n_frames = components.config.get('process_every_n_frames', 2)
            should_process_ai = (frame_counter % process_every_n_frames == 0)
            
            if should_process_ai:
                # Use original frame without preprocessing for natural look
                detection_frame = display_frame.copy()
                
                try:
                    with allow_yolo_model_loading():
                        # Use GPU with optimized settings for accurate detection
                        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                        
                        # Use dynamic threshold from configuration/GUI slider
                        dynamic_threshold = components.threshold_slider.get()
                        
                        # Optimized detection parameters - uses GUI slider value
                        results = model.track(
                            detection_frame, 
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=dynamic_threshold,  # Dynamic threshold from GUI/config
                            iou=0.30,   # Strict IoU threshold for NMS to reduce overlaps
                            imgsz=640,  # Higher resolution for better detection
                            device=device, 
                            half=torch.cuda.is_available(),  # FP16 for 2x speedup
                            stream=False,  # Synchronous processing
                            verbose=False,
                            agnostic_nms=True,  # Class-agnostic NMS to merge overlapping false boxes
                            max_det=100,  # Allow more detections
                        )
                    
                    # Draw detections on the frame (synchronous for max GPU usage)
                    process_detection_results_optimized(results, detection_frame, components, person_model)
                    last_detection_frame = detection_frame
                    
                except Exception as e:
                    logging.error(f"Detection error: {e}")
            
            # Use the last detection frame if available, otherwise use current frame
            frame_to_display = last_detection_frame if last_detection_frame is not None else display_frame
                
            # Put frame in queue for GUI thread to update
            try:
                frame_queue.put_nowait(frame_to_display.copy())
            except:
                pass  # Queue full, skip this frame
            
            # Handle recording
            if components.recording and components.out:
                components.out.write(display_frame)
            
            # Smooth frame rate control - limit to 30 FPS for smooth playback dynamically
            elapsed = time.time() - loop_start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)
            
            # Memory cleanup more frequently for limited RAM
            if frame_counter % 150 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            logging.error(f"Error in detection worker: {e}")
            time.sleep(0.1)

def process_detection_results_optimized(results, frame, components, person_model=None):
    """Process weapon detection results with enhanced accuracy - matches file_weapon_detector.py"""
    stats = {'detections': 0, 'weapons': 0, 'total_confidence': 0.0}
    
    try:
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
                
            for box in result.boxes:
                try:
                    conf = float(box.conf.cpu().item())
                    cls_id = int(box.cls.cpu().item())
                    
                    # Get class name
                    obj_class = result.names[cls_id] if cls_id in result.names else 'unknown'
                    
                    # Check if detected object is a weapon
                    is_weapon = obj_class.lower() in WEAPON_CLASSES
                    track_id = int(box.id.cpu().item()) if box.id is not None else 0
                    
                    # Apply adaptive thresholds based on object type
                    min_threshold = components.threshold_slider.get()
                    
                    # Stricter threshold for weapons to reduce false positives
                    detection_threshold = min_threshold if is_weapon else min_threshold - 0.1
                    
                    if conf >= detection_threshold:
                        stats['detections'] += 1
                        stats['total_confidence'] += conf
                        
                        # Get coordinates with bounds checking
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Ensure coordinates are within frame bounds
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # False Positive Rejection Heuristics
                        box_width = x2 - x1
                        box_height = y2 - y1
                        box_area = box_width * box_height
                        frame_area = w * h
                        
                        # 1. Size constraint: Weapons rarely take up more than 35% of the entire camera view
                        # This prevents large objects like curtains from being flagged
                        if box_area > frame_area * 0.35:
                            continue
                            
                        # 2. Aspect Ratio constraint for furniture:
                        aspect_ratio = box_width / max(1, box_height)
                        if box_area > frame_area * 0.10 and (0.7 < aspect_ratio < 1.3):
                            # Moderately large square objects are usually chairs/TVs, not guns/knives
                            continue
                            
                        # Create enhanced label with class name, confidence, and track ID
                        label = f"ID:{track_id} {obj_class.upper()}: {conf:.2f}"
                        
                        if is_weapon:
                            stats['weapons'] += 1
                        
                        # Draw detection with appropriate styling
                        color = (0, 0, 255) if is_weapon else (0, 255, 0)  # Red for weapons, green for others
                        thickness = 4 if is_weapon else 2
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw label background for better visibility
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
                        
                        # Background rectangle for label
                        cv2.rectangle(frame, 
                                    (x1, label_y - label_size[1] - 5), 
                                    (x1 + label_size[0] + 5, label_y + 5), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(frame, label, (x1 + 2, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add corner markers for weapons (more prominent)
                        if is_weapon:
                            corner_length = 20
                            corner_thickness = 3
                            # Top-left corner
                            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
                            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
                            # Top-right corner
                            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
                            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
                            # Bottom-left corner
                            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
                            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
                            # Bottom-right corner
                            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
                            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
                        # Trigger alerts and logging
                        if conf >= min_threshold: # Use the user's base threshold for alerts
                            if not hasattr(components, 'logged_track_ids'):
                                components.logged_track_ids = set()
                                
                            # Intrusion zone logic
                            in_zone = True
                            if hasattr(components, 'config') and components.config.get('intrusion_zone'):
                                zone = components.config.get('intrusion_zone')
                                if len(zone) >= 3:
                                    import numpy as np
                                    cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                                    pts = np.array(zone, np.int32)
                                    hull = cv2.convexHull(pts)
                                    dist = cv2.pointPolygonTest(hull, (cx, cy), False)
                                    in_zone = dist >= 0
                            
                            is_new_weapon = (track_id == 0 or track_id not in components.logged_track_ids)
                            if in_zone and is_new_weapon:
                                if track_id != 0:
                                    components.logged_track_ids.add(track_id)
                                    
                                if hasattr(components, 'trigger_high_confidence_alert'):
                                    components.trigger_high_confidence_alert(obj_class, conf, frame)
                                log_detection_async(components.db_conn, getattr(components, 'active_camera_id', 0), obj_class, conf, is_weapon, track_id, components)
                                
                            # Suspect Tracking (Full Body + Weapon) - keeps trying until suspect is captured
                            if not hasattr(components, 'suspects_logged_track_ids'):
                                components.suspects_logged_track_ids = set()
                                
                            logging.info(f"DEBUG: in_zone={in_zone}, is_weapon={is_weapon}, person_model={type(person_model)}, track_id={track_id}")
                            
                            if in_zone and is_weapon and person_model is not None and (track_id == 0 or track_id not in components.suspects_logged_track_ids):
                                # Run person detection on the current frame
                                person_results = person_model.predict(frame, conf=0.2, classes=[0], verbose=False)
                                suspect_found = False
                                person_boxes = []
                                for p_res in person_results:
                                    if p_res.boxes is not None and len(p_res.boxes) > 0:
                                        for p_box in p_res.boxes:
                                            px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
                                            person_boxes.append((px1, py1, px2, py2))
                                            
                                            # Check if person box intersects with weapon box
                                            wx_c, wy_c = (x1 + x2) / 2, (y1 + y2) / 2
                                            
                                            if (px1 <= wx_c <= px2) and (py1 <= wy_c <= py2) or (
                                                x1 < px2 and x2 > px1 and y1 < py2 and y2 > py1):
                                                
                                                # Found the suspect!
                                                suspect_crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)].copy()
                                                weapon_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
                                                
                                                # Send to GUI thread
                                                if hasattr(components, 'add_suspect_to_ui'):
                                                    components.root.after(0, lambda p=suspect_crop, w=weapon_crop, t=time.time(), wt=obj_class: components.add_suspect_to_ui(p, w, t, wt))
                                                    
                                                face_dir = os.path.join(components.config.get('output_directory', 'output'), 'suspects')
                                                os.makedirs(face_dir, exist_ok=True)
                                                cv2.imwrite(os.path.join(face_dir, f'suspect_body_{track_id}_{int(time.time())}.jpg'), suspect_crop)
                                                suspect_found = True
                                                logging.info(f"Suspect tracked for weapon ID {track_id}!")
                                                break # Only map to one person
                                    if suspect_found:
                                        break
                                
                                if suspect_found and track_id != 0:
                                    components.suspects_logged_track_ids.add(track_id)
                                elif not suspect_found:
                                    logging.warning(f"Suspect tracking failed for weapon {track_id}. People detected: {len(person_boxes)}. Weapon Box: {(x1, y1, x2, y2)}. Person Boxes: {person_boxes}")
                        
                        # Trigger high confidence alert ONLY for weapons
                        if is_weapon and conf >= components.alert_threshold:
                            components.trigger_high_confidence_alert(obj_class, conf, frame)
                            
                except Exception as e:
                    import traceback
                    logging.error(f"Error processing detection box: {traceback.format_exc()}")
                    continue
        
        # Add warning text if weapons detected
        if stats['weapons'] > 0:
            # Use ASCII text to avoid encoding issues
            warning_text = f"!!! WARNING: {stats['weapons']} WEAPON(S) DETECTED !!!"
            cv2.putText(frame, warning_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
    except Exception as e:
        logging.error(f"Error processing detection results: {e}")
    
    # Update stats
    with components.stats_lock:
        components.detection_stats['detections'] = stats['detections']
        components.detection_stats['weapons'] = stats['weapons']
        components.detection_stats['avg_confidence'] = (
            stats['total_confidence'] / stats['detections'] if stats['detections'] > 0 else 0.0
        )

def log_detection_async(conn, cam_idx, obj_class, conf, is_weapon, track_id, components):
    """Async database logging."""
    def log_worker():
        try:
            with components.db_lock:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                c = conn.cursor()
                c.execute('''INSERT INTO detections
                             (timestamp, camera_index, object_class, confidence, weapon_detected, track_id)
                             VALUES (?, ?, ?, ?, ?, ?)''',
                          (timestamp, cam_idx, obj_class, conf, int(is_weapon), track_id))
                conn.commit()
        except Exception as e:
            logging.error(f"Database logging failed: {e}")
    
    threading.Thread(target=log_worker, daemon=True).start()

def start_live_surveillance(master=None, shared_model=None, config=None, person_model=None):
    """Embeddable main function."""
    components = None
    try:
        logging.info("🔫 Starting AI-Powered Weapon Detection System...")
        
        # Try to retrieve video sources from config
        camera_sources = config.get('camera_sources', ['0']) if config else ['0']
        
        # Initialize system components
        components = SystemComponents(master=master, config=config, shared_model=shared_model)
        
        import cv2
        components.logged_track_ids = set()
        components.logged_track_ids = set()
        
        config = components.config

        # Setup logging
        setup_logging(config['log_path'])

        # Initialize cameras
        components.initialize_cameras()
        if not components.cameras:
            raise RuntimeError("No cameras could be initialized. Cannot continue.")

        logging.info(f"Initialized {len(components.cameras)} camera(s)")

        # Load weapon detection model
        if shared_model is not None:
            model = shared_model
            model_path = "Shared Model (Dashboard)"
            logging.info("✅ Using shared YOLO model from Dashboard!")
        else:
            with allow_yolo_model_loading():
                weapon_model_path = Path('models/weapon_model.pt')
                generic_model_path = Path('models/yolov8n.pt')
                
                if weapon_model_path.exists():
                    model_path = weapon_model_path
                    logging.info("Using weapon-specific model")
                elif generic_model_path.exists():
                    model_path = generic_model_path
                    logging.warning("Using generic YOLOv8 - weapons may not be detected!")
                    logging.warning("Download a weapon detection model for better results")
                else:
                    raise RuntimeError("No model found! Please place weapon_model.pt or yolov8n.pt in models/ folder")
                
                model = YOLO(str(model_path))
            
            # GPU optimization and warm-up for maximum performance
            if torch.cuda.is_available():
                logging.info("GPU detected, applying maximum optimizations...")
                
                # Enable CUDNN benchmarking for optimal convolution algorithms
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Enable TF32 for faster computation on Ampere GPUs (RTX 30 series)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Set device and pre-allocate memory
                torch.cuda.set_device(0)
                
                # Warm-up GPU with multiple iterations for JIT compilation
                import numpy as np
                logging.info("Warming up GPU with FP16 precision...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                for _ in range(5):  # Multiple warm-up iterations
                    model(dummy_frame, device='cuda:0', imgsz=640, half=True, verbose=False)
                
                torch.cuda.empty_cache()
                
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
                logging.info("Model optimized: FP16 precision, CUDNN benchmark, TF32 enabled")
                logging.info("🚀 Maximum GPU utilization mode activated!")
            else:
                logging.info("GPU not available, using CPU")

        logging.info(f"🎯 Weapon detection model loaded: {model_path}")

        # Create frame queue for thread-safe GUI updates
        frame_queue = Queue(maxsize=2)

        # Start detection thread
        detection_thread = threading.Thread(
            target=multi_camera_detection_worker,
            args=(model, components, frame_queue, person_model),
            daemon=True,
            name="WeaponDetectionWorker"
        )
        detection_thread.start()

        # GUI update scheduler for statistics
        def update_gui_stats():
            if is_running:
                components.update_statistics_display()
                components.root.after(1000, update_gui_stats)
        
        # GUI update scheduler for video frames
        def update_video_frame():
            if is_running:
                try:
                    frame = frame_queue.get_nowait()
                    components.update_video_feed_smooth(frame)
                except:
                    pass  # Queue empty, skip this update
                components.root.after(33, update_video_frame)  # 30 FPS

        update_gui_stats()
        update_video_frame()

        logging.info("🚀 Starting weapon detection GUI...")
        components.update_status_bar("🔫 Weapon Detection System Ready")

        # Start main GUI loop if standalone
        if master is None:
            components.root.mainloop()
            
            # Cleanup only when mainloop exits (standalone mode)
            global is_running
            is_running = False
            if components and components.db_conn:
                components.db_conn.close()
            logging.info("Weapon detection system shutdown completed.")
            
        return components

    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)
        is_running = False
        if components and getattr(components, 'db_conn', None):
            components.db_conn.close()

if __name__ == "__main__":
    start_live_surveillance()
