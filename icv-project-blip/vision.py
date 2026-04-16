"""
Vision Module - Real-time Performance Optimized
Handles camera input and YOLOv8 object detection with async threading
"""

import cv2
import time
import numpy as np
import threading
import queue
from ultralytics import YOLO
from config import (
    CAMERA_SOURCE,
    CAMERA_BUFFER_SIZE,
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    DETECTION_INTERVAL,
    IOU_THRESHOLD,
    OBJECT_TRACKING_AGE_THRESHOLD,
    OBJECT_TIMEOUT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    FPS_UPDATE_INTERVAL,
    LABEL_MAPPING
)


class TrackedObject:
    """Represents a tracked object with smooth bounding box updates"""
    
    def __init__(self, name, bbox, conf):
        self.name = name
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.conf = conf
        self.age = 1
        self.last_seen = time.time()
    
    def update(self, bbox, conf):
        """Smoothly update bounding box position"""
        # Exponential moving average for smooth tracking
        alpha = 0.6  # Smoothing factor (0.6 = 60% new, 40% old)
        self.bbox = alpha * np.array(bbox, dtype=np.float32) + (1 - alpha) * self.bbox
        self.conf = conf
        self.age += 1
        self.last_seen = time.time()
    
    def get_bbox(self):
        """Return integer bounding box"""
        return self.bbox.astype(int).tolist()


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class VisionSystem:
    """Manages camera and object detection with async threading"""
    
    def __init__(self):
        # Load YOLO model
        print("Loading YOLO model...")
        self.yolo = YOLO(YOLO_MODEL)
        print("✅ YOLO loaded")
        
        # Setup camera
        print("Connecting to camera...")
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        
        if not self.cap.isOpened():
            print("❌ Failed to connect to camera!")
            raise RuntimeError("Camera connection failed")
        print("✅ Camera connected")
        
        # Queues for async processing
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to prevent buffering
        self.detection_queue = queue.Queue(maxsize=1)  # Only latest detection
        
        # Latest frame and detection results (thread-safe access)
        self.latest_frame = None
        self.latest_detections = None
        self.latest_frame_lock = threading.Lock()
        self.latest_detection_lock = threading.Lock()
        
        # Tracking state
        self.tracked_objects = {}
        self.detected_objects_history = []
        
        # FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.fps_lock = threading.Lock()
        
        # UI Status Overlay
        self.status_text = "READY"
        self.status_color = (0, 255, 0)  # Green
        self.last_answer = ""
        self.info_lock = threading.Lock()
        
        # Control flags
        self.running = True
        
        # Start async threads
        self.camera_thread = threading.Thread(target=self._camera_capture_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        
        self.camera_thread.start()
        self.detection_thread.start()
        
        print("✅ Vision system threads started")
    
    def _camera_capture_loop(self):
        """Continuously capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Update latest frame (thread-safe)
            with self.latest_frame_lock:
                self.latest_frame = small_frame.copy()
            
            # Put frame in queue for detection (drop old frames to prevent buffering)
            try:
                self.frame_queue.put_nowait(small_frame)
            except queue.Full:
                # Drop oldest frame and add new one
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(small_frame)
                except queue.Empty:
                    pass
    
    def _detection_loop(self):
        """Continuously run YOLO detection in separate thread"""
        last_detection_time = 0
        
        while self.running:
            try:
                # Get frame from queue (non-blocking with timeout)
                frame = self.frame_queue.get(timeout=0.1)
                
                current_time = time.time()
                
                # Run YOLO at specified interval
                if current_time - last_detection_time > DETECTION_INTERVAL:
                    # Run YOLO detection
                    results = self.yolo(
                        frame,
                        stream=False,
                        conf=YOLO_CONFIDENCE_THRESHOLD,
                        verbose=False
                    )
                    
                    # Process detections
                    current_detections = []
                    
                    for r in results:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            name = self.yolo.names[cls]
                            # Apply label mapping if configured
                            name = LABEL_MAPPING.get(name, name)
                            conf = float(box.conf[0])
                            bbox = list(map(int, box.xyxy[0].cpu().numpy()))
                            
                            current_detections.append({
                                'name': name,
                                'bbox': bbox,
                                'conf': conf
                            })
                    
                    # Update tracked objects
                    matched = set()
                    for det in current_detections:
                        best_match = None
                        best_iou = IOU_THRESHOLD
                        
                        for key, obj in self.tracked_objects.items():
                            if obj.name == det['name']:
                                iou = calculate_iou(det['bbox'], obj.get_bbox())
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = key
                        
                        if best_match:
                            # Update existing object
                            self.tracked_objects[best_match].update(det['bbox'], det['conf'])
                            matched.add(best_match)
                        else:
                            # Create new tracked object
                            new_key = f"{det['name']}_{len(self.tracked_objects)}_{time.time()}"
                            self.tracked_objects[new_key] = TrackedObject(
                                det['name'],
                                det['bbox'],
                                det['conf']
                            )
                            matched.add(new_key)
                    
                    # Remove objects not seen for timeout period
                    self.tracked_objects = {
                        k: v for k, v in self.tracked_objects.items()
                        if current_time - v.last_seen < OBJECT_TIMEOUT
                    }
                    
                    # Update latest detections (thread-safe)
                    with self.latest_detection_lock:
                        self.latest_detections = {
                            'tracked_objects': self.tracked_objects.copy(),
                            'timestamp': current_time
                        }
                    
                    last_detection_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Detection Error: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self):
        """Get latest frame (non-blocking, thread-safe)"""
        with self.latest_frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_latest_detections(self):
        """Get latest detection results (non-blocking, thread-safe)"""
        with self.latest_detection_lock:
            return self.latest_detections.copy() if self.latest_detections else None

    def get_latest_yolo_result(self):
        """Get flat list of current detections for VQA"""
        detections = self.get_latest_detections()
        if not detections:
            return []
        
        # Convert tracked objects back to simple list
        results = []
        for obj in detections.get('tracked_objects', {}).values():
            results.append({
                'label': obj.name,
                'confidence': obj.conf,
                'box': obj.get_bbox()
            })
        return results
    
    def draw_detections(self, frame):
        """Draw bounding boxes and labels on frame"""
        detections = self.get_latest_detections()
        if detections is None:
            return
        
        self.detected_objects_history.clear()
        tracked_objects = detections.get('tracked_objects', {})
        
        for key, obj in tracked_objects.items():
            # Only draw if object has been tracked for minimum frames
            if obj.age >= OBJECT_TRACKING_AGE_THRESHOLD:
                self.detected_objects_history.append(obj.name)
                
                x1, y1, x2, y2 = obj.get_bbox()
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label
                label = f"{obj.name} {obj.conf:.2f}"
                
                # Calculate label size
                (label_w, label_h), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - 8),
                    (x1 + label_w + 4, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
    
    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            with self.fps_lock:
                self.fps = FPS_UPDATE_INTERVAL / (time.time() - self.fps_start_time)
                self.fps_start_time = time.time()
    
    def get_fps(self):
        """Get current FPS (thread-safe)"""
        with self.fps_lock:
            return self.fps
    
    def set_status(self, text, color=(0, 255, 0)):
        """Update system status text and color"""
        with self.info_lock:
            self.status_text = text
            self.status_color = color

    def set_answer(self, text):
        """Update last answer text"""
        with self.info_lock:
            self.last_answer = text

    def draw_info(self, frame):
        """Draw FPS, Status Bar, and Answer"""
        h, w = frame.shape[:2]
        
        # 1. Draw Top Info (FPS, Count)
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {len(self.detected_objects_history)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 2. Draw Bottom Status Bar (Semi-transparent)
        bar_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        with self.info_lock:
            # Status (Left)
            cv2.putText(frame, self.status_text, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.status_color, 2)
            
            # Answer (Below Status or Next to it)
            # Truncate answer if too long
            ans = self.last_answer
            if len(ans) > 60:
                ans = ans[:57] + "..."
            cv2.putText(frame, ans, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def get_frame_for_vqa(self):
        """Get current frame for VQA processing (thread-safe)"""
        return self.get_latest_frame()
    
    def get_detected_objects(self):
        """Get list of currently detected objects"""
        return self.detected_objects_history.copy()
    
    def release(self):
        """Release camera resources"""
        self.running = False
        time.sleep(0.2)  # Give threads time to finish
        self.cap.release()
        cv2.destroyAllWindows()
