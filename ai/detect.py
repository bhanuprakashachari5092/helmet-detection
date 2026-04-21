import cv2
import base64
import time
import requests
import socketio
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image

import os

# Configuration
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:5000")
API_URL = f"{SERVER_URL}/api/detection"

# Load the Best Trained Model if available, else fallback to base YOLOv8
try:
    # 1. Custom Helmet Model
    custom_model_path = 'best.pt'
    model_helmet = YOLO(custom_model_path)
    print(f"✅ Final Helmet Model Loaded: {custom_model_path}")
    
    # 2. Base YOLO Model for Motorcycle Detection (COCO Class 3)
    model_base = YOLO('yolov8n.pt')
    print("✅ Base Model Loaded for Motorcycle Tracking")
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback
    model_helmet = YOLO('yolov8n.pt')
    model_base = model_helmet

# Load OpenCV Number Plate Cascade
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Initialize Socket.io client with reconnection logic
sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=5)

last_detection_time = 0
cooldown = 1.0 # 1 second cooldown between logging detections

@sio.event
def connect():
    print(f"✅ Connected to backend server: {SERVER_URL}")

@sio.event
def connect_error(data):
    print(f"❌ Connection failed: {data}")

@sio.event
def disconnect():
    print("🔌 Disconnected from server")

# Attempt to connect with retries
def start_socket():
    print("🚀 Neural Engine: Initializing AI Detection Pipeline...")
    while not sio.connected:
        try:
            print(f"📡 Attempting to bridge to server at {SERVER_URL}...")
            # Verify if server is reachable via HTTP first
            try:
                requests.get(SERVER_URL, timeout=5)
                print("✅ Server is reachable via HTTP")
            except:
                print("⚠️ Server not reachable via HTTP yet, but trying Socket.io...")
                
            sio.connect(SERVER_URL, wait_timeout=10)
            print("💎 Neural Link Established: Helmet Detection Active")
            break
        except Exception as e:
            print(f"❌ Connection error: {e}. Retrying in 3s...")
            time.sleep(3)

import easyocr

# Initialize EasyOCR Reader (English)
# This may download models on the first run
try:
    reader = easyocr.Reader(['en'], gpu=False) # Keep gpu=False for stability on CPU-heavy systems
    print("✅ EasyOCR Reader Initialized")
except Exception as e:
    print(f"⚠️ EasyOCR Init Error: {e}")
    reader = None

# Frame counter for OCR throttle
ocr_frame_count = 0

@sio.on('process_this_frame')
def on_process_frame(data):
    global ocr_frame_count
    ocr_frame_count += 1
    
    try:
        # Decode base64 frame
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Dual Model Inference
        # We run the base model to find bikes (for plate search) but don't draw them
        results_base = model_base.predict(frame, conf=0.3, verbose=False, classes=[3])
        
        # Start with a clean frame (no default YOLO boxes)
        annotated_frame = frame.copy()

        # 2. Detect Helmets using Custom Model (Classes 0, 1)
        results_helmet = model_helmet.predict(frame, conf=0.4, verbose=False, classes=[0, 1])

        
        for box in results_helmet[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = "HELMET" if cls == 0 else "NO HELMET"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(annotated_frame, f"{label} {conf:.1%}", (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        # 2. Optimized Number Plate Detection + OCR
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for bike in results_base[0].boxes:
            bx1, by1, bx2, by2 = map(int, bike.xyxy[0])
            # Ensure ROI is within bounds
            by1, by2 = max(0, by1), min(frame.shape[0], by2)
            bx1, bx2 = max(0, bx1), min(frame.shape[1], bx2)
            
            bike_roi_gray = gray_frame[by1:by2, bx1:bx2]
            bike_roi_color = frame[by1:by2, bx1:bx2]
            
            plates = plate_cascade.detectMultiScale(bike_roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (px, py, pw, ph) in plates:
                fx, fy = bx1 + px, by1 + py
                
                # Draw Plate Box
                cv2.rectangle(annotated_frame, (fx, fy), (fx + pw, fy + ph), (0, 165, 255), 3)
                
                # Perform OCR (Throttled to every 10 frames for performance)
                plate_text = "SCANNING..."
                if reader and ocr_frame_count % 10 == 0:
                    plate_roi = bike_roi_color[py:py+ph, px:px+pw]
                    if plate_roi.size > 0:
                        ocr_res = reader.readtext(plate_roi)
                        if ocr_res:
                            plate_text = ocr_res[0][1].upper()
                
                # Draw Tech-style label with OCR Text
                cv2.rectangle(annotated_frame, (fx, fy - 35), (fx + 220, fy), (0, 165, 255), -1)
                cv2.putText(annotated_frame, f"ID: {plate_text}", (fx + 5, fy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)




        # Render Processed Frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit processed frame back to frontend
        sio.emit('frame', jpg_as_text)

        # Handle Alerts/Status Updates
        if len(results_helmet[0].boxes) > 0:
            try:
                # Get the class ID of the first detection
                class_id = int(results_helmet[0].boxes.cls[0])
                status = "Helmet" if class_id == 0 else "No Helmet"
                
                requests.post(API_URL, json={
                    "status": status,
                    "confidence": float(results_helmet[0].boxes.conf[0]),
                    "timestamp": time.strftime("%H:%M:%S")
                })

            except Exception as e:
                print(f"Error sending detection alert: {e}")

                
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Fallback to original frame so UI doesn't freeze
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            sio.emit('frame', base64.b64encode(buffer).decode('utf-8'))
        except:
            pass


if __name__ == "__main__":
    start_socket()
    sio.wait()
