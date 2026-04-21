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

@sio.on('process_this_frame')
def on_process_frame(data):
    """
    Core AI Processing Engine
    """
    print(f"🧠 AI Analysis: Frame received at {time.strftime('%H:%M:%S')} - Processing with YOLOv8 & OpenCV...")
    try:
        # data may start with 'data:image/jpeg;base64,' -- if so, strip it
        if "base64," in data:
            data = data.split("base64,")[1]
            
        # Decode base64 to OpenCV format
        img_data = base64.b64decode(data)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return

        # 1. OpenCV Image Preprocessing (Contrast Enhancement using CLAHE)
        # This makes the detection more robust in different lighting conditions
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        preprocessed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # 2. Run Dual Model Inference
        
        # A. Detect Motorcycles using Base Model (Class 3 is motorcycle)
        results_base = model_base.predict(preprocessed_frame, conf=0.3, verbose=False, classes=[3])
        # Plot base results (Motorcycles)
        annotated_frame = results_base[0].plot()

        # B. Detect Helmets using Custom Model (Classes 0, 1)
        results_helmet = model_helmet.predict(preprocessed_frame, conf=0.4, verbose=False, classes=[0, 1])
        
        # Manually plot helmet results onto the already annotated frame
        # This gives us full control over both detections
        for box in results_helmet[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = "HELMET" if cls == 0 else "NO HELMET"
            color = (0, 255, 0) if cls == 0 else (0, 0, 255) # Green for Helmet, Red for No Helmet
            
            # Draw Thick Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)
            # Add Label with Glow effect
            cv2.putText(annotated_frame, f"{label} {conf:.1%}", (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        
        # 3. Optimized Number Plate Detection (Search within Motorcycle regions)
        # We look for plates specifically on or near the bikes we found
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If we found bikes, we narrow the search area
        for bike in results_base[0].boxes:
            bx1, by1, bx2, by2 = map(int, bike.xyxy[0])
            # Focus on the lower part of the motorcycle detection for the plate
            bike_roi = gray_frame[by1:by2, bx1:bx2]
            
            plates = plate_cascade.detectMultiScale(bike_roi, scaleFactor=1.1, minNeighbors=5)
            for (px, py, pw, ph) in plates:
                # Convert ROI coordinates back to main frame coordinates
                fx, fy = bx1 + px, by1 + py
                cv2.rectangle(annotated_frame, (fx, fy), (fx + pw, fy + ph), (0, 165, 255), 3)
                
                # Tech-style label for Plate
                cv2.rectangle(annotated_frame, (fx, fy - 25), (fx + 140, fy), (0, 165, 255), -1)
                cv2.putText(annotated_frame, "REG. PLATE", (fx + 5, fy - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Fallback: General search if no bikes were detected (less accurate)
        if len(results_base[0].boxes) == 0:
            plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10)
            for (px, py, pw, ph) in plates:
                cv2.rectangle(annotated_frame, (px, py), (px + pw, py + ph), (0, 165, 255), 2)
                cv2.putText(annotated_frame, "PLATE", (px, py - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)



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
