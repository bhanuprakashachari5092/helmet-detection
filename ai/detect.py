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
    custom_model_path = 'runs/detect/helmet_guard_v1/weights/best.pt'
    if os.path.exists(custom_model_path):
        print(f"Loading CUSTOM Trained Model: {custom_model_path}")
        model = YOLO(custom_model_path)
    else:
        print("Loading Base YOLOv8 Model (Waiting for custom training)...")
        model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    model = YOLO('yolov8n.pt') 

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
        
        # 2. Run YOLO Inference on the preprocessed frame
        # Filter classes to only show Helmet related detections (Class 0 and 1 in our custom model)
        # If using base model, 0=person, 1=bicycle, 2=car, 3=motorcycle...
        # We target specific classes to keep the output clean as requested.
        results = model.predict(preprocessed_frame, conf=0.4, verbose=False, classes=[0, 1])
        
        # 3. Use YOLO's native plotting for the "Perfect Output" look
        annotated_frame = results[0].plot()
        
        # 4. Number Plate Detection (OpenCV Haar Cascade)
        # We always check for plates regardless of YOLO results
        gray_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
        for (px, py, pw, ph) in plates:
            # Draw blue box for plates to distinguish from helmet status
            cv2.rectangle(annotated_frame, (px, py), (px + pw, py + ph), (255, 100, 0), 3)
            cv2.putText(annotated_frame, "NUMBER PLATE", (px, py - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)


        # Render Processed Frame back to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit processed frame back to frontend
        sio.emit('frame', jpg_as_text)

        # Handle Alerts/Status Updates
        if len(results[0].boxes) > 0:
            try:
                requests.post(API_URL, json={
                    "status": "Detection Active",
                    "confidence": float(results[0].boxes.conf[0]),
                    "timestamp": time.strftime("%H:%M:%S")
                })
            except:
                pass
                
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
