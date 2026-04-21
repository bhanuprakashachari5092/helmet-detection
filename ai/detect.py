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
# Load YOLO model
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
    while not sio.connected:
        try:
            print(f"Attempting to connect to {SERVER_URL}...")
            sio.connect(SERVER_URL, wait_timeout=10)
            break
        except Exception as e:
            print(f"Connection error: {e}. Retrying in 5s...")
            time.sleep(5)

@sio.on('process_this_frame')
def on_process_frame(data):
    print("AI: Received frame for processing...")
    """
    Called when frontend sends a frame either from WebCam or Uploads
    """
    global last_detection_time
    
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
        results = model(preprocessed_frame, stream=False, verbose=False)
        
        highest_conf = 0.0
        detected = False
        current_status = "No Detection"
        
        # Number Plate Detection using OpenCV Haar Cascades
        gray_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (px, py, pw, ph) in plates:
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 3) # Blue box for plates
            cv2.putText(frame, "Number Plate", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            detected = True
            
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # YOLOv8 default model (COCO) classes: 0: person, 3: motorcycle
                if cls == 0: # Person
                    detected = True
                    highest_conf = max(highest_conf, conf)
                    # Simulated logic for helmet
                    if conf > 0.65:
                        label, color, current_status = "Helmet", (0, 255, 0), "Helmet"
                    else:
                        label, color, current_status = "No Helmet", (0, 0, 255), "No Helmet"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{label} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                elif cls == 3: # Motorcycle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    cv2.putText(frame, "Motorcycle", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Render Processed Frame back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit processed frame back to frontend (ALWAYS)
        sio.emit('frame', jpg_as_text)

        # Send detection info if detected and cooldown passed
        if detected and (time.time() - last_detection_time > cooldown):
            try:
                requests.post(API_URL, json={
                    "status": current_status,
                    "confidence": highest_conf,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                last_detection_time = time.time()
            except Exception as e:
                print(f"Error sending detection alert: {e}")
                
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Fallback: send original frame so the UI doesn't freeze
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            sio.emit('frame', jpg_as_text)
        except:
            pass

if __name__ == "__main__":
    start_socket()
    sio.wait()
