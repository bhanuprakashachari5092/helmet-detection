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

# Initialize Socket.io client
sio = socketio.Client()

last_detection_time = 0
cooldown = 1.0 # 1 second cooldown between logging detections

@sio.event
def connect():
    print("Connected to backend server - Awaiting frames from Frontend System")

@sio.event
def disconnect():
    print("Disconnected from backend server")

@sio.on('process_this_frame')
def on_process_frame(data):
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

        # Run YOLO Inference
        results = model(frame, stream=False, verbose=False) # stream=False for single frames
        
        current_status = "No Helmet" # Demo default
        highest_conf = 0.0
        detected = False
        
        # Number Plate Detection using OpenCV Haar Cascades
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (px, py, pw, ph) in plates:
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 3) # Blue box for plates
            cv2.putText(frame, "Number Plate", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # cls 0 is person, cls 3 is motorcycle
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == 0 and conf > 0.4:
                    detected = True
                    highest_conf = max(highest_conf, conf)
                    
                    # Simulation: if confidence is high, consider it Helmet (Green), else No Helmet (Red)
                    if conf > 0.65:
                        status_text = "Helmet"
                        color = (0, 255, 0) # Green
                        current_status = "Helmet"
                    else:
                        status_text = "No Helmet"
                        color = (0, 0, 255) # Red
                        current_status = "No Helmet"
                    
                    # Draw Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{status_text} {conf*100:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Render Processed Frame back to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Emit processed frame back to frontend
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

def start_ai_service():
    try:
        sio.connect(SERVER_URL)
        sio.wait()
    except Exception as e:
        print("Failed to connect to server:", e)

if __name__ == "__main__":
    start_ai_service()
