from ultralytics import YOLO

def train_helmet_model():
    # Load a pretrained model
    model = YOLO('yolov8n.pt') 

    # Train the model
    # data: path to your dataset.yaml file
    # epochs: number of training iterations
    # imgsz: image size
    # Reference: https://docs.ultralytics.com/modes/train/
    results = model.train(
        data='helmet_dataset/data.yaml', 
        epochs=50, 
        imgsz=640,
        name='helmet_model'
    )
    
    print("Training complete! Model saved in runs/detect/helmet_model/weights/best.pt")

if __name__ == "__main__":
    # Ensure you ran convert_voc_to_yolo.py first!
    print("🚀 Starting Training Process on Custom Dataset...")
    train_helmet_model()
