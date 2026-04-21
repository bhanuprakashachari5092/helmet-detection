import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split

def convert_voc_to_yolo():
    source_dir = r"C:\Users\Banu Prakash\OneDrive\Desktop\archive (1)"
    dest_dir = "helmet_dataset"
    classes = ["With Helmet", "Without Helmet"]
    class_map = {name: i for i, name in enumerate(classes)}

    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, 'labels'), exist_ok=True)

    xml_files = [f for f in os.listdir(os.path.join(source_dir, 'annotations')) if f.endswith('.xml')]
    train_files, val_files = train_test_split(xml_files, test_size=0.2, random_state=42)

    def process_files(files, split):
        for xml_file in files:
            tree = ET.parse(os.path.join(source_dir, 'annotations', xml_file))
            root = tree.getroot()
            
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)
            filename = root.find('filename').text
            
            # Copy image
            img_src = os.path.join(source_dir, 'images', filename)
            img_dest = os.path.join(dest_dir, split, 'images', filename)
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dest)
            
            # Create label file
            label_name = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(dest_dir, split, 'labels', label_name), 'w') as f:
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name not in class_map: continue
                    
                    class_id = class_map[name]
                    xmlbox = obj.find('bndbox')
                    
                    xmin = float(xmlbox.find('xmin').text)
                    ymin = float(xmlbox.find('ymin').text)
                    xmax = float(xmlbox.find('xmax').text)
                    ymax = float(xmlbox.find('ymax').text)
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    w = (xmax - xmin) / width
                    h = (ymax - ymin) / height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("Converting Train set...")
    process_files(train_files, 'train')
    print("Converting Val set...")
    process_files(val_files, 'val')

    # Create data.yaml
    with open(os.path.join(dest_dir, 'data.yaml'), 'w') as f:
        f.write(f"path: {os.path.abspath(dest_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")

    print(f"Dataset conversion complete! Created {dest_dir}")

if __name__ == "__main__":
    convert_voc_to_yolo()
