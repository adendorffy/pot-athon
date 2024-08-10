import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained YOLOv8 model
model = YOLO('/home/karen/Desktop/HACKATHON/pot-athon/runs/detect/train15/weights/best.pt')

# Directories
test_images_dir = 'data/test_images'
output_annotation_dir = 'results/output/annotation'
output_images_dir = 'results/output/images'

# Create output directories if they don't exist
os.makedirs(output_annotation_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

# Function to save bounding boxes in YOLO format
def save_yolo_format(result, file_path):
    with open(file_path, 'w') as f:
        for box, cls in zip(result.boxes.xywh, result.boxes.cls):
            # YOLO format requires class_id, x_center, y_center, width, height normalized by image dimensions
            class_id = int(cls.item())  # Object class ID
            x_center, y_center, width, height = box.cpu().numpy() / [result.orig_shape[1], result.orig_shape[0], result.orig_shape[1], result.orig_shape[0]]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Iterate over all images in the test directory
for image_name in os.listdir(test_images_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        # Full path to the image
        image_path = os.path.join(test_images_dir, image_name)

        # Perform object detection
        results = model(image_path)
        result = results[0]

        # Save bounding boxes in YOLO format
        annotation_file_path = os.path.join(output_annotation_dir, f"{os.path.splitext(image_name)[0]}.txt")
        save_yolo_format(result, annotation_file_path)

        # Plot and save the image with bounding boxes
        plotted_image = result.plot()
        output_image_path = os.path.join(output_images_dir, image_name)
        cv2.imwrite(output_image_path, plotted_image)

        print(f"Processed {image_name}")

print("Object detection and saving completed.")
