#3333333333333333333333333333333333333333333333333333333333333333333333333
# a quick script written by ChatGPT that plots the annotations on the imagaes

import os
import cv2
import numpy as np

# Function to plot annotations
def plot_annotations(image_dir, annotation_dir, output_dir, classes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for annotation_file in os.listdir(annotation_dir):
        image_file = annotation_file.replace('.txt', '.jpg')  # Assuming images are in .jpg format
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, annotation_file)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}")
            continue
        
        # Read annotations
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            class_id = int(class_id)
            
            # Convert YOLO format to bounding box coordinates
            img_h, img_w, _ = image.shape
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            # Draw bounding box
            color = (255, 0, 0)  # Red for all classes, change if needed
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, classes[class_id], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save the annotated image
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

# Define paths
train_image_dir = 'train_images'
train_annotation_dir = 'train-annotations'
train_output_dir = 'annotated-images/train'

test_image_dir = 'test_images'
test_annotation_dir = 'test-annotations'
test_output_dir = 'annotated-images/test'

# Define class names
classes = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2'}

# Plot and save annotations for training images
plot_annotations(train_image_dir, train_annotation_dir, train_output_dir, classes)

# Plot and save annotations for test images
plot_annotations(test_image_dir, test_annotation_dir, test_output_dir, classes)

print("Annotation plotting complete.")
