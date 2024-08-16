import cv2
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def get_stick_length_pixels(image, annotations):
    """
    Calculates the length of a reference stick in pixels.
    """
    stick_length_pixels = 1
    for annotation in annotations:
        components = annotation.strip().split()
        if components[0] == '1':  # Check if the annotation is for the stick
            stick_annotation = annotation.strip().split()
            width = float(stick_annotation[3])
            height = float(stick_annotation[4])
            image_height, image_width, _ = image.shape
            # Compute stick length in pixels using its dimensions and image size
            stick_length_pixels = np.sqrt((width * image_width) ** 2 + (height * image_height) ** 2)
    return stick_length_pixels

def get_pothole_area_pixels(image, annotations):
    """
    Calculates the area of potholes in pixels using a Mask R-CNN model.
    """
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    mask_rcnn_model.eval()  # Set model to evaluation mode
    roi_area = 1  # Default value if no pothole is detected
    
    for annotation in annotations:
        if annotation.startswith("0"):  # Check if the annotation is for a pothole
            pothole_annotation = annotation.strip().split()
            x_center = float(pothole_annotation[1])
            y_center = float(pothole_annotation[2])
            width = float(pothole_annotation[3])
            height = float(pothole_annotation[4])

            image_height, image_width, _ = image.shape
            # Compute the bounding box for the pothole
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            roi = image[y_min:y_max, x_min:x_max]  # Extract the region of interest
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert ROI to RGB format

            # Compute initial area based on bounding box
            roi_area = (height * image_height) * (width * image_width)

            # Prepare the input for the Mask R-CNN model
            input = torch.tensor(roi_rgb).permute(2, 0, 1).unsqueeze(0) / 255
            with torch.no_grad():
                prediction = mask_rcnn_model(input)  # Get model predictions

            masks = prediction[0]['masks']
            scores = prediction[0]['scores']
            threshold = 0.2
            
            # Filter masks based on the confidence score
            filtered_masks = [masks[i] for i in range(len(scores)) if scores[i] > threshold]
            binary_masks = [mask > 0.5 for mask in filtered_masks]
            if len(binary_masks) > 0:
                # Combine all masks into one
                final_mask = torch.zeros_like(binary_masks[0])
                for mask in binary_masks:
                    final_mask = torch.max(final_mask, mask)
                final_mask_np = final_mask.squeeze().cpu().numpy()
                # Compute the area of the pothole from the final mask
                pothole_area_pixels = np.sum(final_mask_np)
                return pothole_area_pixels
            else:
                return roi_area  # Return the area based on the bounding box if no mask is detected
                
def convert_pothole_area(pothole_area_pixels, stick_area_cm, stick_area_pixels):
    """
    Converts the pothole area from pixels to square centimeters.
    """
    pixel_to_cm_ratio = stick_area_cm / stick_area_pixels  # Calculate the ratio of cm^2 per pixel
    pothole_area_cm2 = pothole_area_pixels * pixel_to_cm_ratio  # Convert area to cm^2
    return pothole_area_cm2, pixel_to_cm_ratio

def convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image):
    """
    Converts the height and width of potholes from pixels to centimeters.
    """
    cm_per_pixel = stick_area / stick_length_pixels  # Calculate cm per pixel using the stick
    height_pixels = 0
    width_pixels = 0
    for annotation in annotations:
        components = annotation.strip().split()
        if components[0] == '0':  # Check if the annotation is for a pothole
            pothole_annotation = annotation.strip().split()
            width = float(pothole_annotation[3])
            height = float(pothole_annotation[4])
            image_height, image_width, _ = image.shape
            # Convert width and height from pixels to centimeters
            height_pixels = height * image_height
            width_pixels = width * image_width
    real_height = height_pixels * cm_per_pixel
    real_width = width_pixels * cm_per_pixel
    return real_width, real_height 

def get_real_dimensions(image_path, annotation_path):
    """
    Gets the real-world dimensions and area of potholes from image and annotation files.
    """
    stick_area = 50 * 4  # Area of the reference stick in cm^2
    image = cv2.imread(image_path)  # Read the image
    annotations = None
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()  # Read the annotations
    stick_length_pixels = get_stick_length_pixels(image, annotations)  # Get stick length in pixels
    pothole_area_pixels = get_pothole_area_pixels(image, annotations)  # Get pothole area in pixels
    if pothole_area_pixels is None: 
        pothole_area_pixels = 1  # Default value if no pothole detected
    pothole_area_cm2, ratio = convert_pothole_area(pothole_area_pixels, stick_area, stick_length_pixels)  # Convert area to cm^2
    real_width, real_height = convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image)  # Convert dimensions to cm

    return pothole_area_cm2, real_width, real_height, ratio

def process_image(image_path, annotation_dir, labels_df):
    """
    Processes a single image and its annotations, retrieves data from a DataFrame, and returns a summary.
    """
    image_number = ((image_path.rsplit("/")[-1]).replace(".jpg", "").rsplit("_")[0].replace("p",""))
    annotation_path = os.path.join(annotation_dir, f"p{image_number}.txt")  # Construct the annotation file path
    pothole_area, real_width, real_height, ratio = get_real_dimensions(image_path, annotation_path)  # Get dimensions and area
    label_row = labels_df[labels_df['Pothole number'] == image_number.rsplit("_")[0]]  # Look up data in DataFrame
    bags = label_row['Bags'].values[0] if not label_row.empty else None
    
    return {
        "ID": image_number,
        "Area": pothole_area,
        "Width": real_width,
        "Height": real_height,
        "Bags": bags
    }
