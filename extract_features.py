import cv2
import numpy as np
import pandas as pd
import os, csv
import glob
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

def get_stick_length_pixels(image, annotations):
    stick_length_pixels=0
    for annotation in annotations:
        components = annotation.strip().split()
        if (components[0]=='1'):
            stick_annotation = annotation.strip().split()
            width = float(stick_annotation[3])
            height = float(stick_annotation[4])
            image_height, image_width, _ = image.shape
            stick_length_pixels = np.sqrt((width*image_width)**2 + (height*image_height)**2)
    return stick_length_pixels

def get_pothole_area_pixels(image, annotations):
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    mask_rcnn_model.eval()
    
    for annotation in annotations:
        if annotation.startswith("0"):
            pothole_annotation = annotation.strip().split()
            x_center = float(pothole_annotation[1])
            y_center = float(pothole_annotation[2])
            width = float(pothole_annotation[3])
            height = float(pothole_annotation[4])

            image_height, image_width, _ = image.shape
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            roi = image[y_min:y_max, x_min:x_max]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            input = torch.tensor(roi_rgb).permute(2,0,1).unsqueeze(0)/255
            with torch.no_grad():
                prediction = mask_rcnn_model(input)

            masks = prediction[0]['masks']
            scores = prediction[0]['scores']
            threshold = 0.2
            
            filtered_masks = [masks[i] for i in range(len(scores)) if scores[i] > threshold]
            binary_masks = [mask > 0.5 for mask in filtered_masks]
            if len(binary_masks)>0:
                final_mask = torch.zeros_like(binary_masks[0])
                for mask in binary_masks: final_mask = torch.max(final_mask, mask)
                final_mask_np = final_mask.squeeze().cpu().numpy()
                pothole_area_pixels = np.sum(final_mask_np)
                return pothole_area_pixels
            else: 
                return (height*image_height)*(width*image_width)
                
            
def convert_pothole_area(pothole_area_pixels, stick_area_cm, stick_area_pixels):
    pixel_to_cm_ratio = stick_area_cm / stick_area_pixels # cm^2/pix
    pothole_area_cm2 = pothole_area_pixels * pixel_to_cm_ratio # pixels*(cm^2/pixels) = cm^2
    return pothole_area_cm2, pixel_to_cm_ratio

def convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image):    
    cm_per_pixel = stick_area/stick_length_pixels # CHANGED: 50->200 (50cm*4cm=200cm area stick)
    height_pixels=0
    width_pixels=0
    for annotation in annotations:
        components = annotation.strip().split()
        if (components[0]=='0'): 
            stick_annotation = annotation.strip().split()
            width= float(stick_annotation[3])
            height = float(stick_annotation[4])
            image_height, image_width, _ = image.shape
            height_pixels = height*image_height
            width_pixels=width*image_width
    real_height=height_pixels*cm_per_pixel
    real_width=width_pixels*cm_per_pixel
    return real_width, real_height 

def get_real_dimensions(image_path, annotation_path):
    stick_area = 50*4
    image = cv2.imread(image_path)
    annotations=None
    with open(annotation_path, 'r') as f: annotations = f.readlines()
    stick_length_pixels = get_stick_length_pixels(image, annotations) # length in pixels (area)
    real_width, real_height = convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image)
    pothole_area_pixels = get_pothole_area_pixels(image, annotations) # pothole area in pixels (area)
    pothole_area_cm2, ratio = convert_pothole_area(pothole_area_pixels, stick_area, stick_length_pixels)
    real_width, real_height = convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image)
    
    return pothole_area_cm2, real_width, real_height, ratio

def process_image(image_path, annotation_dir, labels_df):
    image_number = int((image_path.rsplit("/")[-1]).replace(".jpg", "").replace("p", ""))
    annotation_path = os.path.join(annotation_dir, f"p{image_number}.txt")
    pothole_area, real_width, real_height = get_real_dimensions(image_path, annotation_path)
    annotation_path = annotation_dir + "/p" + str(image_number) + ".txt"
    pothole_area, real_width, real_height, ratio = get_real_dimensions(image_path, annotation_path)
    label_row = labels_df[labels_df['Pothole number'] == image_number]
    bags = label_row['Bags'].values[0] if not label_row.empty else None
    
    return {
        "ID": image_number,
        "Area": pothole_area,
        "BoundingBoxArea": real_height*real_width,
        "Width": real_width,
        "Height": real_height,
        "Ratio":ratio,
        "Bags": bags
    }