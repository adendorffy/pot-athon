import cv2
import numpy as np
import pandas as pd
import os, csv
import glob
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_V2_Weights

def get_stick_length_pixels(image, annotations):
    stick_length_pixels=1
    for annotation in annotations:
        if (annotation[0]=='1'):
            stick_annotation = annotation.strip().split()
            x_center = float(stick_annotation[1])
            y_center = float(stick_annotation[2])
            width = float(stick_annotation[3])
            height = float(stick_annotation[4])
            image_height, image_width, _ = image.shape
            
            x_min = int((x_center - width / 2) * image_width)
            x_max = int((x_center + width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            y_max = int((y_center + height / 2) * image_height)
            
            roi = image[y_min:y_max, x_min:x_max]
            
            red_regions = detect_red_regions(roi)
            
            if len(red_regions) >= 2:
                x1, y1 = red_regions[0]
                x2, y2 = red_regions[1]
                stick_length_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return stick_length_pixels  


def detect_red_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 + mask2
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    red_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        red_regions.append((x + w // 2, y + h // 2))
    
    return red_regions

def get_stick(image, annotations):
    red_regions = detect_red_regions(image)
    if len(red_regions) > 2:    
        red_regions = sorted(red_regions, key=lambda x: x[0])
        x1, y1 = red_regions[0]
        x2, y2 = red_regions[1]
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return pixel_distance*4
    else: 
        stick_length_pixels = get_stick_length_pixels(image, annotations)
        return stick_length_pixels

def get_pothole_area_pixels(mask_rcnn_model, image, annotations):
    roi_area = 1
    
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
            roi_area=(height*image_height)*(width*image_width)

            input = torch.tensor(roi_rgb).permute(2,0,1).unsqueeze(0)/255.0
            with torch.no_grad():
                prediction = mask_rcnn_model(input)

            masks = prediction[0]['masks']
            scores = prediction[0]['scores']
            
            if len(scores) > 0:
                binary_mask = (masks[0] > 0.5).squeeze().cpu().numpy()
                kernel = np.ones((5, 5), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                pothole_area = binary_mask.sum()
                return pothole_area
            else:
                return roi_area           
            
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
    stick_area = 50*4 # cm^2
    image = cv2.imread(image_path)
    annotations=None
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    mask_rcnn_model.eval()
    
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f: annotations = f.readlines()
        pothole_area_pixels = get_pothole_area_pixels(mask_rcnn_model, image, annotations) # pothole area in pixels (area)
        stick_length_pixels = get_stick(image, annotations) # length in pixels (area)
        if pothole_area_pixels is None: pothole_area_pixels = 1
        pothole_area_cm2, ratio = convert_pothole_area(pothole_area_pixels, stick_area, stick_length_pixels)
        real_width, real_height = convert_pothole_height_width(stick_area, stick_length_pixels, annotations, image)
        
        return pothole_area_cm2, real_width, real_height, ratio

def process_image(image_path, annotation_dir, labels_df):
    image_number = ((image_path.rsplit("/")[-1]).replace(".jpg", "").replace("p",""))
    annotation_path = os.path.join(annotation_dir, f"p{image_number}.txt")
    result = get_real_dimensions(image_path, annotation_path)
    if result is not None:
        pothole_area, real_width, real_height, ratio = result
        if pothole_area/10000 < 20:
            annotation_path = annotation_dir + "/p" + str(image_number) + ".txt"
            label_row = labels_df[labels_df['Pothole number'] == image_number]
            bags = label_row['Bags used'].values[0] if not label_row.empty else None
            if bags is not None:            
                return {
                    "ID": image_number,
                    "Area": pothole_area/10000,
                    "Width": real_width/100,
                    "Height": real_height/100,
                    "Bags": bags
                }