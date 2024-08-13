# import tensorflow as tf
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn_model.eval()

def get_pothole_area_pixels(image, annotations):
    pothole_annotation = annotations[0].strip().split()
    object_id = pothole_annotation[0]
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
    threshold = 0.3

    filtered_masks = [masks[i] for i in range(len(scores)) if scores[i] > threshold]

    if filtered_masks:
        pothole_mask = filtered_masks[0][0].mul(255).byte().cpu().numpy()
        pothole_area_pixels = np.sum(pothole_mask > 0)
        return pothole_area_pixels
    
# Returns the area of the stick in pixels
def get_stick_length_pixels(image, annotations):
    stick_lengths = []

    # For each annotation get the area of the stick in pixels
    for annotation in annotations[1:]:
        stick_annotation = annotation.strip().split()
        width = float(stick_annotation[3])
        height = float(stick_annotation[4])

        image_height, image_width, _ = image.shape
        stick_length_pixels = np.sqrt((width*image_width)**2 + (height*image_height)**2)
        stick_lengths.append(stick_length_pixels)

    # Assume: largest area is the full length of stick
    stick_lengths = np.max(np.array(stick_lengths))
    return stick_lengths

def convert_area(pothole_area_pixels, stick_area_cm, stick_area_pixels):
    pixel_to_cm_ratio = stick_area_cm / stick_area_pixels # cm^2/pix
    pothole_area_cm2 = pothole_area_pixels * pixel_to_cm_ratio # pixels*(cm^2/pixels) = cm^2
    return pothole_area_cm2


image_path = "output/images/p104.jpg"
annotation_path = "output/annotation/p104.txt"
image = cv2.imread(image_path)

with open(annotation_path, 'r') as f:
    annotations = f.readlines()

stick_length_cm = 50*4 # area in cm2
stick_length_pixels = get_stick_length_pixels(image, annotations) # length in pixels (area)
pothole_area_pixels = get_pothole_area_pixels(image, annotations) # pothole area in pixels (area)

pothole_area_cm2 = convert_area(pothole_area_pixels, stick_length_cm, stick_length_pixels)
print(f"Pothole area in square centimeters: {pothole_area_cm2:.2f} cm² or {pothole_area_cm2/1000:.2f} m²")