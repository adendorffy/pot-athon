
import cv2
import numpy as np
import pandas as pd
import os

ratios = pd.read_csv('test_ratio.csv')

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

def get_real_dimensions(image_path, annotation_path,id):
    image = cv2.imread(image_path)
    annotations=None
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    stick_length_pixels = get_stick_length_pixels(image, annotations) # length in pixels (area)
    cm_per_pixel = 50/stick_length_pixels
    try:
        print(id)
        cm_per_pixel = ratios.loc[ratios['ID'] == id, 'cm_per_pixel'].values[0]
    except IndexError:
        cm_per_pixel = 50/stick_length_pixels 



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
    print(f"pothole width {real_width} and pothole height {real_height}")      
    return real_width, real_height 


image_dir = "data/test_images"
annotation_dir = "data/test-annotations"
labels_csv = "data/train_labels.csv"


labels_df = pd.read_csv(labels_csv)
data = []

for image_file in os.listdir(image_dir):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(image_dir, image_file)
        annotation_file = image_file.replace(".jpg", ".txt")
        annotation_path = os.path.join(annotation_dir, annotation_file)
        image_number = int(image_file.split(".")[0].replace("p", ""))
        real_width, real_height = get_real_dimensions(image_path, annotation_path, image_number)
        image_id = image_file

        try:
            cm_per_pixel = ratios.loc[ratios['ID'] == image_number, 'cm_per_pixel'].values[0]
            label_row = labels_df[labels_df['Pothole number'] == image_number]
            if not label_row.empty:
                bags = label_row['Bags'].values[0]
            else:
                bags = None       
            data.append({
                "ID": image_number,
                "Width": real_width,
                "Height": real_height,
                "Bags": bags
            })
        except IndexError:
           print('skip')
    
       
df = pd.DataFrame(data)
output_csv = "test_features.csv"
df.to_csv(output_csv, index=False)
