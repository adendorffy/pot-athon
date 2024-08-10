import os
import cv2
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

# Define paths
images_dir = 'dataset/images/train'
labels_dir = 'dataset/labels/train'

# Get all image files
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_index = 0

# Load class names if you have a file like 'classes.txt'
class_names = []
if os.path.exists('classes.txt'):
    with open('classes.txt', 'r') as f:
        class_names = f.read().strip().split()

def display_image(index):
    global image_label, image_files

    # Read the image
    image_path = os.path.join(images_dir, image_files[index])
    image = cv2.imread(image_path)

    # Read the corresponding label file
    label_path = os.path.join(labels_dir, image_files[index].replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert to pixel values
                img_height, img_width, _ = image.shape
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                # Calculate top-left and bottom-right corners
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Draw the bounding box
                color = (0, 255, 0)  # Green box for bounding box
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Add class label
                if class_names:
                    class_label = class_names[class_id]
                    image = cv2.putText(image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Display the image in the label
    image_label.config(image=image_tk)
    image_label.image = image_tk
    image_label.pack()

def next_image():
    global image_index
    if image_index < len(image_files) - 1:
        image_index += 1
        display_image(image_index)

def prev_image():
    global image_index
    if image_index > 0:
        image_index -= 1
        display_image(image_index)

# Set up the GUI
root = Tk()
root.title("Image Annotation Viewer")

# Image display label
image_label = Label(root)
image_label.pack()

# Navigation buttons
prev_button = Button(root, text="Previous", command=prev_image)
prev_button.pack(side='left')

next_button = Button(root, text="Next", command=next_image)
next_button.pack(side='right')

# Initial display
display_image(image_index)

# Start the GUI loop
root.mainloop()
