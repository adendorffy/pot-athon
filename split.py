import os
import random
import shutil

# Define paths
data_path = 'data'
images_path = os.path.join(data_path, 'train_images')
labels_path = os.path.join(data_path, 'train_annotations')

# Define output directories for train and val
output_dirs = {
    'train': {'images': 'dataset/images/train', 'labels': 'dataset/labels/train'},
    'val': {'images': 'dataset/images/val', 'labels': 'dataset/labels/val'}
}

# Create directories if they don't exist
for dirs in output_dirs.values():
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

# Get all image file names
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Include all image formats
random.shuffle(image_files)

# Calculate split indices
num_images = len(image_files)
train_idx = int(num_images * 0.80)  # 80% for training
# 20% for validation

# Split into train and val
splits = {
    'train': image_files[:train_idx],
    'val': image_files[train_idx:]
}

# Copy files to respective directories
for split, files in splits.items():
    for file in files:
        # Copy images
        src_image = os.path.join(images_path, file)
        dst_image = os.path.join(output_dirs[split]['images'], file)
        shutil.copyfile(src_image, dst_image)

        # Copy labels
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')  # Adjust label extension
        src_label = os.path.join(labels_path, label_file)
        if os.path.exists(src_label):
            dst_label = os.path.join(output_dirs[split]['labels'], label_file)
            shutil.copyfile(src_label, dst_label)
