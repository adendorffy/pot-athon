import os
from ultralytics import YOLO
model = YOLO('/home/karen/Desktop/HACKATHON/clean clone/pot-athon/fine-tune-training/runs/detect/train20/weights/best.pt')
test_images_dir = '/home/karen/Desktop/HACKATHON/clean clone/pot-athon/imgs'
output_annotation_dir = 'results'


def save_yolo_format(result, file_path):
    with open(file_path, 'w') as f:
        sorted_indices = result.boxes.conf.argsort(descending=True)
        seen_classes = set()
        for idx in sorted_indices:
            box = result.boxes.xywh[idx]
            cls = int(result.boxes.cls[idx].item())
            conf = result.boxes.conf[idx].item()
            #only take class with most confidence
            if cls in seen_classes:
                continue
            else:
                seen_classes.add(cls)
                x_center, y_center, width, height = box.cpu().numpy() / [result.orig_shape[1], result.orig_shape[0], result.orig_shape[1], result.orig_shape[0]]
                f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

# make predictions for each image and save in yolo format
for image_name in os.listdir(test_images_dir):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_images_dir, image_name)
        results = model(image_path, conf=0.00001)
        result = results[0]
        annotation_file_path = os.path.join(output_annotation_dir, f"{os.path.splitext(image_name)[0]}.txt")
        save_yolo_format(result, annotation_file_path)