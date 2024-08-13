## How to run and train

1. Install requirements.txt
2. run yolo task=detect data=data.yaml mode=train model=yolov8n.pt epochs=10 imgsz=640
   NOTE: update the file path in data.yaml
   this fine tunes the model
   In order to train yolov5 out dataset has to be in the format:
   dataset/
   ├── images/
   │ ├── train/
   │ ├── val/
   ├── labels/
   │ ├── train/
   │ ├── val/

and the labels/annotations must be in format: <class_id> <x_center> <y_center> <width> <height>
1239, 1188, 1450 does not have annotation.

## Generate pothole mask

Install torch, torchvision and openvcv-python (also in requirements.txt)
Run: python3 gen_mask.py
Outputs the estimated area
