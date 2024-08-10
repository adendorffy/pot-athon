## How to run and train

1. Install requirements.txt
2. run yolo task=detect data=data.yaml mode=train model=yolov8n.pt epochs=10 imgsz=640
   NOTE: update the file path in data.yaml

1) The yolov5 model was cloned with :git clone https://github.com/ultralytics/yolov5.git

2) In order to train yolov5 out dataset has to be in the format:
   dataset/
   ├── images/
   │ ├── train/
   │ ├── val/
   ├── labels/
   │ ├── train/
   │ ├── val/

and the labels/annotations must be in format: <class_id> <x_center> <y_center> <width> <height>

After this is done the yolov5 model can be trained using: python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --cache

1239, 1188, 1450 does not have annotation.
