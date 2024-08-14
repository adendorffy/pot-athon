import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])) #remove the final classification layer, only use for feat extraction
resnet.eval() #ensures consistent feat extraction behaviour
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#read features extracted by extract_features.py
test_area_features = pd.read_csv("test_area_features.csv")
train_area_features = pd.read_csv("train_area_features.csv")
test_area_features['type'] = 'test'
train_area_features['type'] = 'train'
combined_features = pd.concat([train_area_features, test_area_features], axis=0).reset_index(drop=True)
train_area_features=combined_features

image_dir = "data/cropped_test_and_train"

features_list = []
#for each image:
for index, row in combined_features.iterrows():
    image_id = int(row['ID'])
    image_path = os.path.join(image_dir, f"p{image_id}.jpg")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) 
    with torch.no_grad(): #disable grad calc sice only doing inference
        features = resnet(image).flatten().numpy() #extarct features
    features_list.append(features) #add features for that image to list
    print(index)
features_df = pd.DataFrame(features_list, columns=[f'feature_{i}' for i in range(len(features_list[0]))])


train_area_features_resnet = pd.concat([combined_features, features_df], axis=1)
# add bounding box feature
train_area_features_resnet['BoundingBoxArea']=train_area_features_resnet['Width']*train_area_features_resnet['Height']
train_area_features_resnet.to_csv("features_resNet.csv", index=False)
