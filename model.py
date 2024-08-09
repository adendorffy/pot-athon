from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch

# Load image
local_image_path = 'data/train_images/p101.jpg'
image = Image.open(local_image_path)

# Initialize processor and model
processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Perform object detection
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the output
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.5:  # Confidence threshold
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{model.config.id2label[label.item()]}: {score:.2f}", fill="red")

# Display image with bounding boxes
image.show()
