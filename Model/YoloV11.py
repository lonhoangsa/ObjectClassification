import os

from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLO model
model = YOLO("yolo11x.pt")

# Read the input image
img_path = r'D:\k0d3\Project1\ObjectClassification\Model\kittens-cat-cat-puppy-rush-45170.jpeg'
img = cv2.imread(img_path)

# Perform detection with YOLO
results = model(img_path)

# Create a directory to save the cropped images
output_dir = "detected_objects"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store object names
object_dict = {}

# Iterate through detected objects
for i, (bbox, cls) in enumerate(zip(results.xyxy[0], results.names)):
    x1, y1, x2, y2 = map(int, bbox[:4])
    obj_name = results.names[int(cls)]
    
    # Crop the object from the image
    cropped_img = img[y1:y2, x1:x2]
    
    # Save the cropped image
    cropped_img_path = os.path.join(output_dir, f"{obj_name}_{i}.jpg")
    cv2.imwrite(cropped_img_path, cropped_img)
    
    # Add the object name to the dictionary
    if obj_name not in object_dict:
        object_dict[obj_name] = []
    object_dict[obj_name].append(cropped_img_path)

# Print the dictionary of object names and their corresponding image paths
print(object_dict)
