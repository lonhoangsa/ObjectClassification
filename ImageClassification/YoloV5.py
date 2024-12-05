import torch
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Tải mô hình YOLO từ PyTorch Hub (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Tải YOLOv5s (model nhẹ)

def load_image(image_path):
    # Đọc ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
    return img

def detect_objects(model, image):
    # Dự đoán đối tượng trong ảnh
    results = model(image)  # Thực hiện nhận diện vật thể
    return results

def plot_results(results):
    # Hiển thị ảnh với bounding box
    results.show()  # Tự động hiển thị bounding box trong ảnh


def run_yolo_application(image_path):
    # 1. Tải ảnh
    image = load_image(image_path)

    # 2. Nhận diện vật thể
    results = detect_objects(model, image)

    # 3. Hiển thị kết quả
    plot_results(results)

    # Trả về các bounding box (Nếu cần)
    return results.pandas().xywh  # Trả về kết quả dưới dạng DataFrame Pandas


# Sử dụng ứng dụng với ảnh
image_path = 'D:\k0d3\Project1\ObjectClassification\Dataset\kittens-cat-cat-puppy-rush-45170.jpeg'
run_yolo_application(image_path)
