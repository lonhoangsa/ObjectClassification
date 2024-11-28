import torch
from torchvision import models, transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import requests
import json

# Tải mô hình ResNet50 với trọng số đã huấn luyện sẵn từ ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()  # Chuyển mô hình về chế độ đánh giá (evaluation mode)

# Định nghĩa các bước biến đổi hình ảnh (resize, normalize, v.v.)
transform = transforms.Compose([
    transforms.Resize(256),  # Resize hình ảnh về kích thước 256x256
    transforms.CenterCrop(224),  # Cắt hình ảnh thành vùng trung tâm có kích thước 224x224
    transforms.ToTensor(),  # Chuyển đổi hình ảnh thành tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa theo giá trị của ImageNet
])

# Hàm chuẩn bị hình ảnh để đưa vào mô hình
def prepare_image(image_path):
    image = Image.open(image_path)  # Mở tệp hình ảnh từ đường dẫn
    image = transform(image).unsqueeze(0)  # Thực hiện biến đổi hình ảnh và thêm một chiều batch (unsqueeze)
    return image

# Hàm giải mã kết quả dự đoán (sử dụng nhãn của ImageNet)
def decode_predictions(preds):
    # Tải nhãn của ImageNet
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(labels_url)  # Tải dữ liệu nhãn từ URL
    class_idx = json.loads(response.text)  # Chuyển đổi dữ liệu JSON thành dictionary
    labels = [class_idx[str(i)][1] for i in range(1000)]  # Lấy danh sách nhãn (tên lớp)

    # Lấy 3 kết quả dự đoán có độ tin cậy cao nhất
    _, indices = torch.topk(preds, 3)  # Lấy 3 chỉ số có giá trị cao nhất
    predictions = [{"label": labels[idx], "confidence": float(preds[0, idx])} for idx in indices[0]]  # Tạo danh sách kết quả dự đoán
    return predictions

# Hàm nhận diện hình ảnh từ đường dẫn
def predict_image(image_path):
    image = prepare_image(image_path)  # Chuẩn bị hình ảnh
    with torch.no_grad():  # Tắt tính toán gradient (không cần thiết trong chế độ đánh giá)
        preds = model(image)  # Dự đoán với mô hình

    predictions = decode_predictions(preds)  # Giải mã kết quả dự đoán
    return predictions

# Hàm chọn tệp hình ảnh thông qua pop-up
def select_image_file():
    print("selecting")
    root = tk.Tk()  # Tạo cửa sổ Tkinter
    print("tạo cửa sổ-ed")
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(title="Chọn tệp hình ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Sử dụng cửa sổ pop-up để chọn tệp
image_path = select_image_file()

if image_path:
    # Kiểm tra và nhận diện
    try:
        predictions = predict_image(image_path)
        print("Kết quả phân loại:")
        for prediction in predictions:
            print(f"Nhãn: {prediction['label']}, Độ tin cậy: {prediction['confidence']:.2f}%")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
else:
    print("Không có tệp hình ảnh nào được chọn.")
