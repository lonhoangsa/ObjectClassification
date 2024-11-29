import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import filedialog
import requests
import json

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Tải mô hình Faster R-CNN với trọng số đã huấn luyện sẵn từ COCO
model = models.detection.fasterrcnn_resnet50_fpn(weights= FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()  # Chuyển mô hình về chế độ đánh giá (evaluation mode)

# Định nghĩa các bước biến đổi hình ảnh (resize, normalize, v.v.)
transform = transforms.Compose([
    transforms.ToTensor(),  # Chuyển đổi hình ảnh thành tensor
])

# Hàm chuẩn bị hình ảnh để đưa vào mô hình
def prepare_image(image_path):
    image = Image.open(image_path)  # Mở tệp hình ảnh từ đường dẫn
    image_tensor = transform(image)  # Thực hiện biến đổi hình ảnh thành tensor
    return image, image_tensor.unsqueeze(0)  # Trả về cả hình ảnh gốc và tensor (thêm chiều batch)

# Hàm giải mã kết quả dự đoán và vẽ bounding box
def detect_and_draw_objects(image, image_tensor):
    with torch.no_grad():  # Tắt tính toán gradient (không cần thiết trong chế độ đánh giá)
        prediction = model(image_tensor)  # Dự đoán với mô hình

    # Lấy các hộp bao (bounding boxes), nhãn (labels) và độ tin cậy (scores)
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Lọc ra các đối tượng có độ tin cậy cao hơn một ngưỡng (0.5)
    high_confidence_indices = [i for i, score in enumerate(scores) if score > 0.5]
    boxes = boxes[high_confidence_indices]
    labels = labels[high_confidence_indices]
    scores = scores[high_confidence_indices]

    # Lấy nhãn và độ tin cậy cao nhất
    if len(boxes) == 0:
        return image  # Nếu không có đối tượng, trả về ảnh gốc

    # Tải nhãn ImageNet từ URL
    imagenet_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        response = requests.get(imagenet_labels_url)
        imagenet_labels = response.json()
    except Exception as e:
        print(f"Lỗi khi tải nhãn từ URL: {e}")
        imagenet_labels = {}

    draw = ImageDraw.Draw(image)  # Tạo đối tượng vẽ lên ảnh

    # Vẽ bounding box và ghi tên vật thể
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()  # Chuyển tọa độ từ tensor sang numpy array

        response = requests.get(imagenet_labels_url)  # Tải dữ liệu nhãn từ URL
        class_idx = json.loads(response.text)  # Chuyển đổi dữ liệu JSON thành dictionary
        label = [class_idx[str(i)][1] for i in range(len(class_idx))]  # Lấy danh sách nhãn (tên lớp)
        print(label)
        # label_id = labels[i].item()  # Lấy ID nhãn
        # label = imagenet_labels.get(str(label_id), ["Unknown"])[1]  # Lấy tên nhãn từ ImageNet
        score = scores[i].item()  # Lấy độ tin cậy

        # Vẽ bounding box (dạng [x_min, y_min, x_max, y_max])
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill="red")

    return image

# Hàm giải mã kết quả dự đoán (sử dụng nhãn của ImageNet)
def decode_predictions(preds):
    # Tải nhãn của ImageNet
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(labels_url)  # Tải dữ liệu nhãn từ URL
    class_idx = json.loads(response.text)  # Chuyển đổi dữ liệu JSON thành dictionary
    labels = [class_idx[str(i)][1] for i in range(len(class_idx))]  # Lấy danh sách nhãn (tên lớp)

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
    print("opened window")
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter

    file_path = filedialog.askopenfilename(title="Chọn tệp hình ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Sử dụng cửa sổ pop-up để chọn tệp
image_path = select_image_file()

if image_path:
    # Kiểm tra và nhận diện
    try:
        # predictions = predict_image(image_path)
        # print("Kết quả phân loại:")
        # for prediction in predictions:
        #     print(f"Nhãn: {prediction['label']}, Độ tin cậy: {prediction['confidence']:.2f}%")

        image, image_tensor = prepare_image(image_path)
        image_with_boxes = detect_and_draw_objects(image, image_tensor)

        # Hiển thị ảnh với các hộp bao và tên vật thể
        image_with_boxes.show()

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
else:
    print("Không có tệp hình ảnh nào được chọn.")
