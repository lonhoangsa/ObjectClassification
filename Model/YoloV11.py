from ultralytics import YOLO
import cv2

# Tải mô hình YOLOv8 pre-trained
model = YOLO("yolo11x.pt")  # Sử dụng mô hình nhỏ nhất của YOLOv8

# Đọc hình ảnh đầu vào
img_path = "/ObjectClassification/Dataset/kittens-cat-cat-puppy-rush-45170.jpeg"  # Đường dẫn đến hình ảnh đầu vào
img = cv2.imread(img_path)

# Thực hiện nhận diện với YOLOv8
results = model(img_path)  # Chạy mô hình YOLOv8 trên hình ảnh
for image in results:
    # Hiển thị kết quả nhận diện
    image.show()  # Hiển thị hình ảnh với các bounding boxes

# Lưu kết quả (hình ảnh với bounding boxes)
# output_path = "output_image.jpg"
# results.save(path=output_path)  # Lưu kết quả vào file
#
# print(f"Kết quả đã được lưu tại {output_path}")
