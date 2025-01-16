# Object Classification

Object Classification là một dự án sử dụng các mô hình học sâu để phân loại và phân đoạn đối tượng từ hình ảnh.

## Mục Lục

- [Giới Thiệu](#giới-thiệu)
- [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Hướng Dẫn Cài Đặt](#hướng-dẫn-cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
- [Đóng Góp](#đóng-góp)
- [Liên Hệ](#liên-hệ)

## Giới Thiệu

Dự án này áp dụng các kỹ thuật học sâu để phân loại các đối tượng trong hình ảnh. Các mô hình đã được huấn luyện hoặc bạn có thể tự huấn luyện lại với dữ liệu tùy chỉnh của mình.

**Tính năng chính:**
- Phân loại đối tượng từ hình ảnh với độ chính xác cao.
- Hỗ trợ nhiều phiên bản của Yolo
- Phân loại và cắt lớp đối tượng riêng biệt mong muốn bằng công nghệ Prompt  

## Công Nghệ Sử Dụng

- Yolo(version v5 -> 11)
- Grounding Dino
- Grounding Sam2 (Kết hợp Grounding Dino và Sam2)

## Yêu Cầu Hệ Thống

- Hệ điều hành: Windows, macOS, hoặc Linux
- Bộ nhớ RAM: Tối thiểu 8GB
- GPU: Khuyến nghị sử dụng GPU NVIDIA (CUDA support)
- Python: 3.8 hoặc mới hơn

## Hướng Dẫn Cài Đặt

### 1. Clone Repository
```bash
$ git clone https://github.com/lonhoangsa/ObjectClassification.git
$ cd ObjectClassification
```

### 2. Cài Đặt Môi Trường Ảo (Tùy Chọn)
```bash
$ python -m venv venv
$ source venv/bin/activate # Trên Linux/MacOS
$ venv\Scripts\activate # Trên Windows
```

### 3. Cài Đặt Các Thư Viện Yêu Cầu
```bash
$ pip install -r requirements.txt
```
### 4. Setup ứng dụng trên thiết bị
```bash
python web/manage.py makemigrations
python web/manage.py migrate
```
### 5. Chạy chương trình
```bash
python web/manage.py migrate
start http://127.0.0.1:8000/models/upload/
python web/manage.py runserver
```

## Liên Hệ

Nếu bạn có bất kỳ câu hỏi hoặc vấn đề nào, vui lòng liên hệ qua email: [laihoangson42@gmail.com](mailto:laihoangson42@gmail.com).

---

Cảm ơn bạn đã quan tâm đến dự án!
