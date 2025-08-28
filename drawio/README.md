# FlowMind2Digital - Draw.io Format Support

Phiên bản được cải tiến để hỗ trợ format .drawio và giải quyết xung đột thư viện paddlepaddle/paddleocr.

## Tổng quan

FlowMind2Digital-DrawIO là một hệ thống AI tiên tiến để chuyển đổi hình ảnh flowchart thành các file .drawio có thể chỉnh sửa được. Hệ thống sử dụng Detectron2 để phát hiện và nhận dạng các thành phần flowchart, sau đó tạo ra file XML tương thích với Draw.io.

## Tính năng chính

- **Phát hiện hình dạng**: Nhận dạng các hình dạng flowchart cơ bản
  - Hình chữ nhật bo góc (start/end)
  - Hình thoi (decision)
  - Hình chữ nhật (process)
  - Hình tròn (connector)
  - Hình lục giác (preparation)
  - Hình bình hành (input/output)

- **Nhận dạng kết nối**: Phát hiện mũi tên và đường nối giữa các hình
- **Tạo file Draw.io**: Xuất kết quả dưới dạng file .drawio
- **Tối ưu hóa layout**: Tự động căn chỉnh và sắp xếp các thành phần
- **Không phụ thuộc PaddleOCR**: Sử dụng các thư viện OCR thay thế

## Cài đặt

### Yêu cầu hệ thống

- **Python 3.8+** (bắt buộc)
- **GPU**: Tùy chọn - hệ thống hoạt động tốt trên CPU
- **RAM**: 
  - CPU only: 4GB minimum, 8GB khuyến nghị
  - GPU: 8GB minimum, 16GB khuyến nghị
- **Dung lượng**: 3-5GB (tùy OCR engines)

### 🚀 Cài đặt tự động (Khuyến nghị)

```bash
cd drawio/
python install.py
```

Script sẽ:
- ✅ Kiểm tra Python version
- 🔍 Tự động phát hiện GPU/CPU
- 📦 Cài đặt dependencies phù hợp
- 🧪 Test các thành phần
- 📋 Đưa ra báo cáo kết quả

### 🛠️ Cài đặt thủ công

#### Bước 1: Môi trường
```bash
git clone <repository-url>
cd flowmind2digital/drawio
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

#### Bước 2: PyTorch
```bash
# Tự động phát hiện và cài đặt
pip install torch torchvision torchaudio

# Hoặc cài đặt cụ thể:
# CPU only: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Bước 3: Detectron2
```bash
# CPU only (không có GPU)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# GPU CUDA 11.3
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Các phiên bản khác: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

#### Bước 4: Dependencies
```bash
pip install -r requirements.txt
```

#### Bước 5: OCR Engines

**Option 1: Tesseract (Lightweight, CPU-friendly)**
```bash
# Install pytesseract
pip install pytesseract

# Install Tesseract binary:
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Mac: brew install tesseract
```

**Option 2: EasyOCR (Good accuracy)**
```bash
pip install easyocr
```

**Option 3: PaddleOCR (Best accuracy, may conflict)**
```bash
pip install paddlepaddle
pip install paddleocr
```

### 🧪 Kiểm tra cài đặt

```python
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

import detectron2
print('Detectron2: OK')

# Test OCR
try:
    import pytesseract
    print('Tesseract: OK')
except: pass

try:
    import easyocr
    print('EasyOCR: OK')
except: pass

try:
    import paddleocr
    print('PaddleOCR: OK')
except: pass
"
```

## Cách sử dụng

### 1. Chuẩn bị dữ liệu

Tạo cấu trúc thư mục:
```
dataset_drawio/
├── train/
│   ├── images/           # Ảnh flowchart
│   ├── annotations/      # File .drawio tương ứng
│   └── config.txt        # Danh sách file annotation
└── val/
    ├── images/
    ├── annotations/
    └── config.txt
```

### 2. Tiền xử lý dữ liệu

```bash
python preprocess.py
```

### 3. Huấn luyện mô hình

```bash
python train_keypoint.py
```

### 4. Dự đoán và tạo file Draw.io

```python
from generator import DrawioGenerator
from predict import predict_mode

# Khởi tạo
generator = DrawioGenerator()
predictor = predict_mode()

# Dự đoán từ ảnh
img_path = "path/to/your/flowchart.jpg"
bbox, cls, kpt, siz = model(img_path=img_path, opt=0)

# Tạo file Draw.io
pred = get_pred(bbox, cls)
edge = get_edge(kpt, cls)
edge = build_graph(pred, edge)
pred, edge, scale = scaling(pred, edge, siz)

xml_root = generator.create_drawio_xml(pred, edge)
generator.save_drawio_file(xml_root, "output.drawio")
```

### 5. Chạy toàn bộ pipeline

```bash
python generator.py
```

## Cấu hình

### Các tham số quan trọng

```python
# Trong train_keypoint.py
cfg.SOLVER.IMS_PER_BATCH = 2        # Batch size
cfg.SOLVER.BASE_LR = 0.00025        # Learning rate
cfg.SOLVER.MAX_ITER = 5000          # Số vòng lặp huấn luyện
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Ngưỡng confidence

# Trong generator.py
page_width = 850                     # Chiều rộng trang Draw.io
page_height = 1100                   # Chiều cao trang Draw.io
```

### Tùy chỉnh loại hình dạng

Sửa đổi dictionary `category` trong các file:
```python
category = {
    'rounded_rectangle': 0,  # start/end
    'diamond': 1,            # decision
    'rectangle': 2,          # process
    'circle': 3,             # connector
    'hexagon': 4,            # preparation
    'parallelogram': 5,      # input/output
    'text': 6,               # text
    'arrow': 7,              # connector line
    'line': 8                # simple line
}
```

## Cấu trúc thư mục

```
drawio/
├── data_preprocess.py    # Xử lý dữ liệu .drawio
├── predict.py            # Dự đoán từ mô hình
├── generator.py          # Tạo file .drawio
├── train_keypoint.py     # Huấn luyện mô hình
├── requirements.txt      # Danh sách thư viện
├── README.md            # Tài liệu này
└── units/
    ├── geometry.py       # Tính toán hình học
    └── nms.py           # Non-maximum suppression
```

## So sánh với phiên bản PPT

| Tính năng | PPT Version | DrawIO Version |
|-----------|-------------|----------------|
| Format đầu ra | .pptx | .drawio |
| OCR Engine | PaddleOCR | Tesseract/EasyOCR |
| Xung đột thư viện | Có (PaddlePaddle) | Không |
| Tương thích | PowerPoint | Draw.io, Diagrams.net |
| Khả năng chỉnh sửa | Hạn chế | Cao |

## Xử lý lỗi thường gặp

### 1. Lỗi cài đặt Detectron2
```bash
# Thử phiên bản khác
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

### 2. Lỗi OCR
```bash
# Kiểm tra Tesseract
tesseract --version

# Cài đặt gói ngôn ngữ
sudo apt-get install tesseract-ocr-vie  # Tiếng Việt
```

### 3. Lỗi GPU memory
```python
# Giảm batch size
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
```

### 4. File .drawio không mở được
- Kiểm tra cấu trúc XML
- Đảm bảo encoding UTF-8
- Validate với Draw.io online

## API Reference

### DrawioGenerator Class

```python
class DrawioGenerator:
    def create_drawio_xml(self, shapes, connections, texts=None, page_width=850, page_height=1100)
    def save_drawio_file(self, xml_element, filename)
```

### predict_mode Class

```python
class predict_mode:
    def __init__(self, category=None, keypoint_names=None, keypoint_flip_map=None)
    def get_drawio_dicts(self, img_dir)
    def dataset_register(self, dataset_path)
    def predict_flowchart(self, img_path, drawio_metadata=None, save_path=None)
```

## Đóng góp

1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Mở Pull Request

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## Liên hệ

- Email: caijianfeng@example.com
- Issues: [GitHub Issues](repository-url/issues)

## Changelog

### v2.0.0 (2024-01-20)
- Thêm hỗ trợ format .drawio
- Loại bỏ dependency PaddleOCR
- Cải thiện hiệu suất và độ ổn định
- Thêm tài liệu chi tiết

### v1.0.0 (2022-12-29)
- Phiên bản đầu tiên hỗ trợ .pptx
- Tích hợp PaddleOCR
- Hỗ trợ các hình dạng cơ bản
