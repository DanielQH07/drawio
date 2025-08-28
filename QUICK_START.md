# 🚀 Quick Start Guide - FlowMind2Digital DrawIO

## Bước 1: Clone Repository

```bash
git clone https://github.com/DanielQH07/drawio.git
cd drawio
```

## Bước 2: Cài đặt Dependencies

### Option A: Cài đặt tự động (Khuyến nghị)
```bash
cd drawio/
python install.py
```

### Option B: Cài đặt thủ công
```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt PyTorch
pip install torch torchvision torchaudio

# Cài đặt Detectron2
# CPU only:
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# GPU CUDA 11.3:
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Cài đặt OCR engines
pip install pytesseract easyocr

# Cài đặt dependencies khác
pip install -r requirements.txt
```

## Bước 3: Download Model Weights

```bash
# Tạo thư mục weights
mkdir weights

# Download model (450MB)
# Bạn cần copy file model_final_80k_add_simple.pth vào weights/
```

## Bước 4: Test Setup

```bash
cd drawio/
python test_setup.py
```

## Bước 5: Chạy Inference

### Quick Demo
```bash
cd drawio/
python demo_inference.py
```

### Full Inference
```bash
cd drawio/
python generator.py [image_path]
# hoặc
python run_inference.py --image test.jpg --output result.drawio
```

## 📂 Cấu trúc thư mục sau cài đặt

```
flowmind2digital/
├── weights/
│   └── model_final_80k_add_simple.pth  # Model weights (cần download)
├── drawio/                              # Main working directory
│   ├── test.jpg                        # Sample image
│   ├── generator.py                    # Main script
│   ├── demo_inference.py               # Quick demo
│   ├── install.py                      # Auto installer
│   └── requirements.txt                # Dependencies
├── ppt/                                # Original PPT version
└── README.md                           # Documentation
```

## 🎯 Kết quả mong đợi

- **Input**: Flowchart image (JPG/PNG)
- **Output**: 
  - `result.drawio` - File Draw.io có thể edit
  - `output/inference_result.jpg` - Visualization

## 📖 Các lệnh hữu ích

```bash
# Test với image khác
python generator.py my_flowchart.jpg

# Chạy với OCR engine cụ thể
python run_inference.py --image test.jpg --engine tesseract

# Bỏ qua OCR
python run_inference.py --image test.jpg --no-ocr

# Kiểm tra setup
python test_setup.py
```

## 🔧 Troubleshooting

### 1. PyTorch không cài được
```bash
# Cài version cụ thể
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

### 2. Detectron2 lỗi
```bash
# Build từ source
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

### 3. OCR không hoạt động
```bash
# Cài Tesseract binary
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu: sudo apt-get install tesseract-ocr
# Mac: brew install tesseract
```

### 4. Model weight không tìm thấy
- Đảm bảo file `model_final_80k_add_simple.pth` ở đúng vị trí `weights/`
- Kiểm tra đường dẫn trong code

## 🎉 Hoàn thành!

Sau khi setup thành công, bạn có thể:
1. Chụp/tải flowchart image
2. Chạy `python generator.py your_image.jpg`
3. Mở `result.drawio` trong Draw.io
4. Edit và export như bình thường!

## 📱 Liên hệ

- GitHub Issues: [Report bugs](https://github.com/DanielQH07/drawio/issues)
- Documentation: Xem README.md trong từng thư mục
