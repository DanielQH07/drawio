#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: install.py
@Date: 2024/01/20 18:30
@Author: AI Assistant
"""
import subprocess
import sys
import platform
import torch

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu():
    """Check GPU availability"""
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU detected: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("💻 No GPU detected - will use CPU")
            return False
    except:
        print("💻 PyTorch not installed yet - will detect GPU later")
        return False

def install_package(package, description=""):
    """Install a package with error handling"""
    try:
        print(f"📦 Installing {package}...")
        if description:
            print(f"   Purpose: {description}")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def install_detectron2(use_gpu=False):
    """Install Detectron2 based on system configuration"""
    print("\n🔧 Installing Detectron2...")
    
    if use_gpu:
        # Try to detect CUDA version
        try:
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"🎯 CUDA version detected: {cuda_version}")
                # Map common CUDA versions to wheel URLs
                cuda_wheels = {
                    "11.3": "cu113",
                    "11.1": "cu111", 
                    "10.2": "cu102"
                }
                
                cuda_short = cuda_wheels.get(cuda_version, "cu113")  # Default to 11.3
                wheel_url = f"https://dl.fbaipublicfiles.com/detectron2/wheels/{cuda_short}/torch1.10/index.html"
            else:
                print("⚠️  CUDA version not detected, using default")
                wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html"
        except:
            wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html"
    else:
        print("💻 Installing CPU-only version")
        wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html"
    
    cmd = [sys.executable, "-m", "pip", "install", "detectron2", "-f", wheel_url]
    try:
        subprocess.check_call(cmd)
        print("✅ Detectron2 installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Detectron2 installation failed: {e}")
        print("💡 Try manual installation from: https://detectron2.readthedocs.io/")
        return False

def install_tesseract_instructions():
    """Provide Tesseract installation instructions"""
    print("\n🔤 Tesseract OCR Installation:")
    system = platform.system().lower()
    
    if "windows" in system:
        print("📥 Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("   1. Download Windows installer")
        print("   2. Run installer and note installation path")
        print("   3. Add to PATH or set TESSERACT_CMD environment variable")
    elif "darwin" in system:  # macOS
        print("📥 macOS: brew install tesseract")
    elif "linux" in system:
        print("📥 Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("📥 CentOS/RHEL: sudo yum install tesseract")
    else:
        print("📥 Check your system's package manager for tesseract installation")

def test_installations():
    """Test if key components work"""
    print("\n🧪 Testing installations...")
    
    # Test torch
    try:
        import torch
        print("✅ PyTorch working")
        if torch.cuda.is_available():
            print(f"✅ GPU support: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 CPU mode confirmed")
    except ImportError:
        print("❌ PyTorch not working")
    
    # Test detectron2
    try:
        import detectron2
        print("✅ Detectron2 working")
    except ImportError:
        print("❌ Detectron2 not working")
    
    # Test OCR engines
    ocr_engines = []
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("✅ Tesseract working")
        ocr_engines.append("Tesseract")
    except:
        print("⚠️  Tesseract not working")
    
    try:
        import easyocr
        print("✅ EasyOCR available")
        ocr_engines.append("EasyOCR")
    except ImportError:
        print("⚠️  EasyOCR not available")
    
    try:
        import paddleocr
        print("✅ PaddleOCR available")
        ocr_engines.append("PaddleOCR")
    except ImportError:
        print("⚠️  PaddleOCR not available")
    
    if ocr_engines:
        print(f"🎯 Available OCR engines: {', '.join(ocr_engines)}")
    else:
        print("❌ No OCR engines available")
    
    return len(ocr_engines) > 0

def main():
    """Main installation process"""
    print("🚀 FlowMind2Digital DrawIO Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check GPU
    has_gpu = check_gpu()
    
    print("\n📋 Installation Plan:")
    print("1. Core dependencies (torch, opencv, etc.)")
    print("2. Detectron2 (CPU or GPU version)")
    print("3. OCR engines (Tesseract + EasyOCR)")
    print("4. Other utilities")
    
    # Ask user for confirmation
    response = input("\n❓ Continue with installation? (y/N): ").lower()
    if response != 'y':
        print("❌ Installation cancelled")
        return False
    
    # Install core packages
    print("\n📦 Installing core packages...")
    core_packages = [
        ("torch>=1.10.0", "Deep learning framework"),
        ("torchvision>=0.11.0", "Computer vision utilities"), 
        ("opencv-python>=4.5.0", "Image processing"),
        ("numpy>=1.21.0", "Numerical computing"),
        ("scipy>=1.7.0", "Scientific computing"),
        ("scikit-learn>=1.0.0", "Machine learning utilities"),
        ("pillow>=8.3.0", "Image processing"),
        ("pandas>=1.3.0", "Data processing"),
        ("matplotlib>=3.5.0", "Plotting"),
        ("tqdm>=4.62.0", "Progress bars"),
        ("lxml>=4.6.0", "XML processing"),
        ("python-pptx>=0.6.21", "PowerPoint generation")
    ]
    
    failed_packages = []
    for package, desc in core_packages:
        if not install_package(package, desc):
            failed_packages.append(package)
    
    # Install Detectron2
    if not install_detectron2(has_gpu):
        failed_packages.append("detectron2")
    
    # Install OCR engines
    print("\n🔤 Installing OCR engines...")
    
    # Tesseract
    if install_package("pytesseract>=0.3.8", "Tesseract OCR wrapper"):
        install_tesseract_instructions()
    
    # EasyOCR
    install_package("easyocr>=1.6.0", "Multi-language OCR")
    
    # Optional: PaddleOCR
    paddle_response = input("\n❓ Install PaddleOCR? (better accuracy but may conflict) (y/N): ").lower()
    if paddle_response == 'y':
        install_package("paddlepaddle>=2.4.0", "PaddlePaddle framework")
        install_package("paddleocr>=2.6.0", "PaddleOCR engine")
    
    # Test installations
    print("\n" + "=" * 50)
    ocr_available = test_installations()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    
    if failed_packages:
        print(f"❌ Failed packages: {', '.join(failed_packages)}")
        print("💡 Try installing these manually")
    else:
        print("✅ All core packages installed successfully")
    
    if ocr_available:
        print("✅ At least one OCR engine available")
    else:
        print("⚠️  No OCR engines available - text recognition disabled")
    
    print("\n🎯 Next steps:")
    print("1. Run: python generator.py")
    print("2. Check output files")
    print("3. If issues occur, check the logs for specific errors")
    
    print("\n📖 Documentation: See README.md for detailed usage")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
