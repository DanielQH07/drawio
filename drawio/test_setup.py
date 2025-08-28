#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: test_setup.py
@Date: 2024/01/20 19:00
@Author: AI Assistant
"""
import sys
import traceback

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        import cv2
        print("✅ OpenCV:", cv2.__version__)
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow: OK")
    except ImportError:
        print("❌ Pillow not available")
        return False
    
    return True

def test_torch():
    """Test PyTorch installation"""
    print("\n🔥 Testing PyTorch...")
    
    try:
        import torch
        print("✅ PyTorch:", torch.__version__)
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU: {gpu_name} (Count: {gpu_count})")
            print(f"   CUDA: {torch.version.cuda}")
            return "gpu"
        else:
            print("💻 GPU not available - CPU mode")
            return "cpu"
    except ImportError:
        print("❌ PyTorch not available")
        return False

def test_detectron2():
    """Test Detectron2 installation"""
    print("\n🎯 Testing Detectron2...")
    
    try:
        import detectron2
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        print("✅ Detectron2:", detectron2.__version__)
        
        # Test basic config
        cfg = get_cfg()
        print("✅ Config loading: OK")
        return True
    except ImportError:
        print("❌ Detectron2 not available")
        return False
    except Exception as e:
        print(f"❌ Detectron2 error: {e}")
        return False

def test_ocr_engines():
    """Test available OCR engines"""
    print("\n🔤 Testing OCR engines...")
    
    engines = []
    
    # Test Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract: {version}")
        engines.append("tesseract")
    except ImportError:
        print("⚠️  pytesseract not installed")
    except Exception as e:
        print(f"⚠️  Tesseract binary not found: {e}")
    
    # Test EasyOCR
    try:
        import easyocr
        print("✅ EasyOCR: Available")
        engines.append("easyocr")
    except ImportError:
        print("⚠️  EasyOCR not installed")
    except Exception as e:
        print(f"⚠️  EasyOCR error: {e}")
    
    # Test PaddleOCR
    try:
        import paddleocr
        print("✅ PaddleOCR: Available")
        engines.append("paddleocr")
    except ImportError:
        print("⚠️  PaddleOCR not installed")
    except Exception as e:
        print(f"⚠️  PaddleOCR error: {e}")
    
    return engines

def test_ocr_functionality():
    """Test OCR functionality with our smart engine"""
    print("\n🔍 Testing OCR functionality...")
    
    try:
        from ocr_engines import OCRManager
        manager = OCRManager()
        
        info = manager.get_engine_info()
        print(f"Current engine: {info['current']}")
        print(f"Available engines: {info['available']}")
        
        if info['available']:
            print("✅ OCR system ready")
            return True
        else:
            print("⚠️  No OCR engines configured")
            return False
    except ImportError:
        print("⚠️  OCR system not available")
        return False

def test_xml_processing():
    """Test XML processing capabilities"""
    print("\n📄 Testing XML processing...")
    
    try:
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        import lxml
        print("✅ XML processing: OK")
        return True
    except ImportError as e:
        print(f"❌ XML processing error: {e}")
        return False

def create_test_drawio():
    """Create a simple test .drawio file"""
    print("\n🎨 Creating test .drawio file...")
    
    try:
        from generator import DrawioGenerator
        
        # Create simple test data
        shapes = [
            [100, 100, 200, 150, 2],  # rectangle
            [300, 100, 400, 150, 1],  # diamond
        ]
        
        connections = [
            [1, 0, 3, 1, 1, 1, 7]  # arrow from shape 0 to shape 1
        ]
        
        texts = ["Start", "Decision"]
        
        generator = DrawioGenerator()
        xml_root = generator.create_drawio_xml(shapes, connections, texts)
        generator.save_drawio_file(xml_root, "test_output.drawio")
        
        print("✅ Test .drawio file created: test_output.drawio")
        return True
    except Exception as e:
        print(f"❌ .drawio creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 FlowMind2Digital DrawIO Setup Test")
    print("=" * 50)
    
    results = {}
    
    # Test basic imports
    results['basic'] = test_basic_imports()
    
    # Test PyTorch
    results['torch'] = test_torch()
    
    # Test Detectron2
    results['detectron2'] = test_detectron2()
    
    # Test OCR engines
    results['ocr_engines'] = test_ocr_engines()
    
    # Test OCR functionality
    results['ocr_system'] = test_ocr_functionality()
    
    # Test XML processing
    results['xml'] = test_xml_processing()
    
    # Test DrawIO creation
    results['drawio'] = create_test_drawio()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        total += 1
        if result:
            passed += 1
            status = "✅ PASS"
        elif isinstance(result, list):
            if len(result) > 0:
                passed += 1
                status = f"✅ PASS ({len(result)} engines)"
            else:
                status = "⚠️  WARN (no engines)"
        else:
            status = "❌ FAIL"
        
        print(f"{test_name:15} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Recommendations
    print("\n💡 Recommendations:")
    
    if not results['basic']:
        print("❗ Install basic packages: pip install numpy opencv-python pillow")
    
    if not results['torch']:
        print("❗ Install PyTorch: pip install torch torchvision")
    
    if not results['detectron2']:
        print("❗ Install Detectron2 - check installation guide")
    
    if isinstance(results['ocr_engines'], list) and len(results['ocr_engines']) == 0:
        print("❗ Install at least one OCR engine:")
        print("   - Tesseract: pip install pytesseract")
        print("   - EasyOCR: pip install easyocr") 
        print("   - PaddleOCR: pip install paddlepaddle paddleocr")
    
    if results['torch'] == 'cpu':
        print("💻 Running in CPU mode - training will be slower")
        print("   Consider using a GPU-enabled environment for training")
    
    if passed >= total - 1:  # Allow 1 failure
        print("\n🎉 Setup looks good! Ready to use FlowMind2Digital.")
        return True
    else:
        print("\n⚠️  Setup needs attention. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
