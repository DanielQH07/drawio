# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: ocr_engines.py
@Date: 2024/01/20 18:00
@Author: AI Assistant
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import warnings

class OCREngine:
    """Base class for OCR engines"""
    def __init__(self):
        self.name = "Base OCR"
        self.available = False
    
    def extract_text(self, img_path: str) -> Tuple[List, List]:
        """
        Extract text from image
        Returns: (points, texts) where points are coordinates and texts are recognized strings
        """
        raise NotImplementedError

class PaddleOCREngine(OCREngine):
    """PaddleOCR implementation"""
    def __init__(self):
        super().__init__()
        self.name = "PaddleOCR"
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            self.available = True
            print("✅ PaddleOCR loaded successfully")
        except ImportError:
            print("❌ PaddleOCR not available (pip install paddleocr)")
            self.available = False
        except Exception as e:
            print(f"❌ PaddleOCR initialization failed: {e}")
            self.available = False
    
    def extract_text(self, img_path: str) -> Tuple[List, List]:
        if not self.available:
            return [], []
        
        try:
            result = self.ocr.ocr(img_path, det=True)
            if not result or not result[0]:
                return [], []
            
            points = []
            texts = []
            
            for item in result[0]:
                if item and len(item) >= 2:
                    bbox = item[0]  # Bounding box coordinates
                    text_info = item[1]  # (text, confidence)
                    
                    if bbox and text_info:
                        # Get center point of bounding box
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        points.append([center_x, center_y])
                        texts.append(text_info[0] if isinstance(text_info, tuple) else text_info)
            
            return points, texts
        except Exception as e:
            print(f"PaddleOCR extraction failed: {e}")
            return [], []

class TesseractOCREngine(OCREngine):
    """Tesseract OCR implementation"""
    def __init__(self):
        super().__init__()
        self.name = "Tesseract"
        try:
            import pytesseract
            from PIL import Image
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
            self.Image = Image
            self.available = True
            print("✅ Tesseract OCR loaded successfully")
        except ImportError:
            print("❌ pytesseract not available (pip install pytesseract)")
            self.available = False
        except Exception as e:
            print(f"❌ Tesseract not found: {e}")
            print("💡 Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            self.available = False
    
    def extract_text(self, img_path: str) -> Tuple[List, List]:
        if not self.available:
            return [], []
        
        try:
            image = self.Image.open(img_path)
            
            # Get detailed data including bounding boxes
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            
            points = []
            texts = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                # Filter out low confidence and empty text
                if confidence > 30 and text:
                    x = data['left'][i]
                    y = data['top'][i]
                    width = data['width'][i]
                    height = data['height'][i]
                    
                    # Calculate center point
                    center_x = x + width / 2
                    center_y = y + height / 2
                    
                    points.append([center_x, center_y])
                    texts.append(text)
            
            return points, texts
        except Exception as e:
            print(f"Tesseract extraction failed: {e}")
            return [], []

class EasyOCREngine(OCREngine):
    """EasyOCR implementation"""
    def __init__(self):
        super().__init__()
        self.name = "EasyOCR"
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False)  # Force CPU to avoid GPU issues
            self.available = True
            print("✅ EasyOCR loaded successfully")
        except ImportError:
            print("❌ EasyOCR not available (pip install easyocr)")
            self.available = False
        except Exception as e:
            print(f"❌ EasyOCR initialization failed: {e}")
            self.available = False
    
    def extract_text(self, img_path: str) -> Tuple[List, List]:
        if not self.available:
            return [], []
        
        try:
            results = self.reader.readtext(img_path)
            
            points = []
            texts = []
            
            for result in results:
                if len(result) >= 3:
                    bbox = result[0]  # Bounding box coordinates
                    text = result[1]  # Recognized text
                    confidence = result[2]  # Confidence score
                    
                    # Filter by confidence
                    if confidence > 0.3 and text.strip():
                        # Calculate center point
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        
                        points.append([center_x, center_y])
                        texts.append(text.strip())
            
            return points, texts
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}")
            return [], []

class OCRManager:
    """Manages multiple OCR engines with fallback"""
    def __init__(self, preferred_engine: Optional[str] = None):
        self.engines = {
            'paddleocr': PaddleOCREngine(),
            'tesseract': TesseractOCREngine(),
            'easyocr': EasyOCREngine()
        }
        
        # Find available engines
        self.available_engines = [name for name, engine in self.engines.items() if engine.available]
        
        if not self.available_engines:
            print("⚠️  No OCR engines available! Text recognition will be disabled.")
            self.current_engine = None
        else:
            # Set preferred engine or default
            if preferred_engine and preferred_engine in self.available_engines:
                self.current_engine = preferred_engine
            else:
                # Default priority: paddleocr > easyocr > tesseract
                priority = ['paddleocr', 'easyocr', 'tesseract']
                for engine in priority:
                    if engine in self.available_engines:
                        self.current_engine = engine
                        break
            
            print(f"🎯 Using OCR engine: {self.current_engine}")
            print(f"📝 Available engines: {', '.join(self.available_engines)}")
    
    def extract_text(self, img_path: str, scale: float = 1.0) -> Tuple[List, List]:
        """Extract text with automatic fallback"""
        if not self.current_engine:
            print("⚠️  No OCR engine available")
            return [], []
        
        # Try current engine
        engine = self.engines[self.current_engine]
        points, texts = engine.extract_text(img_path)
        
        if points and texts:
            # Scale points if needed
            if scale != 1.0:
                from pptx.util import Inches
                scaled_points = []
                for point in points:
                    scaled_points.append([Inches(point[0] / scale), Inches(point[1] / scale)])
                return scaled_points, texts
            return points, texts
        
        # Fallback to other engines
        print(f"⚠️  {self.current_engine} failed, trying fallback engines...")
        for engine_name in self.available_engines:
            if engine_name != self.current_engine:
                try:
                    fallback_engine = self.engines[engine_name]
                    points, texts = fallback_engine.extract_text(img_path)
                    if points and texts:
                        print(f"✅ Fallback to {engine_name} successful")
                        return points, texts
                except Exception as e:
                    print(f"❌ Fallback {engine_name} failed: {e}")
        
        print("❌ All OCR engines failed")
        return [], []
    
    def switch_engine(self, engine_name: str) -> bool:
        """Switch to a different OCR engine"""
        if engine_name in self.available_engines:
            self.current_engine = engine_name
            print(f"🔄 Switched to OCR engine: {engine_name}")
            return True
        else:
            print(f"❌ Engine {engine_name} not available")
            return False
    
    def get_engine_info(self) -> dict:
        """Get information about available engines"""
        return {
            'current': self.current_engine,
            'available': self.available_engines,
            'all_engines': list(self.engines.keys())
        }

# Global OCR manager instance
ocr_manager = OCRManager()

def work_ocr(scale: float, img_path: str, engine: Optional[str] = None) -> Tuple[List, List]:
    """
    Main OCR function with multiple engine support
    
    Args:
        scale: Scale factor for coordinates
        img_path: Path to image file
        engine: Preferred OCR engine ('paddleocr', 'tesseract', 'easyocr')
    
    Returns:
        Tuple of (points, texts)
    """
    global ocr_manager
    
    if engine and engine != ocr_manager.current_engine:
        ocr_manager.switch_engine(engine)
    
    return ocr_manager.extract_text(img_path, scale)

# Compatibility function
def work_ocr_alternative(img_path: str) -> Tuple[List, List]:
    """Alternative OCR without paddleocr dependency"""
    return work_ocr(1.0, img_path)
