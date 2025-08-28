#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: demo_inference.py
@Date: 2024/01/20 20:10
@Author: AI Assistant
"""
import os
import sys

def check_requirements():
    """Check if basic requirements are met"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   GPU available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import detectron2
        print(f"✅ Detectron2: {detectron2.__version__}")
    except ImportError:
        print("❌ Detectron2 not found")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    return True

def find_weight_file():
    """Find the model weight file"""
    possible_paths = [
        '../weights/model_final_80k_add_simple.pth',
        'weights/model_final_80k_add_simple.pth',
        './weights/model_final_80k_add_simple.pth',
        os.path.join('..', 'weights', 'model_final_80k_add_simple.pth')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📦 Found weight file: {path}")
            return path
    
    print("❌ Weight file not found. Tried:")
    for path in possible_paths:
        print(f"   - {path}")
    return None

def find_test_image():
    """Find test image"""
    possible_images = [
        'test.jpg',
        '../ppt/test.jpg',
        'sample.jpg',
        'demo.jpg'
    ]
    
    for img in possible_images:
        if os.path.exists(img):
            print(f"📸 Found test image: {img}")
            return img
    
    print("❌ No test image found. Tried:")
    for img in possible_images:
        print(f"   - {img}")
    return None

def main():
    """Quick demo inference"""
    print("🚀 FlowMind2Digital Quick Demo")
    print("=" * 40)
    
    # Check requirements
    print("1️⃣ Checking requirements...")
    if not check_requirements():
        print("💡 Run: pip install torch detectron2 opencv-python")
        return False
    
    # Find weight file
    print("\n2️⃣ Looking for model weights...")
    weight_path = find_weight_file()
    if not weight_path:
        print("💡 Make sure model_final_80k_add_simple.pth is in weights/ folder")
        return False
    
    # Find test image
    print("\n3️⃣ Looking for test image...")
    img_path = find_test_image()
    if not img_path:
        print("💡 Place a flowchart image as 'test.jpg' in current directory")
        return False
    
    # Run inference
    print("\n4️⃣ Running inference...")
    try:
        # Import modules
        from generator import model, get_pred, get_edge, build_graph, scaling, align, adjust_shape, DrawioGenerator
        
        print("   🔍 Loading model and running prediction...")
        bbox, cls, kpt, siz = model(img_path=img_path, opt=0)
        
        if len(bbox) == 0:
            print("   ⚠️  No objects detected!")
            return False
        
        print(f"   ✅ Detected {len(bbox)} objects")
        
        # Process results
        print("   🎨 Processing results...")
        pred = get_pred(bbox, cls)
        edge = get_edge(kpt, cls)
        
        if len(pred) == 0:
            print("   ⚠️  No valid shapes found!")
            return False
        
        print(f"   - Shapes: {len(pred)}")
        print(f"   - Connections: {len(edge)}")
        
        # Build graph and optimize
        edge = build_graph(pred, edge)
        pred, edge, scale = scaling(pred, edge, siz)
        pred = align(pred)
        pred = adjust_shape(pred)
        
        # Generate output
        print("   📄 Generating Draw.io file...")
        generator = DrawioGenerator()
        xml_root = generator.create_drawio_xml(pred, edge, [])  # No OCR for quick demo
        generator.save_drawio_file(xml_root, "demo_result.drawio")
        
        print("\n🎉 Demo completed successfully!")
        print(f"📄 Output: demo_result.drawio")
        print(f"🖼️  Visualization: output/inference_result.jpg")
        print("\n💡 Open demo_result.drawio in Draw.io to see the result!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Demo failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✨ Demo successful! Check the output files.")
