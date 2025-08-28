#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: run_inference.py
@Date: 2024/01/20 20:00
@Author: AI Assistant
"""
import os
import sys
import time
import argparse

def main():
    """Run inference with the trained model"""
    parser = argparse.ArgumentParser(description='FlowMind2Digital Inference')
    parser.add_argument('--image', '-i', type=str, default='test.jpg', 
                       help='Input image path (default: test.jpg)')
    parser.add_argument('--output', '-o', type=str, default='result.drawio',
                       help='Output .drawio file path (default: result.drawio)')
    parser.add_argument('--engine', '-e', type=str, choices=['paddleocr', 'tesseract', 'easyocr'],
                       help='Preferred OCR engine')
    parser.add_argument('--no-ocr', action='store_true',
                       help='Skip OCR text recognition')
    
    args = parser.parse_args()
    
    print("🚀 FlowMind2Digital DrawIO Inference")
    print("=" * 50)
    print(f"📸 Input image: {args.image}")
    print(f"📄 Output file: {args.output}")
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"❌ Image file not found: {args.image}")
        sys.exit(1)
    
    # Import after argument parsing to catch import errors early
    try:
        from generator import (model, get_pred, get_edge, build_graph, 
                             scaling, align, adjust_shape, work_ocr_smart, 
                             DrawioGenerator)
        print("✅ Modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the drawio/ directory and have installed dependencies")
        sys.exit(1)
    
    # Start inference
    T0 = time.time()
    
    try:
        # Step 1: Model prediction
        print("\n🔍 Step 1: Running model inference...")
        bbox, cls, kpt, siz = model(img_path=args.image, opt=0)
        
        if len(bbox) == 0:
            print("⚠️  No objects detected in the image!")
            print("💡 Try adjusting the detection threshold or using a different image")
            sys.exit(1)
        
        T1 = time.time()
        print(f"⏱️  Model inference time: {T1 - T0:.2f}s")
        
        # Step 2: Process predictions
        print("\n📊 Step 2: Processing predictions...")
        pred = get_pred(bbox, cls)  # Get shapes
        edge = get_edge(kpt, cls)   # Get connections
        
        print(f"   - Detected shapes: {len(pred)}")
        print(f"   - Detected connections: {len(edge)}")
        
        if len(pred) == 0:
            print("❌ No valid shapes found!")
            sys.exit(1)
        
        # Step 3: Build connection graph
        print("\n🔗 Step 3: Building connection graph...")
        edge = build_graph(pred, edge)
        pred, edge, scale = scaling(pred, edge, siz)
        
        # Step 4: Layout optimization
        print("\n🎨 Step 4: Optimizing layout...")
        pred = align(pred)
        pred = adjust_shape(pred)
        
        T2 = time.time()
        print(f"⏱️  Layout optimization time: {T2 - T1:.2f}s")
        
        # Step 5: OCR text recognition
        print("\n🔤 Step 5: Text recognition...")
        if args.no_ocr:
            print("   ⏭️  OCR disabled by user")
            points, txts = [], []
        else:
            points, txts = work_ocr_smart(scale, args.image, args.engine)
            print(f"   - Extracted texts: {len(txts)}")
            if txts:
                print(f"   - Sample texts: {txts[:3]}{'...' if len(txts) > 3 else ''}")
        
        T3 = time.time()
        print(f"⏱️  OCR time: {T3 - T2:.2f}s")
        
        # Step 6: Generate Draw.io XML
        print("\n📄 Step 6: Generating Draw.io file...")
        generator = DrawioGenerator()
        xml_root = generator.create_drawio_xml(pred, edge, txts)
        generator.save_drawio_file(xml_root, args.output)
        
        T4 = time.time()
        print(f"⏱️  Generation time: {T4 - T3:.2f}s")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 Inference completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Total time: {T4 - T0:.2f}s")
        print(f"   - Shapes detected: {len(pred)}")
        print(f"   - Connections: {len(edge)}")
        print(f"   - Text elements: {len(txts)}")
        print(f"📄 Output saved: {args.output}")
        print(f"🖼️  Visualization: output/inference_result.jpg")
        
        print("\n💡 Next steps:")
        print(f"   1. Open {args.output} in Draw.io (https://app.diagrams.net/)")
        print(f"   2. Check output/inference_result.jpg for detection results")
        print(f"   3. Edit the diagram as needed")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
