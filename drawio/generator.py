# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: generator.py
@Date: 2024/01/20 12:00
@Author: caijianfeng (modified for drawio)
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from sklearn.cluster import KMeans
import math
import random
import time
import os
from predict import predict_mode
from data_preprocess import data_process
from units import geometry
import sys
import uuid


class Canopy:
    """Canopy clustering algorithm for layout optimization"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0

    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    @staticmethod
    def euclideanDistance(vec1, vec2):
        return math.sqrt(((vec1 - vec2)**2).sum())

    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        canopies = []
        while len(self.dataset) > 1:
            rand_index = self.getRandIndex()
            current_center = self.dataset[rand_index]
            current_center_list = []
            delete_list = []
            self.dataset = np.delete(self.dataset, rand_index, 0)
            
            for datum_j in range(len(self.dataset)):
                datum = self.dataset[datum_j]
                distance = self.euclideanDistance(current_center, datum)
                if distance < self.t1:
                    current_center_list.append(datum)
                if distance < self.t2:
                    delete_list.append(datum_j)
            
            self.dataset = np.delete(self.dataset, delete_list, 0)
            canopies.append((current_center, current_center_list))

        k = len(canopies)
        if len(self.dataset) == 1:
            k += 1
        return k


def clustering(X, t1=1.5, t2=0.5, dim=1):
    """Cluster layout elements"""
    X = np.array(X)
    X = X.reshape(-1, dim)
    gc = Canopy(X)
    gc.setThreshold(t1, t2)
    k = gc.clustering()
    print("t2: ", t2, "k: ", k)
    
    if k == 1:
        Y = np.zeros(len(X), dtype='int32')
    else:
        Y = KMeans(n_clusters=k).fit_predict(X)
    
    avg = np.zeros((k, dim))
    cnt = np.zeros((k, dim))
    for x, y in zip(X, Y):
        avg[y] += x
        cnt[y] += 1
    avg = avg / cnt
    
    ret = np.zeros_like(X)
    for i, y in enumerate(Y):
        ret[i] = avg[y]
    return ret


def align(pred):
    """Align bounding boxes using clustering"""
    pred = np.array(pred)
    tx = 1e18
    ty = 1e18
    
    for box in pred:
        tx = min(box[2] - box[0], tx)
        ty = min(box[3] - box[1], ty)
    tx /= 1.618
    ty /= 1.618

    for i in range(4):
        x = pred[:, i]
        if i & 1:
            x = clustering(x, t2=ty)
        else:
            x = clustering(x, t2=tx)
        pred[:, i] = x.reshape(-1)
    return pred.tolist()


def adjust_shape(pred):
    """Adjust shape dimensions"""
    X = np.zeros((len(pred), 2))
    t2 = 1e18
    
    for i, box in enumerate(pred):
        X[i][0] = box[2] - box[0]
        X[i][1] = box[3] - box[1]
        t2 = min(t2, math.sqrt(X[i][0]**2 + X[i][1]**2))
    
    X = clustering(X, dim=2, t2=t2/1.618)

    for i, box in enumerate(pred):
        midx = (box[2] + box[0]) / 2
        box[0] = midx - X[i][0] / 2
        box[2] = midx + X[i][0] / 2

        midy = (box[3] + box[1]) / 2
        box[1] = midy - X[i][1] / 2
        box[3] = midy + X[i][1] / 2
    return pred


class DrawioGenerator:
    """Generate Draw.io XML from flowchart predictions"""
    
    def __init__(self):
        self.category = {'rounded_rectangle': 0,
                         'diamond': 1,
                         'rectangle': 2,
                         'circle': 3,
                         'hexagon': 4,
                         'parallelogram': 5,
                         'text': 6,
                         'arrow': 7,
                         'line': 8}
        self.shape_styles = {
            0: 'rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;',  # rounded_rectangle
            1: 'rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;',    # diamond
            2: 'rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;',  # rectangle
            3: 'ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;',    # circle
            4: 'shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;',  # hexagon
            5: 'shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;',  # parallelogram
            6: 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;'  # text
        }

    def create_drawio_xml(self, shapes, connections, texts=None, page_width=850, page_height=1100):
        """Create Draw.io XML structure"""
        # Create root mxfile element
        mxfile = ET.Element('mxfile')
        mxfile.set('host', 'app.diagrams.net')
        mxfile.set('agent', 'FlowMind2Digital')
        mxfile.set('version', '24.1.0')

        # Create diagram element
        diagram = ET.SubElement(mxfile, 'diagram')
        diagram.set('name', 'Page-1')
        diagram.set('id', str(uuid.uuid4()))

        # Create mxGraphModel
        model = ET.SubElement(diagram, 'mxGraphModel')
        model.set('grid', '1')
        model.set('page', '1')
        model.set('gridSize', '10')
        model.set('guides', '1')
        model.set('tooltips', '1')
        model.set('connect', '1')
        model.set('arrows', '1')
        model.set('fold', '1')
        model.set('pageScale', '1')
        model.set('pageWidth', str(page_width))
        model.set('pageHeight', str(page_height))
        model.set('math', '0')
        model.set('shadow', '0')

        # Create root element
        root = ET.SubElement(model, 'root')

        # Add default cells
        cell0 = ET.SubElement(root, 'mxCell')
        cell0.set('id', '0')

        cell1 = ET.SubElement(root, 'mxCell')
        cell1.set('id', '1')
        cell1.set('parent', '0')

        # Add shapes
        cell_id = 2
        shape_ids = {}
        
        for i, shape in enumerate(shapes):
            shape_id = str(cell_id)
            shape_ids[i] = shape_id
            
            cell = ET.SubElement(root, 'mxCell')
            cell.set('id', shape_id)
            cell.set('value', texts[i] if texts and i < len(texts) else '')
            cell.set('style', self.shape_styles.get(shape[4], self.shape_styles[2]))
            cell.set('vertex', '1')
            cell.set('parent', '1')

            # Add geometry
            geometry = ET.SubElement(cell, 'mxGeometry')
            geometry.set('x', str(shape[0]))
            geometry.set('y', str(shape[1]))
            geometry.set('width', str(shape[2] - shape[0]))
            geometry.set('height', str(shape[3] - shape[1]))
            geometry.set('as', 'geometry')

            cell_id += 1

        # Add connections
        for connection in connections:
            conn_id = str(cell_id)
            
            cell = ET.SubElement(root, 'mxCell')
            cell.set('id', conn_id)
            cell.set('value', '')
            
            # Set connection style
            if connection[6] == 7:  # arrow
                style = 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;'
            else:  # line
                style = 'edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=none;'
            
            cell.set('style', style)
            cell.set('edge', '1')
            cell.set('parent', '1')

            # Set source and target if available
            if connection[0] != -1 and connection[1] < len(shapes):
                cell.set('source', shape_ids[connection[1]])
            if connection[3] != -1 and connection[4] < len(shapes):
                cell.set('target', shape_ids[connection[4]])

            # Add geometry
            geometry = ET.SubElement(cell, 'mxGeometry')
            geometry.set('relative', '1')
            geometry.set('as', 'geometry')

            # Add source and target points if no shape connections
            if connection[0] == -1 or connection[3] == -1:
                if connection[0] == -1:
                    source_point = ET.SubElement(geometry, 'mxPoint')
                    source_point.set('x', str(connection[1]))
                    source_point.set('y', str(connection[2]))
                    source_point.set('as', 'sourcePoint')
                
                if connection[3] == -1:
                    target_point = ET.SubElement(geometry, 'mxPoint')
                    target_point.set('x', str(connection[4]))
                    target_point.set('y', str(connection[5]))
                    target_point.set('as', 'targetPoint')

            cell_id += 1

        return mxfile

    def save_drawio_file(self, xml_element, filename):
        """Save XML to .drawio file"""
        # Convert to string with proper formatting
        rough_string = ET.tostring(xml_element, 'unicode')
        reparsed = minidom.parseString(rough_string)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(reparsed.toprettyxml(indent="  "))


def model(img_path, opt=0):
    """Model prediction function with real weight loading"""
    print(f"🚀 Loading model for prediction on: {img_path}")
    
    if opt == 0:  # Real prediction mode
        import cv2
        import torch
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.data import MetadataCatalog, DatasetCatalog
        from detectron2.structures import BoxMode
        import os
        
        # Setup configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
        
        # Load the actual weight file
        weight_path = os.path.join('..', 'weights', 'model_final_80k_add_simple.pth')
        if not os.path.exists(weight_path):
            # Try alternative paths
            alternative_paths = [
                '../weights/model_final_80k_add_simple.pth',
                'weights/model_final_80k_add_simple.pth',
                './weights/model_final_80k_add_simple.pth',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'model_final_80k_add_simple.pth')
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    weight_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"❌ Could not find model weight file. Tried paths: {alternative_paths}")
        
        print(f"📦 Loading weights from: {weight_path}")
        cfg.MODEL.WEIGHTS = weight_path
        
        # Auto-detect device
        if torch.cuda.is_available():
            print("🚀 Using GPU for inference")
        else:
            print("💻 Using CPU for inference (slower)")
            cfg.MODEL.DEVICE = 'cpu'
        
        # Model configuration for the specific weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold for better detection
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Based on original ppt categories
        cfg.MODEL.RETINANET.NUM_CLASSES = 12
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2  # begin, end
        cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()
        
        # Create predictor
        print("🔧 Initializing predictor...")
        predictor = DefaultPredictor(cfg)
        
        # Load and predict
        print("🔍 Running inference...")
        im = cv2.imread(img_path)
        if im is None:
            raise ValueError(f"❌ Could not load image: {img_path}")
        
        print(f"📸 Image shape: {im.shape}")
        outputs = predictor(im)
        
        # Apply NMS
        try:
            from units import nms
            outputs = nms.work(outputs)
            print("✅ NMS applied")
        except ImportError:
            print("⚠️  NMS module not found, using raw outputs")
        
        predict_output = outputs['instances']
        
        print(f"🎯 Detection results:")
        print(f"   - Found {len(predict_output)} objects")
        print(f"   - Classes: {predict_output.pred_classes.cpu().numpy()}")
        print(f"   - Scores: {predict_output.scores.cpu().numpy()}")
        
        bbox = predict_output.pred_boxes.tensor
        cls = predict_output.pred_classes
        kpt = predict_output.pred_keypoints
        siz = predict_output.image_size
        
        # Save visualization
        os.makedirs('output', exist_ok=True)
        save_path = 'output/inference_result.jpg'
        
        from detectron2.utils.visualizer import Visualizer, ColorMode
        # Create metadata for visualization
        thing_classes = ['circle', 'diamonds', 'long_oval', 'hexagon', 'parallelogram', 
                        'rectangle', 'trapezoid', 'triangle', 'text', 'arrow', 'double_arrow', 'line']
        
        # Simple metadata for visualization
        class SimpleMetadata:
            def __init__(self):
                self.thing_classes = thing_classes
        
        metadata = SimpleMetadata()
        
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        out = v.draw_instance_predictions(predict_output.to("cpu"))
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f"💾 Visualization saved: {save_path}")
        
    else:  # Use sample data
        print("📝 Using sample data mode")
        data_util = data_process()
        bbox, cls, kpt = data_util.get_drawio_dicts('./dataset_drawio/annotations/sample.drawio')
        siz = [1100, 850]  # Default size

    # Convert to numpy
    bbox = bbox.cpu().numpy().astype('int32')
    cls = cls.cpu().numpy().astype('int32') 
    kpt = kpt.cpu().numpy().astype('int32')
    kpt = kpt[:, :, :-1]
    
    print(f"📊 Final results: {len(bbox)} shapes, {len([c for c in cls if c >= 9])} connections")
    return bbox, cls, kpt, siz


def get_pred(bbox, cls):
    """Get shape predictions"""
    ret = list()
    for i, x in enumerate(cls):
        if x >= 7:  # Skip text and connector classes for shapes
            continue
        tmp = bbox[i].tolist()
        tmp.append(x)
        ret.append(tmp)
    return ret


def get_edge(kpt, cls):
    """Get edge/connector predictions"""
    ret = list()
    for i, x in enumerate(cls):
        if x < 7:  # Only process arrow and line classes
            continue
        tmp = np.reshape(kpt[i], -1)
        tmp = tmp.tolist()
        tmp.append(x)
        ret.append(tmp)
    return ret


def find_closest_shape(pred, x, y):
    """Find closest shape to given coordinates"""
    mxdis = 1e18
    op = -1
    sp = -1
    d = -1
    
    for i, shape in enumerate(pred):
        dis, direction = geometry.calc(x, y, shape)
        if dis < mxdis:
            mxdis = dis
            op = 1
            sp = i
            d = direction
    return op, sp, d


def build_graph(pred, edge):
    """Build connection graph"""
    ret = list()
    for e in edge:
        op1, sp1, d1 = find_closest_shape(pred, e[0], e[1])
        if op1 == -1:
            sp1, d1 = e[0], e[1]
        op2, sp2, d2 = find_closest_shape(pred, e[2], e[3])
        if op2 == -1:
            sp2, d2 = e[2], e[3]
        cur = [op1, sp1, d1, op2, sp2, d2, e[4]]
        ret.append(cur)
    return ret


def scaling(pred, edge, siz, target_width=850, target_height=1100):
    """Scale predictions to target dimensions"""
    scale_x = target_width / siz[1]
    scale_y = target_height / siz[0]
    scale = min(scale_x, scale_y)
    
    for box in pred:
        for i in range(4):
            if i % 2 == 0:  # x coordinates
                box[i] *= scale_x
            else:  # y coordinates
                box[i] *= scale_y
    
    for e in edge:
        if e[0] == -1:
            e[1] *= scale_x
            e[2] *= scale_y
        if e[3] == -1:
            e[4] *= scale_x
            e[5] *= scale_y
    
    return pred, edge, scale


def work_ocr_smart(scale, img_path, preferred_engine=None):
    """Smart OCR with multiple engine support and fallback"""
    try:
        from ocr_engines import work_ocr
        print("🔍 Using smart OCR with multiple engines...")
        return work_ocr(scale, img_path, preferred_engine)
    except ImportError:
        print("⚠️  ocr_engines module not found, using simple fallback")
        return work_ocr_simple(img_path)

def work_ocr_simple(img_path):
    """Simple fallback OCR using basic methods"""
    try:
        # Try pytesseract first
        import pytesseract
        from PIL import Image
        
        print("🔍 Using Tesseract OCR...")
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image)
        
        # Simple text extraction - place in center
        if text.strip():
            height, width = image.size
            center_point = [width//2, height//2]
            return [center_point], [text.strip()]
        
    except ImportError:
        print("❌ Tesseract not available")
    except Exception as e:
        print(f"❌ Tesseract failed: {e}")
    
    try:
        # Try easyocr as backup
        import easyocr
        print("🔍 Using EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_path)
        
        points = []
        texts = []
        for result in results:
            if len(result) >= 2 and result[2] > 0.3:  # confidence > 0.3
                bbox = result[0]
                text = result[1]
                # Calculate center
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                points.append([center_x, center_y])
                texts.append(text)
        
        return points, texts
        
    except ImportError:
        print("❌ EasyOCR not available")
    except Exception as e:
        print(f"❌ EasyOCR failed: {e}")
    
    print("⚠️  No OCR engines available - text recognition disabled")
    return [], []


if __name__ == "__main__":
    print("🚀 FlowMind2Digital DrawIO Generator")
    print("=" * 50)
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = './test.jpg'
    
    print(f"📸 Input image: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        print("💡 Usage: python generator.py [image_path]")
        print("💡 Default: python generator.py (uses test.jpg)")
        sys.exit(1)
    
    T0 = time.time()
    
    try:
        # Step 1: Model inference
        print("\n🔍 Step 1: Running model inference...")
        bbox, cls, kpt, siz = model(img_path=img_path, opt=0)
        
        pred = get_pred(bbox, cls)
        edge = get_edge(kpt, cls)
        
        if len(pred) == 0:
            print("❌ No shapes detected in the image!")
            print("💡 Try:")
            print("   - Using a clearer flowchart image")
            print("   - Adjusting MODEL.ROI_HEADS.SCORE_THRESH_TEST in model() function")
            sys.exit(1)
        
        T1 = time.time()
        print(f"✅ Model inference completed in {T1 - T0:.2f}s")
        print(f"   - Detected {len(pred)} shapes")
        print(f"   - Detected {len(edge)} connections")
        
        # Step 2: Graph processing
        print("\n🔗 Step 2: Processing connections and layout...")
        edge = build_graph(pred, edge)
        pred, edge, scale = scaling(pred, edge, siz)
        
        pred = align(pred)
        pred = adjust_shape(pred)
        
        T2 = time.time()
        print(f"✅ Layout optimization completed in {T2 - T1:.2f}s")
        
        # Step 3: OCR
        print("\n🔤 Step 3: Text recognition...")
        points, txts = work_ocr_smart(scale, img_path, preferred_engine='paddleocr')
        
        T3 = time.time()
        print(f"✅ Text processing completed in {T3 - T2:.2f}s")
        if txts:
            print(f"   - Extracted {len(txts)} text elements")
            print(f"   - Sample texts: {txts[:3]}")
        else:
            print("   - No text detected or OCR unavailable")
        
        # Step 4: Generate Draw.io
        print("\n📄 Step 4: Generating Draw.io file...")
        generator = DrawioGenerator()
        xml_root = generator.create_drawio_xml(pred, edge, txts)
        generator.save_drawio_file(xml_root, "result.drawio")
        
        T4 = time.time()
        print(f"✅ File generation completed in {T4 - T3:.2f}s")
        
        # Summary
        print("\n" + "=" * 50)
        print("🎉 Success! Draw.io file generated!")
        print(f"📊 Summary:")
        print(f"   - Total time: {T4 - T0:.2f}s")
        print(f"   - Shapes: {len(pred)}")
        print(f"   - Connections: {len(edge)}")
        print(f"   - Text elements: {len(txts)}")
        print(f"📄 Output files:")
        print(f"   - result.drawio (main output)")
        print(f"   - output/inference_result.jpg (visualization)")
        print(f"\n💡 Next steps:")
        print(f"   1. Open result.drawio in Draw.io (https://app.diagrams.net/)")
        print(f"   2. Check the visualization image for accuracy")
        print(f"   3. Edit the diagram as needed")
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   - Check if the model weight file exists")
        print("   - Ensure all dependencies are installed")
        print("   - Try running demo_inference.py for a simpler test")
        sys.exit(1)
