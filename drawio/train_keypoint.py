# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: train_keypoint.py
@Date: 2024/01/20 16:50
@Author: caijianfeng (modified for drawio)
"""
import random
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

category = {'rounded_rectangle': 0,  # start/end
            'diamond': 1,            # decision
            'rectangle': 2,          # process
            'circle': 3,             # connector
            'hexagon': 4,            # preparation
            'parallelogram': 5,      # input/output
            'text': 6,               # text
            'arrow': 7,              # connector line
            'line': 8                # simple line
            }
keypoint_names = ['begin', 'end']
keypoint_flip_map = [('begin', 'end')]

def get_drawio_dicts(img_dir):
    """Load drawio format dataset for training"""
    dataset_config = os.path.join(img_dir, 'config.txt')
    with open(dataset_config, 'r') as f:
        dataset_annotation_files = f.readlines()

    dataset_dicts = []
    dataset_dir = os.path.join(img_dir, 'images')
    
    for idx, dataset_annotation_file in enumerate(dataset_annotation_files):
        dataset_annotation_file = dataset_annotation_file.strip()
        record = {}
        
        # Parse drawio XML file
        tree = ET.parse(dataset_annotation_file)
        root = tree.getroot()
        
        # Extract image filename
        base_filename = os.path.splitext(os.path.basename(dataset_annotation_file))[0]
        filename = os.path.join(dataset_dir, f"{base_filename}.png")
        
        # Get diagram dimensions
        diagram = root.find('diagram')
        model = diagram.find('mxGraphModel') if diagram is not None else None
        
        if model is not None:
            width = int(float(model.get('pageWidth', 850)))
            height = int(float(model.get('pageHeight', 1100)))
        else:
            width, height = 850, 1100
        
        record['file_name'] = filename
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width
        
        # Parse cells from drawio
        objs = []
        if model is not None:
            model_root = model.find('root')
            cells = model_root.findall('mxCell') if model_root is not None else []
            
            for cell in cells:
                cell_id = cell.get('id')
                if cell_id in ['0', '1']:  # Skip default cells
                    continue
                
                geometry = cell.find('mxGeometry')
                if geometry is None:
                    continue
                
                x = float(geometry.get('x', 0))
                y = float(geometry.get('y', 0))
                cell_width = float(geometry.get('width', 0))
                cell_height = float(geometry.get('height', 0))
                
                if cell_width == 0 or cell_height == 0:
                    continue
                
                bbox = [int(x), int(y), int(x + cell_width), int(y + cell_height)]
                
                # Determine shape type
                style = cell.get('style', '')
                value = cell.get('value', '')
                shape_class = _get_shape_class(style, value)
                
                # Extract keypoints for edges
                if 'edge' in style.lower():
                    keypoints = _extract_edge_keypoints(cell, geometry, height)
                else:
                    keypoints = [0, 0, 0, 0, 0, 0]
                
                obj_info = {
                    'iscrowd': 0,
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': shape_class,
                    'keypoints': keypoints
                }
                objs.append(obj_info)
        
        record['annotations'] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def _get_shape_class(style, value):
    """Determine shape class from style string"""
    style_lower = style.lower()
    
    if 'rounded' in style_lower:
        return category['rounded_rectangle']
    elif 'rhombus' in style_lower or 'diamond' in style_lower:
        return category['diamond']
    elif 'rectangle' in style_lower or 'swimlane' in style_lower:
        return category['rectangle']
    elif 'ellipse' in style_lower:
        return category['circle']
    elif 'hexagon' in style_lower:
        return category['hexagon']
    elif 'parallelogram' in style_lower:
        return category['parallelogram']
    elif 'edge' in style_lower:
        if 'arrow' in style_lower:
            return category['arrow']
        else:
            return category['line']
    else:
        if value and value.strip():
            return category['text']
        return category['rectangle']

def _extract_edge_keypoints(cell, geometry, img_height):
    """Extract keypoints for edge elements"""
    source_point = geometry.find('mxPoint[@as="sourcePoint"]')
    target_point = geometry.find('mxPoint[@as="targetPoint"]')
    
    if source_point is not None and target_point is not None:
        x_from = int(float(source_point.get('x', 0)))
        y_from = img_height - int(float(source_point.get('y', 0)))
        x_to = int(float(target_point.get('x', 0)))
        y_to = img_height - int(float(target_point.get('y', 0)))
        return [x_from, y_from, 2, x_to, y_to, 2]
    else:
        x = float(geometry.get('x', 0))
        y = float(geometry.get('y', 0))
        width = float(geometry.get('width', 0))
        height = float(geometry.get('height', 0))
        
        x_from = int(x)
        y_from = img_height - int(y)
        x_to = int(x + width)
        y_to = img_height - int(y + height)
        return [x_from, y_from, 2, x_to, y_to, 2]

if __name__ == '__main__':
    # Register datasets
    for d in ['train', 'val']:
        DatasetCatalog.register('drawio_' + d, lambda d=d: get_drawio_dicts('hddrawio/' + d))
        MetadataCatalog.get('drawio_' + d).set(thing_classes=['{}'.format(categ) for categ in category.keys()])
    
    MetadataCatalog.get("drawio_train").keypoint_names = keypoint_names
    MetadataCatalog.get("drawio_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("drawio_train").evaluator_type = "coco"
    drawio_metadata = MetadataCatalog.get('drawio_train')
    print('register succeed!!')

    # Training configuration
    print('-----train begin-----')
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ('drawio_train',)
    cfg.DATASETS.TEST = ()
    # Check if CUDA is available
    import torch
    if torch.cuda.is_available():
        print("🚀 GPU detected - using CUDA acceleration")
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.SOLVER.MAX_ITER = 5000
    else:
        print("💻 No GPU detected - using CPU (will be slower)")
        cfg.MODEL.DEVICE = 'cpu'
        cfg.DATALOADER.NUM_WORKERS = 0  # Avoid multiprocessing issues on CPU
        cfg.SOLVER.IMS_PER_BATCH = 1    # Smaller batch for CPU
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64  # Reduce memory usage
        cfg.SOLVER.MAX_ITER = 2000      # Shorter training for CPU testing
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(category)
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((2, 1), dtype=float).tolist()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Save the model
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save('model_final_keypoint_detection')
    print('-----train end------')

    # Evaluation
    print('-----eval begin-----')
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("drawio_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "drawio_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
