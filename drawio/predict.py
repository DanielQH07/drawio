# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: predict.py
@Date: 2024/01/20 11:29
@Author: caijianfeng (modified for drawio)
"""
import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from units import nms


class predict_mode:
    def __init__(self, category=None, keypoint_names=None, keypoint_flip_map=None):
        self.category = {'rounded_rectangle': 0,  # start/end
                         'diamond': 1,            # decision
                         'rectangle': 2,          # process
                         'circle': 3,             # connector
                         'hexagon': 4,            # preparation
                         'parallelogram': 5,      # input/output
                         'text': 6,               # text
                         'arrow': 7,              # connector line
                         'line': 8                # simple line
                         } if not category else category
        self.keypoint_names = ['begin', 'end'] if not keypoint_names else keypoint_names
        self.keypoint_flip_map = [('begin', 'end')] if not keypoint_flip_map else keypoint_flip_map

    def extra_setup(self):
        # Setup detectron2 logger
        setup_logger()

    def get_drawio_dicts(self, img_dir):
        """Load drawio format dataset"""
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
            
            # For drawio format, we need to extract image filename differently
            # Assuming we have a corresponding image file
            base_filename = os.path.splitext(os.path.basename(dataset_annotation_file))[0]
            filename = os.path.join(dataset_dir, f"{base_filename}.png")  # or .jpg
            
            # Get diagram dimensions
            diagram = root.find('diagram')
            model = diagram.find('mxGraphModel') if diagram is not None else None
            
            if model is not None:
                width = int(float(model.get('pageWidth', 850)))
                height = int(float(model.get('pageHeight', 1100)))
            else:
                width, height = 850, 1100  # default
            
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
                    shape_class = self._get_shape_class(style, value)
                    
                    # Extract keypoints for edges
                    if 'edge' in style.lower():
                        keypoints = self._extract_edge_keypoints(cell, geometry, height)
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

    def _get_shape_class(self, style, value):
        """Determine shape class from style string"""
        style_lower = style.lower()
        
        if 'rounded' in style_lower or ('ellipse' in style_lower and 'rounded' in style_lower):
            return self.category['rounded_rectangle']
        elif 'rhombus' in style_lower or 'diamond' in style_lower:
            return self.category['diamond']
        elif 'rectangle' in style_lower or 'swimlane' in style_lower:
            return self.category['rectangle']
        elif 'ellipse' in style_lower and 'rounded' not in style_lower:
            return self.category['circle']
        elif 'hexagon' in style_lower:
            return self.category['hexagon']
        elif 'parallelogram' in style_lower:
            return self.category['parallelogram']
        elif 'edge' in style_lower or 'connector' in style_lower:
            if 'arrow' in style_lower:
                return self.category['arrow']
            else:
                return self.category['line']
        else:
            if value and value.strip():
                return self.category['text']
            return self.category['rectangle']

    def _extract_edge_keypoints(self, cell, geometry, img_height):
        """Extract keypoints for edge elements"""
        # Try to get source and target points
        source_point = geometry.find('mxPoint[@as="sourcePoint"]')
        target_point = geometry.find('mxPoint[@as="targetPoint"]')
        
        if source_point is not None and target_point is not None:
            x_from = int(float(source_point.get('x', 0)))
            y_from = img_height - int(float(source_point.get('y', 0)))  # Flip Y coordinate
            x_to = int(float(target_point.get('x', 0)))
            y_to = img_height - int(float(target_point.get('y', 0)))    # Flip Y coordinate
            return [x_from, y_from, 2, x_to, y_to, 2]
        else:
            # Fallback to geometry bounds
            x = float(geometry.get('x', 0))
            y = float(geometry.get('y', 0))
            width = float(geometry.get('width', 0))
            height = float(geometry.get('height', 0))
            
            x_from = int(x)
            y_from = img_height - int(y)
            x_to = int(x + width)
            y_to = img_height - int(y + height)
            return [x_from, y_from, 2, x_to, y_to, 2]

    def dataset_register(self, dataset_path):
        for i, d in enumerate(['train', 'eval']):
            DatasetCatalog.register('drawio_' + d, lambda d=d, i=i: self.get_drawio_dicts(dataset_path[i]))
            MetadataCatalog.get('drawio_' + d).set(thing_classes=['{}'.format(categ) for categ in self.category.keys()])
        MetadataCatalog.get("drawio_train").keypoint_names = self.keypoint_names
        MetadataCatalog.get("drawio_train").keypoint_flip_map = self.keypoint_flip_map
        MetadataCatalog.get("drawio_train").evaluator_type = "coco"
        drawio_metadata = MetadataCatalog.get('drawio_train')
        print('register succeed!!')
        return drawio_metadata

    def predict_flowchart(self, img_path, drawio_metadata=None, save_path=None):
        """Predict flowchart from image"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_keypoint_detection.pth")
        
        # Auto-detect device
        if not torch.cuda.is_available():
            print("💻 No GPU detected - using CPU for inference")
            cfg.MODEL.DEVICE = 'cpu'
        else:
            print("🚀 Using GPU for inference")
            
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.category)
        cfg.MODEL.RETINANET.NUM_CLASSES = len(self.category)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(self.keypoint_names)
        cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((len(self.keypoint_names), 1), dtype=float).tolist()

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        predictor = DefaultPredictor(cfg)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        outputs = nms.work(outputs)
        
        if save_path is not None:
            v = Visualizer(im[:, :, ::-1],
                           metadata=drawio_metadata,
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        
        return outputs

    def shapes_connect(self, keypoint, cls, box):
        """Calculate connection points for different shapes"""
        # Similar to original but adapted for drawio shapes
        if cls == 0:  # rounded_rectangle
            up = [(box[0] + box[2]) / 2, box[3]]
            down = [(box[0] + box[2]) / 2, box[1]]
            left = [box[0], (box[3] + box[1]) / 2]
            right = [box[2], (box[3] + box[1]) / 2]
            center = [(box[0] + box[2]) / 2, (box[3] + box[1]) / 2]
            points = torch.tensor([up, center, left, down, right])
        elif cls == 1:  # diamond
            up = [(box[0] + box[2]) / 2, box[3]]
            down = [(box[0] + box[2]) / 2, box[1]]
            left = [box[0], (box[3] + box[1]) / 2]
            right = [box[2], (box[3] + box[1]) / 2]
            points = torch.tensor([down, right, up, left])
        elif cls == 2:  # rectangle
            up = [(box[0] + box[2]) / 2, box[3]]
            down = [(box[0] + box[2]) / 2, box[1]]
            left = [box[0], (box[3] + box[1]) / 2]
            right = [box[2], (box[3] + box[1]) / 2]
            center = [(box[0] + box[2]) / 2, (box[3] + box[1]) / 2]
            points = torch.tensor([down, right, up, left, center])
        elif cls == 3:  # circle
            up = [(box[0] + box[2]) / 2, box[3]]
            down = [(box[0] + box[2]) / 2, box[1]]
            left = [box[0], (box[3] + box[1]) / 2]
            right = [box[2], (box[3] + box[1]) / 2]
            center = [(box[0] + box[2]) / 2, (box[3] + box[1]) / 2]
            points = torch.tensor([up, center, left, down, right])
        else:  # default case
            center = [(box[0] + box[2]) / 2, (box[3] + box[1]) / 2]
            points = torch.tensor([center])
        
        distances = F.pairwise_distance(keypoint, points, p=2)
        point = torch.argmin(distances)
        distance = torch.min(distances)
        return point, distance
