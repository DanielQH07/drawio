# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: data_preprocess.py
@Date: 2024/01/20 12:52
@Author: caijianfeng (modified for drawio)
"""
import xml.etree.ElementTree as ET
import torch

class data_process:
    def __init__(self, category=None):
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

    def get_drawio_dicts(self, drawio_file):
        """
        Parse .drawio XML file and extract shape information
        """
        record = {}
        tree = ET.parse(drawio_file)
        root = tree.getroot()
        
        # Get diagram element
        diagram = root.find('diagram')
        if diagram is None:
            print("No diagram found in drawio file")
            return None
            
        # Parse mxGraphModel
        model = diagram.find('mxGraphModel')
        if model is None:
            print("No mxGraphModel found")
            return None
        
        # Get page dimensions
        page_width = float(model.get('pageWidth', 850))
        page_height = float(model.get('pageHeight', 1100))
        
        record['height'] = int(page_height)
        record['width'] = int(page_width)
        
        # Parse root and cells
        model_root = model.find('root')
        cells = model_root.findall('mxCell')
        
        boxes = []
        classes = []
        arrow_keypoints = []
        
        for cell in cells:
            cell_id = cell.get('id')
            if cell_id in ['0', '1']:  # Skip default cells
                continue
                
            # Get geometry
            geometry = cell.find('mxGeometry')
            if geometry is None:
                continue
                
            x = float(geometry.get('x', 0))
            y = float(geometry.get('y', 0))
            width = float(geometry.get('width', 0))
            height = float(geometry.get('height', 0))
            
            if width == 0 or height == 0:
                continue
                
            # Create bounding box
            bbox = [int(x), int(y), int(x + width), int(y + height)]
            boxes.append(bbox)
            
            # Determine shape type from style or value
            value = cell.get('value', '')
            style = cell.get('style', '')
            
            shape_class = self._get_shape_class(style, value)
            classes.append(shape_class)
            
            # For arrows, extract connection points
            if 'edge' in style.lower() or 'connector' in style.lower():
                # Extract source and target points
                keypoints = self._extract_edge_keypoints(cell, geometry)
            else:
                keypoints = [[0, 0, 0], [0, 0, 0]]
            
            arrow_keypoints.append(keypoints)
        
        return torch.tensor(boxes), torch.tensor(classes), torch.tensor(arrow_keypoints)
    
    def _get_shape_class(self, style, value):
        """Determine shape class from style string"""
        style_lower = style.lower()
        
        if 'rounded' in style_lower or 'ellipse' in style_lower:
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
            # Default to text if contains text content
            if value and value.strip():
                return self.category['text']
            return self.category['rectangle']
    
    def _extract_edge_keypoints(self, cell, geometry):
        """Extract start and end points for edges/connectors"""
        source = cell.get('source')
        target = cell.get('target')
        
        # If we have source and target references, we'd need to look them up
        # For now, use geometry points
        source_point = geometry.find('mxPoint[@as="sourcePoint"]')
        target_point = geometry.find('mxPoint[@as="targetPoint"]')
        
        if source_point is not None and target_point is not None:
            sx = float(source_point.get('x', 0))
            sy = float(source_point.get('y', 0))
            tx = float(target_point.get('x', 0))
            ty = float(target_point.get('y', 0))
            return [[sx, sy, 2], [tx, ty, 2]]
        else:
            # Default to geometry bounds
            x = float(geometry.get('x', 0))
            y = float(geometry.get('y', 0))
            width = float(geometry.get('width', 0))
            height = float(geometry.get('height', 0))
            return [[x, y, 2], [x + width, y + height, 2]]
