# _*_ coding:utf-8 _*_
"""
@Software: flowmind2digital_drawio
@FileName: preprocess.py
@Date: 2024/01/20 16:59
@Author: caijianfeng (modified for drawio)
"""
import os

# Process drawio dataset
datasets_dir = './hddrawio/'
dataset_split = os.listdir(datasets_dir)  # [train, val, test]

for dataset in dataset_split:
    dataset_path = os.path.join(datasets_dir, dataset)  # ./hddrawio/train
    dataset_ann_dir = os.path.join(dataset_path, 'annotations')  # ./hddrawio/train/annotations
    
    if not os.path.exists(dataset_ann_dir):
        print(f"Directory {dataset_ann_dir} does not exist, skipping...")
        continue
        
    dataset_annotations_files = os.listdir(dataset_ann_dir)
    dataset_config = os.path.join(dataset_path, 'config.txt')  # ./hddrawio/train/config.txt
    
    with open(dataset_config, 'w') as f:
        for dataset_annotations_file in dataset_annotations_files:
            if dataset_annotations_file.endswith('.drawio') or dataset_annotations_file.endswith('.xml'):
                f.write(os.path.join(dataset_ann_dir, dataset_annotations_file) + '\n')
    
    print(f'Dataset config file for {dataset} saved successfully!')
