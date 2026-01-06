import os
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DualFoVTrafficDataset(Dataset):
    
    def __init__(
        self,
        root_dir: str = '../Unified/Dataset/',
        split: str = 'train',
        split_file: Optional[str] = None,
        odd_list: List[str] = ['highway', 'night', 'rainy', 'urban'],
        sequence_length: int = 30,
        cameras: List[str] = ['F_MIDRANGECAM_C', 'F_LONGRANGECAM_C'],
        transform=None,
        load_traffic_lights: bool = True,
        load_traffic_signs: bool = True,
        img_size: Tuple[int, int] = (1920, 1080),
        stride: int = 30,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.odd_list = odd_list
        self.sequence_length = sequence_length
        self.cameras = cameras
        self.transform = transform
        self.load_traffic_lights = load_traffic_lights
        self.load_traffic_signs = load_traffic_signs
        self.img_size = img_size
        self.stride = stride
        
        self.class_names = [
            'stop_sign',
            'speed_limit_35',
            'speed_limit_45', 
            'speed_limit_55',
            'traffic_light_red',
            'traffic_light_green',
            'traffic_light_yellow',
            'one_way',
            'yield'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        if split_file and Path(split_file).exists():
            self.sequences = self._load_from_split(split_file, split)
        else:
            self.sequences = self._build_sequence_index()
        
        logger.info(f"Loaded {len(self.sequences)} sequences for split '{split}'")
    
    def _load_from_split(self, split_file: str, split: str) -> List[Dict]:
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        if split not in split_data:
            raise ValueError(f"Split '{split}' not found in {split_file}")
        
        sequences = []
        for seq_path_str in split_data[split]:
            seq_path = Path(seq_path_str)
            if not seq_path.exists():
                logger.warning(f"Sequence path does not exist: {seq_path}")
                continue
            
            seq_info = self._parse_sequence_folder(seq_path)
            if seq_info:
                sequences.append(seq_info)
        
        return sequences
    
    def _build_sequence_index(self) -> List[Dict]:
        sequences = []
        
        for odd in self.odd_list:
            odd_path = self.root_dir / odd
            if not odd_path.exists():
                logger.warning(f"ODD path not found: {odd_path}")
                continue
            
            for seq_folder in sorted(odd_path.iterdir()):
                if not seq_folder.is_dir():
                    continue
                
                seq_info = self._parse_sequence_folder(seq_folder)
                if seq_info:
                    sequences.append(seq_info)
        
        return sequences
    
    def _parse_sequence_folder(self, seq_folder: Path) -> Optional[Dict]:
        odd = seq_folder.parent.name
        camera_paths = {}
        sensor_path = seq_folder / 'sensor' / 'camera'
        
        if not sensor_path.exists():
            return None
        
        for camera in self.cameras:
            cam_path = sensor_path / camera
            if not cam_path.exists():
                return None
            camera_paths[camera] = cam_path
        
        first_camera = self.cameras[0]
        frame_files = sorted(camera_paths[first_camera].glob('*.jpg'))
        num_frames = len(frame_files)
        
        if num_frames < self.sequence_length:
            return None
        
        traffic_light_path = seq_folder / 'traffic_light' / 'box' / '3d_body'
        traffic_sign_path = seq_folder / 'traffic_sign' / 'box' / '3d_body'
        calibration_path = seq_folder / 'sensor' / 'calibration' / 'calibration.json'
        
        seq_info = {
            'odd': odd,
            'sequence_folder': seq_folder,
            'camera_paths': camera_paths,
            'num_frames': num_frames,
            'traffic_light_path': traffic_light_path if traffic_light_path.exists() else None,
            'traffic_sign_path': traffic_sign_path if traffic_sign_path.exists() else None,
            'calibration_path': calibration_path if calibration_path.exists() else None,
        }
        
        return seq_info
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _load_frame(self, camera_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        camera_name = camera_path.name
        frame_name = f"{camera_name}_{frame_idx:07d}.jpg"
        frame_path = camera_path / frame_name
        
        if not frame_path.exists():
            logger.debug(f"Frame not found: {frame_path}")
            return None
        
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning(f"Failed to load image: {frame_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_annotations(self, annotation_path: Path, frame_idx: int) -> List[Dict]:
        if annotation_path is None or not annotation_path.exists():
            return []
        
        anno_file = annotation_path / f"frame_{frame_idx:07d}.json"
        
        if not anno_file.exists():
            return []
        
        try:
            with open(anno_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {anno_file}")
            return []
        
        annotations = []
        
        if 'objects' in data:
            for obj in data['objects']:
                anno = {
                    'class_name': obj.get('class', 'unknown'),
                    'bbox_3d': obj.get('bbox_3d', {}),
                    'bbox_2d': obj.get('bbox_2d', {}),
                    'confidence': obj.get('confidence', 1.0),
                    'occlusion': obj.get('occlusion', 0.0),
                    'state': obj.get('state', None),
                }
                annotations.append(anno)
        
        return annotations
    
    def _parse_annotations_to_yolo(
        self, 
        annotations: List[Dict], 
        img_width: int, 
        img_height: int
    ) -> np.ndarray:
        boxes = []
        
        for anno in annotations:
            class_name = self._normalize_class_name(anno['class_name'])
            
            if class_name not in self.class_to_idx:
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            if 'bbox_2d' in anno and anno['bbox_2d']:
                bbox = anno['bbox_2d']
                x_min = bbox.get('x_min', bbox.get('xmin', 0))
                y_min = bbox.get('y_min', bbox.get('ymin', 0))
                x_max = bbox.get('x_max', bbox.get('xmax', img_width))
                y_max = bbox.get('y_max', bbox.get('ymax', img_height))
            else:
                continue
            
            x_min = max(0, min(x_min, img_width))
            y_min = max(0, min(y_min, img_height))
            x_max = max(0, min(x_max, img_width))
            y_max = max(0, min(y_max, img_height))
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            boxes.append([class_idx, x_center, y_center, width, height])
        
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        
        return np.array(boxes, dtype=np.float32)
    
    def _normalize_class_name(self, class_name: str) -> str:
        class_name = class_name.lower().strip()
        
        if 'stop' in class_name:
            return 'stop_sign'
        elif 'speed' in class_name or 'limit' in class_name:
            if '35' in class_name:
                return 'speed_limit_35'
            elif '45' in class_name:
                return 'speed_limit_45'
            elif '55' in class_name:
                return 'speed_limit_55'
            else:
                return 'speed_limit_45'
        elif 'traffic_light' in class_name or 'light' in class_name:
            if 'red' in class_name:
                return 'traffic_light_red'
            elif 'green' in class_name:
                return 'traffic_light_green'
            elif 'yellow' in class_name or 'amber' in class_name:
                return 'traffic_light_yellow'
            else:
                return 'traffic_light_red'
        elif 'one_way' in class_name or 'oneway' in class_name:
            return 'one_way'
        elif 'yield' in class_name:
            return 'yield'
        
        return class_name
    
    def _load_calibration(self, calibration_path: Optional[Path]) -> Dict:
        if calibration_path is None or not calibration_path.exists():
            return {}
        
        try:
            with open(calibration_path, 'r') as f:
                calibration = json.load(f)
            return calibration
        except:
            return {}
    
    def __getitem__(self, idx: int) -> Dict:
        seq_info = self.sequences[idx]
        
        max_start = seq_info['num_frames'] - self.sequence_length
        if self.split == 'train':
            start_frame = np.random.randint(1, max(2, max_start + 1))
        else:
            start_frame = 1
        
        calibration = self._load_calibration(seq_info['calibration_path'])
        
        frames = {camera: [] for camera in self.cameras}
        sequence_annotations = []
        valid_frames = []
        
        for i in range(self.sequence_length):
            frame_idx = start_frame + i
            
            if frame_idx > seq_info['num_frames']:
                break
            
            frame_valid = True
            frame_data = {}
            
            for camera in self.cameras:
                img = self._load_frame(seq_info['camera_paths'][camera], frame_idx)
                if img is None:
                    frame_valid = False
                    break
                frames[camera].append(img)
            
            if not frame_valid:
                continue
            
            valid_frames.append(frame_idx)
            
            frame_annos = {'frame_idx': frame_idx}
            all_annos = []
            
            if self.load_traffic_lights and seq_info['traffic_light_path']:
                tl_annos = self._load_annotations(seq_info['traffic_light_path'], frame_idx)
                all_annos.extend(tl_annos)
            
            if self.load_traffic_signs and seq_info['traffic_sign_path']:
                ts_annos = self._load_annotations(seq_info['traffic_sign_path'], frame_idx)
                all_annos.extend(ts_annos)
            
            frame_annos['raw_annotations'] = all_annos
            yolo_boxes = self._parse_annotations_to_yolo(all_annos, self.img_size[0], self.img_size[1])
            frame_annos['boxes'] = yolo_boxes
            
            sequence_annotations.append(frame_annos)
        
        if len(frames[self.cameras[0]]) == 0:
            logger.warning(f"No valid frames found for sequence {seq_info['sequence_folder']}")
            return self.__getitem__((idx + 1) % len(self))
        
        mid_range = np.stack(frames[self.cameras[0]], axis=0)
        long_range = np.stack(frames[self.cameras[1]], axis=0)
        
        if self.transform:
            mid_range, long_range = self.transform(mid_range, long_range)
        else:
            mid_range = torch.from_numpy(mid_range).permute(0, 3, 1, 2).float() / 255.0
            long_range = torch.from_numpy(long_range).permute(0, 3, 1, 2).float() / 255.0
        
        return {
            'mid_range': mid_range,
            'long_range': long_range,
            'annotations': sequence_annotations,
            'odd': seq_info['odd'],
            'sequence_id': seq_info['sequence_folder'].name,
            'calibration': calibration,
            'valid_frames': valid_frames,
        }


def collate_fn(batch):
    mid_range_list = []
    long_range_list = []
    annotations_list = []
    odd_list = []
    sequence_id_list = []
    calibration_list = []
    valid_frames_list = []
    
    for item in batch:
        mid_range_list.append(item['mid_range'])
        long_range_list.append(item['long_range'])
        annotations_list.append(item['annotations'])
        odd_list.append(item['odd'])
        sequence_id_list.append(item['sequence_id'])
        calibration_list.append(item['calibration'])
        valid_frames_list.append(item['valid_frames'])
    
    return {
        'mid_range': torch.stack(mid_range_list),
        'long_range': torch.stack(long_range_list),
        'annotations': annotations_list,
        'odd': odd_list,
        'sequence_id': sequence_id_list,
        'calibration': calibration_list,
        'valid_frames': valid_frames_list,
    }