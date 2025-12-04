import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional

class DualFoVDatasetLoader:
    """
    Loader for Dual-FoV Traffic Sign and Light Recognition Dataset
    """
    
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: Root directory containing ODD folders (highway/night/rainy/urban)
        """
        self.root_dir = Path(root_dir)
        self.odds = ['highway', 'night', 'rainy', 'urban']
        self.sequences = self._get_all_sequences()
    
    def _get_all_sequences(self) -> Dict[str, List[str]]:
        """Get all sequence paths organized by ODD"""
        sequences = {}
        for odd in self.odds:
            odd_path = self.root_dir / odd
            if odd_path.exists():
                sequences[odd] = sorted([
                    str(seq.name) for seq in odd_path.iterdir() 
                    if seq.is_dir()
                ])
        return sequences
    
    def load_image(self, odd: str, sequence: str, frame_id: str, 
                   camera: str = 'F_MIDRANGECAM_C') -> Image.Image:
        """
        Load a camera image
        
        Args:
            odd: Operational design domain (highway/night/rainy/urban)
            sequence: Sequence folder name
            frame_id: Frame ID (e.g., '0000001')
            camera: Camera name (F_MIDRANGECAM_C or F_LONGRANGECAM_C)
        
        Returns:
            PIL Image object
        """
        img_path = (self.root_dir / odd / sequence / 'sensor' / 'camera' / 
                    camera / f"{camera}_{frame_id}.jpg")
        return Image.open(img_path)
    
    def load_traffic_lights(self, odd: str, sequence: str, 
                           frame_id: str) -> Dict:
        """
        Load traffic light annotations for a frame
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            frame_id: Frame ID (e.g., '0000001')
        
        Returns:
            Dictionary containing annotation data
        """
        anno_path = (self.root_dir / odd / sequence / 'traffic_light' / 
                    'box' / '3d_body' / f"frame_{frame_id}.json")
        with open(anno_path, 'r') as f:
            return json.load(f)
    
    def load_traffic_signs(self, odd: str, sequence: str, 
                          frame_id: str) -> Dict:
        """
        Load traffic sign annotations for a frame
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            frame_id: Frame ID (e.g., '0000001')
        
        Returns:
            Dictionary containing annotation data
        """
        anno_path = (self.root_dir / odd / sequence / 'traffic_sign' / 
                    'box' / '3d_body' / f"frame_{frame_id}.json")
        with open(anno_path, 'r') as f:
            return json.load(f)
    
    def load_calibration(self, odd: str, sequence: str) -> Dict:
        """
        Load camera calibration parameters
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
        
        Returns:
            Dictionary containing calibration parameters
        """
        calib_path = (self.root_dir / odd / sequence / 'sensor' / 
                     'calibration' / 'calibration.json')
        with open(calib_path, 'r') as f:
            return json.load(f)
    
    def load_extrinsics(self, odd: str, sequence: str) -> Dict:
        """
        Load camera extrinsic matrices
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
        
        Returns:
            Dictionary containing extrinsic matrices
        """
        ext_path = (self.root_dir / odd / sequence / 'sensor' / 
                   'calibration' / 'extrinsic_matrices.json')
        with open(ext_path, 'r') as f:
            return json.load(f)
    
    def load_egomotion(self, odd: str, sequence: str) -> Dict:
        """
        Load GNSS/INS trajectory data
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
        
        Returns:
            Dictionary containing ego-motion data
        """
        ego_path = (self.root_dir / odd / sequence / 'sensor' / 
                   'gnssins' / 'egomotion2.json')
        with open(ego_path, 'r') as f:
            return json.load(f)
    
    def get_frame_ids(self, odd: str, sequence: str, 
                     camera: str = 'F_MIDRANGECAM_C') -> List[str]:
        """
        Get all frame IDs for a sequence
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            camera: Camera name to check
        
        Returns:
            List of frame IDs (sorted)
        """
        camera_path = (self.root_dir / odd / sequence / 'sensor' / 
                      'camera' / camera)
        frame_files = sorted(camera_path.glob(f"{camera}_*.jpg"))
        return [f.stem.split('_')[-1] for f in frame_files]
    
    def load_dual_fov_frame(self, odd: str, sequence: str, 
                           frame_id: str) -> Tuple[Image.Image, Image.Image]:
        """
        Load both mid-range and long-range images for a frame
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            frame_id: Frame ID
        
        Returns:
            Tuple of (mid_range_image, long_range_image)
        """
        mid_img = self.load_image(odd, sequence, frame_id, 'F_MIDRANGECAM_C')
        long_img = self.load_image(odd, sequence, frame_id, 'F_LONGRANGECAM_C')
        return mid_img, long_img
    
    def load_complete_frame(self, odd: str, sequence: str, 
                           frame_id: str) -> Dict:
        """
        Load complete data for a single frame (images + annotations)
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            frame_id: Frame ID
        
        Returns:
            Dictionary with all frame data
        """
        mid_img, long_img = self.load_dual_fov_frame(odd, sequence, frame_id)
        
        return {
            'frame_id': frame_id,
            'odd': odd,
            'sequence': sequence,
            'images': {
                'mid_range': mid_img,
                'long_range': long_img
            },
            'annotations': {
                'traffic_lights': self.load_traffic_lights(odd, sequence, frame_id),
                'traffic_signs': self.load_traffic_signs(odd, sequence, frame_id)
            }
        }
    
    def iterate_sequence(self, odd: str, sequence: str, 
                        camera: str = 'F_MIDRANGECAM_C'):
        """
        Generator to iterate through all frames in a sequence
        
        Args:
            odd: Operational design domain
            sequence: Sequence folder name
            camera: Camera to iterate over
        
        Yields:
            Dictionary with frame data for each frame
        """
        frame_ids = self.get_frame_ids(odd, sequence, camera)
        for frame_id in frame_ids:
            yield self.load_complete_frame(odd, sequence, frame_id)