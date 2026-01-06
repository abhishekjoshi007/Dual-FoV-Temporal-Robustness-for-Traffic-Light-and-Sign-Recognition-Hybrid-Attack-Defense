import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from .baseline_yolov8m import BaselineYOLOv8m

logger = logging.getLogger(__name__)


class UnifiedDefenseStack(nn.Module):
    
    def __init__(
        self,
        base_model: BaselineYOLOv8m,
        bit_depth: int = 5,
        median_kernel: int = 3,
        temperature: float = 3.0,
        entropy_threshold: float = 0.5,
        use_feature_squeeze: bool = True,
        use_temperature_scaling: bool = True,
        use_entropy_gating: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.bit_depth = bit_depth
        self.median_kernel = median_kernel
        self.temperature = temperature
        self.entropy_threshold = entropy_threshold
        self.use_feature_squeeze = use_feature_squeeze
        self.use_temperature_scaling = use_temperature_scaling
        self.use_entropy_gating = use_entropy_gating
        
        logger.info(f"UnifiedDefenseStack initialized with bit_depth={bit_depth}, "
                   f"median_kernel={median_kernel}, temperature={temperature}")
    
    def feature_squeeze(self, images: torch.Tensor) -> torch.Tensor:
        if not self.use_feature_squeeze:
            return images
        
        device = images.device
        images_np = (images * 255).cpu().numpy().astype(np.uint8)
        
        if images_np.ndim == 4:
            B, C, H, W = images.shape
            squeezed = []
            
            for i in range(B):
                img = images_np[i].transpose(1, 2, 0)
                squeezed_img = self._squeeze_single_image(img)
                squeezed.append(squeezed_img)
            
            squeezed = np.stack(squeezed, axis=0)
            squeezed = torch.from_numpy(squeezed).permute(0, 3, 1, 2).float() / 255.0
        else:
            img = images_np.transpose(1, 2, 0)
            squeezed = self._squeeze_single_image(img)
            squeezed = torch.from_numpy(squeezed).permute(2, 0, 1).float() / 255.0
        
        return squeezed.to(device)
    
    def _squeeze_single_image(self, img: np.ndarray) -> np.ndarray:
        quantized = np.floor(img * (2 ** self.bit_depth) / 256) * (256 / (2 ** self.bit_depth))
        quantized = quantized.astype(np.uint8)
        
        if self.median_kernel > 1:
            filtered = cv2.medianBlur(quantized, self.median_kernel)
        else:
            filtered = quantized
        
        return filtered
    
    def temperature_scaling(self, probs: torch.Tensor) -> torch.Tensor:
        if not self.use_temperature_scaling:
            return probs

        return F.softmax(torch.log(probs + 1e-10) / self.temperature, dim=-1)

    def _apply_temperature_to_detections(self, results: List[Dict], temperature: float) -> List[Dict]:
        """Apply temperature-like smoothing to detection confidences

        Note: This is a post-processing approximation since YOLOv8's Ultralytics API
        doesn't expose raw class logits. We apply power transform to approximate
        the effect of temperature scaling on softmax outputs.
        """
        if not self.use_temperature_scaling:
            return results

        smoothed_results = []
        for det in results:
            if len(det['scores']) == 0:
                smoothed_results.append(det)
                continue

            scores = det['scores']
            # Smooth via power transform (approximates softmax temperature effect)
            smoothed = scores ** (1.0 / temperature)
            # Normalize to maintain probability distribution properties
            smoothed = smoothed / (smoothed.sum() + 1e-10)

            smoothed_det = {
                'boxes': det['boxes'],
                'scores': smoothed,
                'classes': det['classes']
            }
            smoothed_results.append(smoothed_det)

        return smoothed_results
    
    def calculate_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    def forward(
        self,
        mid_range: torch.Tensor,
        long_range: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Dict:
        # Layer 1: Feature Squeezing (bit-depth reduction + median filtering)
        mid_squeezed = self.feature_squeeze(mid_range)
        long_squeezed = self.feature_squeeze(long_range)

        # Get base model predictions
        mid_results = self.base_model.predict(mid_squeezed, conf_threshold)
        long_results = self.base_model.predict(long_squeezed, conf_threshold)

        # Layer 2: Temperature Scaling (confidence calibration)
        mid_results = self._apply_temperature_to_detections(mid_results, self.temperature)
        long_results = self._apply_temperature_to_detections(long_results, self.temperature)
        
        # Layer 3: Entropy-Based Cross-FoV Gating
        if self.use_entropy_gating:
            mid_entropy = self._compute_frame_entropy(mid_results)
            long_entropy = self._compute_frame_entropy(long_results)

            # Apply entropy threshold gating
            # Select FoV with lower entropy (higher confidence)
            # Track if selected FoV exceeds threshold (high uncertainty)
            if mid_entropy < long_entropy:
                selected_fov = 'mid_range'
                selected_results = mid_results
                high_uncertainty = mid_entropy > self.entropy_threshold
            else:
                selected_fov = 'long_range'
                selected_results = long_results
                high_uncertainty = long_entropy > self.entropy_threshold
        else:
            selected_fov = 'mid_range'
            selected_results = mid_results
            high_uncertainty = False

        return {
            'mid_results': mid_results,
            'long_results': long_results,
            'selected_fov': selected_fov,
            'selected_results': selected_results,
            'mid_entropy': self._compute_frame_entropy(mid_results) if self.use_entropy_gating else 0.0,
            'long_entropy': self._compute_frame_entropy(long_results) if self.use_entropy_gating else 0.0,
            'high_uncertainty': high_uncertainty,
        }
    
    def _compute_frame_entropy(self, results: List[Dict]) -> float:
        if len(results) == 0:
            return 1.0
        
        entropies = []
        for det in results:
            scores = det['scores']
            if len(scores) == 0:
                entropies.append(1.0)
                continue
            
            probs = torch.from_numpy(scores).float()
            probs = probs / (probs.sum() + 1e-10)
            
            ent = -torch.sum(probs * torch.log(probs + 1e-10))
            entropies.append(ent.item())
        
        if len(entropies) == 0:
            return 1.0
        
        return np.mean(entropies)
    
    def predict_sequence(
        self,
        mid_range_seq: torch.Tensor,
        long_range_seq: torch.Tensor,
        conf_threshold: float = 0.25,
    ) -> Dict:
        if mid_range_seq.dim() == 5:
            B, T, C, H, W = mid_range_seq.shape
            
            all_mid_results = []
            all_long_results = []
            all_selected_fovs = []
            
            for t in range(T):
                mid_frame = mid_range_seq[:, t, :, :, :]
                long_frame = long_range_seq[:, t, :, :, :]
                
                frame_output = self.forward(mid_frame, long_frame, conf_threshold)
                
                all_mid_results.append(frame_output['mid_results'])
                all_long_results.append(frame_output['long_results'])
                all_selected_fovs.append(frame_output['selected_fov'])
            
            return {
                'mid_results': all_mid_results,
                'long_results': all_long_results,
                'selected_fovs': all_selected_fovs,
            }
        else:
            return self.forward(mid_range_seq, long_range_seq, conf_threshold)
    
    def save(self, path: str):
        torch.save({
            'base_model_state': self.base_model.state_dict(),
            'bit_depth': self.bit_depth,
            'median_kernel': self.median_kernel,
            'temperature': self.temperature,
            'entropy_threshold': self.entropy_threshold,
        }, path)
        logger.info(f"Defense stack saved to {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.base_model.load_state_dict(checkpoint['base_model_state'])
        self.bit_depth = checkpoint['bit_depth']
        self.median_kernel = checkpoint['median_kernel']
        self.temperature = checkpoint['temperature']
        self.entropy_threshold = checkpoint['entropy_threshold']
        logger.info(f"Defense stack loaded from {path}")