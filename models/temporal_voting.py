import torch
import numpy as np
from typing import List, Dict, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class TemporalVoting:
    
    def __init__(
        self,
        window_size: int = 5,
        quality_weight: bool = True,
        nms_threshold: float = 0.5,
        min_confidence: float = 0.3,
        persistence_threshold: int = 3,
    ):
        self.window_size = window_size
        self.quality_weight = quality_weight
        self.nms_threshold = nms_threshold
        self.min_confidence = min_confidence
        self.persistence_threshold = persistence_threshold
        self.memory_buffer = []
        
        logger.info(f"TemporalVoting initialized with window_size={window_size}, "
                   f"quality_weight={quality_weight}")
    
    def compute_quality_metrics(self, frame: np.ndarray) -> float:
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
        
        contrast = (gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-10)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        sharpness = min(sharpness / 1000.0, 1.0)
        
        quality = contrast * sharpness
        
        return max(0.1, quality)
    
    def vote(
        self,
        sequence_detections: List[Dict],
        sequence_frames: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        T = len(sequence_detections)
        voted_results = []
        
        for t in range(T):
            start_idx = max(0, t - self.window_size // 2)
            end_idx = min(T, t + self.window_size // 2 + 1)
            
            window_detections = sequence_detections[start_idx:end_idx]
            
            if self.quality_weight and sequence_frames is not None:
                weights = []
                for i in range(start_idx, end_idx):
                    if i < len(sequence_frames):
                        frame = sequence_frames[i]
                        quality = self.compute_quality_metrics(frame)
                        weights.append(quality)
                    else:
                        weights.append(0.1)
                weights = np.array(weights)
                weights = weights / (weights.sum() + 1e-10)
            else:
                weights = np.ones(len(window_detections)) / len(window_detections)
            
            aggregated = self._aggregate_detections(window_detections, weights)
            voted_results.append(aggregated)
        
        return voted_results
    
    def _aggregate_detections(
        self,
        detections: List[Dict],
        weights: np.ndarray,
    ) -> Dict:
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for i, det in enumerate(detections):
            weight = weights[i]
            boxes = det['boxes']
            scores = det['scores']
            classes = det['classes']
            
            if len(boxes) == 0:
                continue
            
            weighted_scores = scores * weight
            
            all_boxes.append(boxes)
            all_scores.append(weighted_scores)
            all_classes.append(classes)
        
        if len(all_boxes) == 0:
            return {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'classes': np.array([], dtype=int),
            }
        
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        
        keep_indices = self._nms(all_boxes, all_scores, self.nms_threshold)
        
        final_boxes = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_classes = all_classes[keep_indices]
        
        confidence_mask = final_scores >= self.min_confidence
        final_boxes = final_boxes[confidence_mask]
        final_scores = final_scores[confidence_mask]
        final_classes = final_classes[confidence_mask]
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'classes': final_classes,
        }
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def vote_batch(
        self,
        batch_sequence_detections: List[List[Dict]],
        batch_sequence_frames: Optional[List[np.ndarray]] = None,
    ) -> List[List[Dict]]:
        batch_results = []
        
        for i, seq_dets in enumerate(batch_sequence_detections):
            seq_frames = batch_sequence_frames[i] if batch_sequence_frames else None
            voted = self.vote(seq_dets, seq_frames)
            batch_results.append(voted)
        
        return batch_results
    
    def update_memory(self, detection: Dict):
        self.memory_buffer.append(detection)
        if len(self.memory_buffer) > 15:
            self.memory_buffer.pop(0)
    
    def get_persistent_objects(self) -> List[Dict]:
        if len(self.memory_buffer) < self.persistence_threshold:
            return []
        
        recent_detections = self.memory_buffer[-self.persistence_threshold:]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for det in recent_detections:
            all_boxes.append(det['boxes'])
            all_scores.append(det['scores'])
            all_classes.append(det['classes'])
        
        if len(all_boxes) == 0:
            return []
        
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_classes = np.concatenate(all_classes, axis=0)
        
        keep_indices = self._nms(all_boxes, all_scores, self.nms_threshold)
        
        return [{
            'boxes': all_boxes[keep_indices],
            'scores': all_scores[keep_indices],
            'classes': all_classes[keep_indices],
        }]
    
    def reset_memory(self):
        self.memory_buffer = []