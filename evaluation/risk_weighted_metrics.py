import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)


class RiskWeightedMetrics:
    
    def __init__(
        self,
        class_names: List[str] = None,
        severity_matrix: Optional[np.ndarray] = None,
    ):
        if class_names is None:
            class_names = [
                'stop_sign', 'speed_limit_35', 'speed_limit_45', 'speed_limit_55',
                'traffic_light_red', 'traffic_light_green', 'traffic_light_yellow',
                'one_way', 'yield'
            ]
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        if severity_matrix is None:
            self.severity_matrix = self._create_mutcd_severity_matrix()
        else:
            self.severity_matrix = severity_matrix
        
        logger.info(f"RiskWeightedMetrics initialized with {self.num_classes} classes")
    
    def _create_mutcd_severity_matrix(self) -> np.ndarray:
        n = self.num_classes
        severity = np.ones((n, n), dtype=np.float32)
        
        stop_idx = self.class_names.index('stop_sign') if 'stop_sign' in self.class_names else -1
        red_idx = self.class_names.index('traffic_light_red') if 'traffic_light_red' in self.class_names else -1
        green_idx = self.class_names.index('traffic_light_green') if 'traffic_light_green' in self.class_names else -1
        yellow_idx = self.class_names.index('traffic_light_yellow') if 'traffic_light_yellow' in self.class_names else -1
        
        speed_35_idx = self.class_names.index('speed_limit_35') if 'speed_limit_35' in self.class_names else -1
        speed_45_idx = self.class_names.index('speed_limit_45') if 'speed_limit_45' in self.class_names else -1
        speed_55_idx = self.class_names.index('speed_limit_55') if 'speed_limit_55' in self.class_names else -1
        
        if stop_idx >= 0:
            severity[stop_idx, :] = 8.0
            severity[stop_idx, stop_idx] = 0.0
            
            for speed_idx in [speed_35_idx, speed_45_idx, speed_55_idx]:
                if speed_idx >= 0:
                    severity[stop_idx, speed_idx] = 10.0
        
        if red_idx >= 0 and green_idx >= 0:
            severity[red_idx, green_idx] = 10.0
            severity[green_idx, red_idx] = 10.0
        
        if red_idx >= 0 and yellow_idx >= 0:
            severity[red_idx, yellow_idx] = 6.0
            severity[yellow_idx, red_idx] = 4.0
        
        if green_idx >= 0 and yellow_idx >= 0:
            severity[green_idx, yellow_idx] = 3.0
            severity[yellow_idx, green_idx] = 2.0
        
        speed_indices = [speed_35_idx, speed_45_idx, speed_55_idx]
        for i, idx1 in enumerate(speed_indices):
            if idx1 < 0:
                continue
            for j, idx2 in enumerate(speed_indices):
                if idx2 < 0 or i == j:
                    continue
                speed_diff = abs(i - j) * 10
                if speed_diff <= 5:
                    severity[idx1, idx2] = 1.0
                elif speed_diff <= 10:
                    severity[idx1, idx2] = 2.0
                else:
                    severity[idx1, idx2] = 4.0
        
        np.fill_diagonal(severity, 0.0)
        
        return severity
    
    def compute_map(
        self,
        all_predictions: List[Dict],
        all_ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict:
        per_class_ap = []
        
        for class_idx in range(self.num_classes):
            y_true = []
            y_scores = []
            
            for pred, gt in zip(all_predictions, all_ground_truth):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                pred_classes = pred['classes']
                
                gt_boxes = gt['boxes']
                gt_classes = gt['classes']
                
                class_pred_mask = pred_classes == class_idx
                class_pred_boxes = pred_boxes[class_pred_mask]
                class_pred_scores = pred_scores[class_pred_mask]
                
                class_gt_mask = gt_classes == class_idx
                class_gt_boxes = gt_boxes[class_gt_mask]
                
                matched_gt = set()
                
                for pred_box, score in zip(class_pred_boxes, class_pred_scores):
                    max_iou = 0
                    max_idx = -1
                    
                    for gt_idx, gt_box in enumerate(class_gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = self._compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = gt_idx
                    
                    if max_iou >= iou_threshold and max_idx >= 0:
                        y_true.append(1)
                        matched_gt.add(max_idx)
                    else:
                        y_true.append(0)
                    
                    y_scores.append(score)
                
                for gt_idx in range(len(class_gt_boxes)):
                    if gt_idx not in matched_gt:
                        y_true.append(1)
                        y_scores.append(0.0)
            
            if len(y_true) == 0 or sum(y_true) == 0:
                ap = 0.0
            else:
                y_true = np.array(y_true)
                y_scores = np.array(y_scores)
                ap = average_precision_score(y_true, y_scores)
            
            per_class_ap.append(ap)
        
        mean_ap = np.mean(per_class_ap)
        
        return {
            'mAP': mean_ap,
            'per_class_AP': {self.class_names[i]: per_class_ap[i] for i in range(self.num_classes)},
        }
    
    def compute_risk_weighted_map(
        self,
        all_predictions: List[Dict],
        all_ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict:
        conf_matrix = self._compute_confusion_matrix(all_predictions, all_ground_truth, iou_threshold)
        
        weighted_conf = conf_matrix * self.severity_matrix
        
        total_weighted_errors = weighted_conf.sum() - np.diag(weighted_conf).sum()
        total_predictions = conf_matrix.sum()
        
        rw_error_rate = total_weighted_errors / (total_predictions + 1e-10)
        
        rw_map = max(0, 1.0 - rw_error_rate / 10.0)
        
        per_class_rw_ap = []
        for i in range(self.num_classes):
            class_errors = weighted_conf[i, :].sum() - weighted_conf[i, i]
            class_total = conf_matrix[i, :].sum()
            class_error_rate = class_errors / (class_total + 1e-10)
            class_ap = max(0, 1.0 - class_error_rate / 10.0)
            per_class_rw_ap.append(class_ap)
        
        return {
            'RW-mAP': rw_map,
            'per_class_RW_AP': {self.class_names[i]: per_class_rw_ap[i] for i in range(self.num_classes)},
            'confusion_matrix': conf_matrix,
            'weighted_confusion': weighted_conf,
            'total_weighted_errors': total_weighted_errors,
        }
    
    def compute_attack_success_rate(
        self,
        clean_predictions: List[Dict],
        perturbed_predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict:
        total_objects = 0
        failed_detections = 0
        misclassifications = 0
        
        for clean_pred, pert_pred, gt in zip(clean_predictions, perturbed_predictions, ground_truth):
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            total_objects += len(gt_boxes)
            
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                clean_detected = False
                clean_class = None
                
                for pred_box, pred_class in zip(clean_pred['boxes'], clean_pred['classes']):
                    if self._compute_iou(pred_box, gt_box) >= iou_threshold:
                        clean_detected = True
                        clean_class = pred_class
                        break
                
                if not clean_detected:
                    continue
                
                pert_detected = False
                pert_class = None
                
                for pred_box, pred_class in zip(pert_pred['boxes'], pert_pred['classes']):
                    if self._compute_iou(pred_box, gt_box) >= iou_threshold:
                        pert_detected = True
                        pert_class = pred_class
                        break
                
                if not pert_detected:
                    failed_detections += 1
                elif pert_class != gt_class:
                    misclassifications += 1
        
        asr = ((failed_detections + misclassifications) / (total_objects + 1e-10)) * 100
        
        return {
            'ASR': asr,
            'failed_detections': failed_detections,
            'misclassifications': misclassifications,
            'total_objects': total_objects,
        }
    
    def compute_risk_weighted_asr(
        self,
        clean_predictions: List[Dict],
        perturbed_predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict:
        total_objects = 0
        total_weighted_errors = 0
        
        for clean_pred, pert_pred, gt in zip(clean_predictions, perturbed_predictions, ground_truth):
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            total_objects += len(gt_boxes)
            
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                clean_detected = False
                clean_class = None
                
                for pred_box, pred_class in zip(clean_pred['boxes'], clean_pred['classes']):
                    if self._compute_iou(pred_box, gt_box) >= iou_threshold:
                        clean_detected = True
                        clean_class = pred_class
                        break
                
                if not clean_detected:
                    continue
                
                pert_detected = False
                pert_class = None
                
                for pred_box, pred_class in zip(pert_pred['boxes'], pert_pred['classes']):
                    if self._compute_iou(pred_box, gt_box) >= iou_threshold:
                        pert_detected = True
                        pert_class = pred_class
                        break
                
                if not pert_detected:
                    misclass_idx = int(gt_class)
                    weight = self.severity_matrix[misclass_idx, :].mean()
                    total_weighted_errors += weight
                elif pert_class != gt_class:
                    true_idx = int(gt_class)
                    pred_idx = int(pert_class)
                    weight = self.severity_matrix[true_idx, pred_idx]
                    total_weighted_errors += weight
        
        rw_asr = (total_weighted_errors / (total_objects + 1e-10)) * 100
        
        return {
            'RW-ASR': rw_asr,
            'total_weighted_errors': total_weighted_errors,
            'total_objects': total_objects,
        }
    
    def compute_critical_failure_rate(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> Dict:
        total_objects = 0
        critical_failures = 0
        
        critical_misclass_pairs = [
            ('stop_sign', 'speed_limit_55'),
            ('traffic_light_red', 'traffic_light_green'),
            ('traffic_light_green', 'traffic_light_red'),
        ]
        
        for pred, gt in zip(predictions, ground_truth):
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            total_objects += len(gt_boxes)
            
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                pred_class = None
                
                for pred_box, p_class in zip(pred['boxes'], pred['classes']):
                    if self._compute_iou(pred_box, gt_box) >= iou_threshold:
                        pred_class = p_class
                        break
                
                if pred_class is None:
                    continue
                
                gt_class_name = self.class_names[int(gt_class)]
                pred_class_name = self.class_names[int(pred_class)]
                
                if (gt_class_name, pred_class_name) in critical_misclass_pairs:
                    critical_failures += 1
        
        cfr = (critical_failures / (total_objects + 1e-10)) * 100
        
        return {
            'CFR': cfr,
            'critical_failures': critical_failures,
            'total_objects': total_objects,
        }
    
    def compute_stability_metrics(
        self,
        sequence_predictions: List[List[Dict]],
    ) -> Dict:
        all_confidences = []
        label_flips = 0
        total_tracks = 0
        
        for seq_preds in sequence_predictions:
            if len(seq_preds) < 2:
                continue
            
            tracked_objects = self._track_objects_in_sequence(seq_preds)
            
            for track in tracked_objects:
                if len(track) < 2:
                    continue
                
                total_tracks += 1
                
                confidences = [det['score'] for det in track]
                all_confidences.extend(confidences)
                
                classes = [det['class'] for det in track]
                for i in range(1, len(classes)):
                    if classes[i] != classes[i-1]:
                        label_flips += 1
        
        if len(all_confidences) > 0:
            confidence_mean = np.mean(all_confidences)
            confidence_std = np.std(all_confidences)
            confidence_volatility = confidence_std / (confidence_mean + 1e-10)
        else:
            confidence_mean = 0.0
            confidence_std = 0.0
            confidence_volatility = 0.0
        
        label_flip_rate = (label_flips / (total_tracks + 1e-10)) * 100
        
        stability_score = max(0, 1.0 - confidence_volatility - label_flip_rate / 100)
        
        return {
            'stability_score': stability_score,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'confidence_volatility': confidence_volatility,
            'label_flip_rate': label_flip_rate,
            'total_tracks': total_tracks,
        }
    
    def _compute_confusion_matrix(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float = 0.5,
    ) -> np.ndarray:
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        
        for pred, gt in zip(predictions, ground_truth):
            gt_boxes = gt['boxes']
            gt_classes = gt['classes']
            
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                pred_class = None
                max_iou = 0
                
                for pred_box, p_class in zip(pred['boxes'], pred['classes']):
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou > max_iou and iou >= iou_threshold:
                        max_iou = iou
                        pred_class = p_class
                
                if pred_class is not None:
                    conf_matrix[int(gt_class), int(pred_class)] += 1
        
        return conf_matrix
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-10)
        
        return iou
    
    def _track_objects_in_sequence(self, sequence_predictions: List[Dict]) -> List[List[Dict]]:
        tracks = []
        
        for frame_idx, pred in enumerate(sequence_predictions):
            boxes = pred['boxes']
            scores = pred['scores']
            classes = pred['classes']
            
            for box, score, cls in zip(boxes, scores, classes):
                matched = False
                
                for track in tracks:
                    if len(track) > 0 and track[-1]['frame'] == frame_idx - 1:
                        last_box = track[-1]['box']
                        iou = self._compute_iou(box, last_box)
                        
                        if iou > 0.3:
                            track.append({
                                'frame': frame_idx,
                                'box': box,
                                'score': score,
                                'class': cls,
                            })
                            matched = True
                            break
                
                if not matched:
                    tracks.append([{
                        'frame': frame_idx,
                        'box': box,
                        'score': score,
                        'class': cls,
                    }])
        
        return tracks