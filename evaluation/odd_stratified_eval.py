import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from .risk_weighted_metrics import RiskWeightedMetrics

logger = logging.getLogger(__name__)


class ODDStratifiedEvaluator:
    
    def __init__(
        self,
        odd_list: List[str] = ['highway', 'night', 'rainy', 'urban'],
        metrics_calculator: Optional[RiskWeightedMetrics] = None,
    ):
        self.odd_list = odd_list
        self.metrics_calculator = metrics_calculator if metrics_calculator else RiskWeightedMetrics()
        
        self.odd_predictions = {odd: [] for odd in odd_list}
        self.odd_ground_truth = {odd: [] for odd in odd_list}
        self.odd_metadata = {odd: [] for odd in odd_list}
        
        logger.info(f"ODDStratifiedEvaluator initialized with ODDs: {odd_list}")
    
    def add_batch(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        odd_labels: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        for i, (pred, gt, odd) in enumerate(zip(predictions, ground_truth, odd_labels)):
            if odd not in self.odd_list:
                logger.warning(f"Unknown ODD: {odd}")
                continue
            
            self.odd_predictions[odd].append(pred)
            self.odd_ground_truth[odd].append(gt)
            
            if metadata:
                self.odd_metadata[odd].append(metadata[i])
    
    def evaluate_all(self, iou_threshold: float = 0.5) -> Dict:
        results = {}
        
        for odd in self.odd_list:
            if len(self.odd_predictions[odd]) == 0:
                logger.warning(f"No predictions for ODD: {odd}")
                continue
            
            odd_results = self._evaluate_odd(
                odd,
                self.odd_predictions[odd],
                self.odd_ground_truth[odd],
                iou_threshold
            )
            
            results[odd] = odd_results
        
        overall_results = self._aggregate_results(results)
        results['overall'] = overall_results
        
        return results
    
    def _evaluate_odd(
        self,
        odd: str,
        predictions: List[Dict],
        ground_truth: List[Dict],
        iou_threshold: float,
    ) -> Dict:
        map_results = self.metrics_calculator.compute_map(predictions, ground_truth, iou_threshold)
        
        rw_map_results = self.metrics_calculator.compute_risk_weighted_map(
            predictions, ground_truth, iou_threshold
        )
        
        cfr_results = self.metrics_calculator.compute_critical_failure_rate(
            predictions, ground_truth, iou_threshold
        )
        
        return {
            'mAP': map_results['mAP'],
            'per_class_AP': map_results['per_class_AP'],
            'RW-mAP': rw_map_results['RW-mAP'],
            'per_class_RW_AP': rw_map_results['per_class_RW_AP'],
            'CFR': cfr_results['CFR'],
            'num_samples': len(predictions),
            'confusion_matrix': rw_map_results['confusion_matrix'],
        }
    
    def _aggregate_results(self, odd_results: Dict) -> Dict:
        all_maps = []
        all_rw_maps = []
        all_cfrs = []
        total_samples = 0
        
        per_class_ap_sum = defaultdict(float)
        per_class_rw_ap_sum = defaultdict(float)
        per_class_counts = defaultdict(int)
        
        for odd, results in odd_results.items():
            if odd == 'overall':
                continue
            
            all_maps.append(results['mAP'])
            all_rw_maps.append(results['RW-mAP'])
            all_cfrs.append(results['CFR'])
            total_samples += results['num_samples']
            
            for class_name, ap in results['per_class_AP'].items():
                per_class_ap_sum[class_name] += ap
                per_class_counts[class_name] += 1
            
            for class_name, rw_ap in results['per_class_RW_AP'].items():
                per_class_rw_ap_sum[class_name] += rw_ap
        
        overall_map = np.mean(all_maps) if all_maps else 0.0
        overall_rw_map = np.mean(all_rw_maps) if all_rw_maps else 0.0
        overall_cfr = np.mean(all_cfrs) if all_cfrs else 0.0
        
        per_class_ap_avg = {
            class_name: per_class_ap_sum[class_name] / per_class_counts[class_name]
            for class_name in per_class_ap_sum.keys()
        }
        
        per_class_rw_ap_avg = {
            class_name: per_class_rw_ap_sum[class_name] / per_class_counts[class_name]
            for class_name in per_class_rw_ap_sum.keys()
        }
        
        return {
            'mAP': overall_map,
            'per_class_AP': per_class_ap_avg,
            'RW-mAP': overall_rw_map,
            'per_class_RW_AP': per_class_rw_ap_avg,
            'CFR': overall_cfr,
            'num_samples': total_samples,
        }
    
    def evaluate_with_perturbations(
        self,
        clean_predictions: Dict[str, List[Dict]],
        perturbed_predictions: Dict[str, List[Dict]],
        ground_truth: Dict[str, List[Dict]],
        perturbation_types: List[str],
        iou_threshold: float = 0.5,
    ) -> Dict:
        results = {}
        
        for odd in self.odd_list:
            if odd not in clean_predictions or len(clean_predictions[odd]) == 0:
                continue
            
            odd_results = {}
            
            clean_map = self.metrics_calculator.compute_map(
                clean_predictions[odd],
                ground_truth[odd],
                iou_threshold
            )
            odd_results['clean'] = {'mAP': clean_map['mAP']}
            
            for pert_type in perturbation_types:
                if pert_type not in perturbed_predictions:
                    continue
                
                pert_map = self.metrics_calculator.compute_map(
                    perturbed_predictions[pert_type][odd],
                    ground_truth[odd],
                    iou_threshold
                )
                
                asr = self.metrics_calculator.compute_attack_success_rate(
                    clean_predictions[odd],
                    perturbed_predictions[pert_type][odd],
                    ground_truth[odd],
                    iou_threshold
                )
                
                odd_results[pert_type] = {
                    'mAP': pert_map['mAP'],
                    'ASR': asr['ASR'],
                    'mAP_drop': clean_map['mAP'] - pert_map['mAP'],
                }
            
            results[odd] = odd_results
        
        return results
    
    def compute_odd_difficulty_ranking(self) -> List[Tuple[str, float]]:
        difficulty_scores = []
        
        for odd in self.odd_list:
            if len(self.odd_predictions[odd]) == 0:
                continue
            
            map_results = self.metrics_calculator.compute_map(
                self.odd_predictions[odd],
                self.odd_ground_truth[odd]
            )
            
            cfr_results = self.metrics_calculator.compute_critical_failure_rate(
                self.odd_predictions[odd],
                self.odd_ground_truth[odd]
            )
            
            difficulty = (1.0 - map_results['mAP']) + (cfr_results['CFR'] / 100.0)
            
            difficulty_scores.append((odd, difficulty))
        
        difficulty_scores.sort(key=lambda x: x[1], reverse=True)
        
        return difficulty_scores
    
    def analyze_failure_modes_by_odd(self) -> Dict:
        failure_analysis = {}
        
        for odd in self.odd_list:
            if len(self.odd_predictions[odd]) == 0:
                continue
            
            conf_matrix = self.metrics_calculator._compute_confusion_matrix(
                self.odd_predictions[odd],
                self.odd_ground_truth[odd]
            )
            
            total_errors = conf_matrix.sum() - np.diag(conf_matrix).sum()
            
            top_errors = []
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    if i != j and conf_matrix[i, j] > 0:
                        top_errors.append({
                            'true_class': self.metrics_calculator.class_names[i],
                            'pred_class': self.metrics_calculator.class_names[j],
                            'count': int(conf_matrix[i, j]),
                            'percentage': float(conf_matrix[i, j] / (total_errors + 1e-10) * 100),
                        })
            
            top_errors.sort(key=lambda x: x['count'], reverse=True)
            
            failure_analysis[odd] = {
                'total_errors': int(total_errors),
                'top_errors': top_errors[:5],
                'confusion_matrix': conf_matrix.tolist(),
            }
        
        return failure_analysis
    
    def reset(self):
        self.odd_predictions = {odd: [] for odd in self.odd_list}
        self.odd_ground_truth = {odd: [] for odd in self.odd_list}
        self.odd_metadata = {odd: [] for odd in self.odd_list}