import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_colors = {
            'stop_sign': (255, 0, 0),
            'speed_limit_35': (0, 255, 0),
            'speed_limit_45': (0, 255, 100),
            'speed_limit_55': (0, 255, 200),
            'traffic_light_red': (255, 0, 0),
            'traffic_light_green': (0, 255, 0),
            'traffic_light_yellow': (255, 255, 0),
            'one_way': (255, 0, 255),
            'yield': (255, 128, 0),
        }
    
    def plot_odd_performance(
        self,
        odd_results: Dict[str, Dict],
        metric: str = 'mAP',
        save_name: Optional[str] = None,
    ):
        odds = list(odd_results.keys())
        values = [odd_results[odd][metric] for odd in odds]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(odds, values, color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Operational Design Domain', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} across ODDs', fontsize=14)
        ax.set_ylim(0, 1.0 if 'mAP' in metric else max(values) * 1.2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_perturbation_impact(
        self,
        perturbation_results: Dict[str, float],
        baseline_map: float,
        save_name: Optional[str] = None,
    ):
        perturbations = list(perturbation_results.keys())
        map_drops = [baseline_map - perturbation_results[p] for p in perturbations]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(perturbations, map_drops, color='coral', alpha=0.7)
        
        ax.set_xlabel('mAP Drop (%)', fontsize=12)
        ax.set_ylabel('Perturbation Type', fontsize=12)
        ax.set_title('Performance Degradation under Natural Perturbations', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}%',
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        save_name: Optional[str] = None,
    ):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, int(confusion_matrix[i, j]),
                             ha="center", va="center",
                             color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
        
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_training_curves(
        self,
        metrics_history: List[Dict],
        metrics: List[str] = ['mAP', 'loss'],
        save_name: Optional[str] = None,
    ):
        epochs = [m['epoch'] for m in metrics_history]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = [m.get(metric, 0) for m in metrics_history]
            
            axes[idx].plot(epochs, values, marker='o', linewidth=2, markersize=4)
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(f'{metric} over Training', fontsize=14)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_ablation_study(
        self,
        ablation_results: Dict[str, float],
        baseline_score: float,
        save_name: Optional[str] = None,
    ):
        components = list(ablation_results.keys())
        improvements = [ablation_results[c] - baseline_score for c in components]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(components, improvements, color='teal', alpha=0.7)
        
        ax.set_xlabel('Defense Component', fontsize=12)
        ax.set_ylabel('mAP Improvement (%)', fontsize=12)
        ax.set_title('Ablation Study: Component Contributions', fontsize=14)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.2f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_temporal_voting_effect(
        self,
        frame_by_frame: List[float],
        with_voting: List[float],
        save_name: Optional[str] = None,
    ):
        frames = list(range(len(frame_by_frame)))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(frames, frame_by_frame, label='Frame-by-Frame', 
               marker='o', linewidth=2, markersize=3, alpha=0.7)
        ax.plot(frames, with_voting, label='With Temporal Voting',
               marker='s', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Temporal Voting Recovery Effect', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: Dict,
        class_names: List[str],
        confidence_threshold: float = 0.5,
    ) -> np.ndarray:
        img_viz = image.copy()
        
        if img_viz.dtype == np.float32 or img_viz.dtype == np.float64:
            img_viz = (img_viz * 255).astype(np.uint8)
        
        boxes = detections['boxes']
        scores = detections['scores']
        classes = detections['classes']
        
        for box, score, cls in zip(boxes, scores, classes):
            if score < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)]
            color = self.class_colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(img_viz, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img_viz, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_viz
    
    def visualize_dual_fov(
        self,
        mid_range_img: np.ndarray,
        long_range_img: np.ndarray,
        mid_detections: Dict,
        long_detections: Dict,
        class_names: List[str],
        save_name: Optional[str] = None,
    ):
        mid_viz = self.draw_detections(mid_range_img, mid_detections, class_names)
        long_viz = self.draw_detections(long_range_img, long_detections, class_names)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        axes[0].imshow(mid_viz)
        axes[0].set_title('Mid-Range Camera', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(long_viz)
        axes[1].set_title('Long-Range Camera', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_attack_comparison(
        self,
        attack_results: Dict[str, Dict[str, float]],
        save_name: Optional[str] = None,
    ):
        attacks = list(attack_results.keys())
        metrics = list(attack_results[attacks[0]].keys())
        
        x = np.arange(len(attacks))
        width = 0.8 / len(metrics)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics):
            values = [attack_results[attack][metric] for attack in attacks]
            offset = width * (i - len(metrics)/2 + 0.5)
            ax.bar(x + offset, values, width, label=metric, alpha=0.8)
        
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Attack Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_per_class_performance(
        self,
        per_class_results: Dict[str, float],
        save_name: Optional[str] = None,
    ):
        classes = list(per_class_results.keys())
        values = list(per_class_results.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(classes, values, color='skyblue', alpha=0.8)
        
        ax.set_xlabel('Average Precision', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title('Per-Class Performance', fontsize=14)
        ax.set_xlim(0, 1.0)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}',
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        plt.close()