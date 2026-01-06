import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaselineYOLOv8m(nn.Module):
    
    def __init__(
        self,
        num_classes: int = 9,
        img_size: int = 640,
        pretrained: bool = True,
        model_path: Optional[str] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        elif pretrained:
            self.model = YOLO('yolov8m.pt')
            logger.info("Loaded pretrained YOLOv8m")
        else:
            self.model = YOLO('yolov8m.yaml')
            logger.info("Initialized YOLOv8m from scratch")
    
    def forward(self, x: torch.Tensor, conf_threshold: float = 0.25) -> List[Dict]:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            results = self.model(x, verbose=False, conf=conf_threshold)
            parsed = self._parse_results(results)
            
            batched_results = []
            for i in range(B):
                batch_detections = parsed[i*T:(i+1)*T]
                batched_results.append(batch_detections)
            return batched_results
        else:
            results = self.model(x, verbose=False, conf=conf_threshold)
            return self._parse_results(results)
    
    def _parse_results(self, results) -> List[Dict]:
        parsed = []
        for result in results:
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                detection = {
                    'boxes': boxes.xyxy.cpu().numpy(),
                    'scores': boxes.conf.cpu().numpy(),
                    'classes': boxes.cls.cpu().numpy().astype(int),
                }
            else:
                detection = {
                    'boxes': np.array([]).reshape(0, 4),
                    'scores': np.array([]),
                    'classes': np.array([], dtype=int),
                }
            
            parsed.append(detection)
        return parsed
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25) -> List[Dict]:
        with torch.no_grad():
            return self.forward(x, conf_threshold)
    
    def train_model(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-4,
        device: int = 0,
        workers: int = 8,
        project: str = 'runs/detect',
        name: str = 'train',
        patience: int = 10,
        save_period: int = -1,
        resume: bool = False,
    ):
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=lr,
            imgsz=self.img_size,
            patience=patience,
            save=True,
            device=device,
            workers=workers,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            save_period=save_period,
            resume=resume,
        )
        return results
    
    def validate(
        self,
        data_yaml: str,
        batch_size: int = 32,
        device: int = 0,
        workers: int = 8,
    ):
        results = self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=self.img_size,
            device=device,
            workers=workers,
            verbose=True,
        )
        return results
    
    def export_onnx(self, output_path: str):
        self.model.export(format='onnx', dynamic=True, simplify=True)
        logger.info(f"Model exported to ONNX format at {output_path}")
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        self.model = YOLO(path)
        logger.info(f"Model loaded from {path}")
    
    def get_model_info(self):
        return {
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad),
        }
    
    def freeze_backbone(self):
        for name, param in self.model.model.named_parameters():
            if 'model.0' in name or 'model.1' in name or 'model.2' in name:
                param.requires_grad = False
        logger.info("Backbone frozen")
    
    def unfreeze_all(self):
        for param in self.model.model.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")