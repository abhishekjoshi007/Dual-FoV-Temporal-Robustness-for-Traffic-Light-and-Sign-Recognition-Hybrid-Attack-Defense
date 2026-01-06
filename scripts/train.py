import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DualFoVTrafficDataset, collate_fn
from data.transforms import get_train_transforms, get_val_transforms
from models.baseline_yolov8m import BaselineYOLOv8m
from models.unified_defense_stack import UnifiedDefenseStack
from models.temporal_voting import TemporalVoting
from attacks.natural_perturbations import NaturalPerturbationSuite
from evaluation.risk_weighted_metrics import RiskWeightedMetrics
from evaluation.odd_stratified_eval import ODDStratifiedEvaluator
from utils.logger import setup_logger, ExperimentLogger
from utils.visualization import Visualizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data_yaml(config, output_path):
    class_names = [
        'stop_sign', 'speed_limit_35', 'speed_limit_45', 'speed_limit_55',
        'traffic_light_red', 'traffic_light_green', 'traffic_light_yellow',
        'one_way', 'yield'
    ]
    
    data_yaml = {
        'path': config['data']['root_dir'],
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'nc': len(class_names),
        'names': class_names,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    return str(output_path)


def train_baseline(config, logger):
    logger.info("Training baseline YOLOv8m model")
    
    model = BaselineYOLOv8m(
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        pretrained=config['model']['pretrained']
    )
    
    data_yaml_path = create_data_yaml(config, 'data/dataset.yaml')
    
    results = model.train_model(
        data_yaml=data_yaml_path,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['lr'],
        device=config['training']['device'],
        workers=config['data']['num_workers'],
        project=config['logging']['log_dir'],
        name=config['experiment']['name'],
        patience=config['training']['patience'],
        save_period=config['training']['save_period'],
    )
    
    save_path = Path(config['logging']['save_dir']) / 'best.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    
    logger.info(f"Training completed. Model saved to {save_path}")
    
    return model


def train_with_natural_augmentation(config, logger):
    logger.info("Training with natural perturbation augmentation")
    
    model = BaselineYOLOv8m(
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        pretrained=config['model']['pretrained']
    )
    
    data_yaml_path = create_data_yaml(config, 'data/dataset.yaml')
    
    results = model.train_model(
        data_yaml=data_yaml_path,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['lr'],
        device=config['training']['device'],
        workers=config['data']['num_workers'],
        project=config['logging']['log_dir'],
        name=config['experiment']['name'],
        patience=config['training']['patience'],
        save_period=config['training']['save_period'],
    )
    
    save_path = Path(config['logging']['save_dir']) / 'best.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    
    logger.info(f"Training completed. Model saved to {save_path}")
    
    return model


def train_unified_defense(config, logger, exp_logger):
    logger.info("Training unified defense stack")
    
    exp_logger.log_hyperparameters(config)
    
    base_model_path = config['model'].get('base_model_checkpoint', None)
    if base_model_path and Path(base_model_path).exists():
        logger.info(f"Loading base model from {base_model_path}")
        base_model = BaselineYOLOv8m(
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size'],
            model_path=base_model_path
        )
    else:
        logger.info("Training base model from scratch")
        base_model = BaselineYOLOv8m(
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size'],
            pretrained=config['model']['pretrained']
        )
        
        data_yaml_path = create_data_yaml(config, 'data/dataset.yaml')
        base_model.train_model(
            data_yaml=data_yaml_path,
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            lr=config['training']['lr'],
            device=config['training']['device'],
            workers=config['data']['num_workers'],
            project=config['logging']['log_dir'],
            name='base_model',
            patience=config['training']['patience'],
        )
    
    defense_stack = UnifiedDefenseStack(
        base_model=base_model,
        bit_depth=config['defense']['feature_squeeze']['bit_depth'],
        median_kernel=config['defense']['feature_squeeze']['median_kernel'],
        temperature=config['defense']['temperature_scaling']['temperature'],
        entropy_threshold=config['defense']['entropy_gating']['entropy_threshold'],
        use_feature_squeeze=config['defense']['feature_squeeze']['enabled'],
        use_temperature_scaling=config['defense']['temperature_scaling']['enabled'],
        use_entropy_gating=config['defense']['entropy_gating']['enabled'],
    )
    
    save_path = Path(config['logging']['save_dir']) / 'unified_defense.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    defense_stack.save(str(save_path))
    
    logger.info(f"Unified defense stack saved to {save_path}")
    
    return defense_stack


def main():
    parser = argparse.ArgumentParser(description='Train traffic sign/light detection models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, default='baseline', 
                       choices=['baseline', 'natural', 'unified'],
                       help='Training mode')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['experiment']['seed'])
    
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name=config['experiment']['name'],
        log_file=str(log_dir / 'train.log'),
        level='INFO'
    )
    
    exp_logger = ExperimentLogger(
        experiment_name=config['experiment']['name'],
        log_dir=str(log_dir)
    )
    
    logger.info(f"Starting training: {config['experiment']['description']}")
    logger.info(f"Mode: {args.mode}")
    
    if args.mode == 'baseline':
        model = train_baseline(config, logger)
    elif args.mode == 'natural':
        model = train_with_natural_augmentation(config, logger)
    elif args.mode == 'unified':
        model = train_unified_defense(config, logger, exp_logger)
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()