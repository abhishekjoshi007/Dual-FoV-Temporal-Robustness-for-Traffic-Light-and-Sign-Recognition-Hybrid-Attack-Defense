import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = 'dual_fov_defense',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'dual_fov_defense') -> logging.Logger:
    return logging.getLogger(name)


class ExperimentLogger:
    
    def __init__(self, experiment_name: str, log_dir: str = 'logs'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'
        
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file),
            level=logging.INFO
        )
        
        self.metrics_history = []
    
    def log_hyperparameters(self, hparams: dict):
        self.logger.info("Hyperparameters:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        self.logger.info(f"Epoch {epoch}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
        
        metrics['epoch'] = epoch
        self.metrics_history.append(metrics)
    
    def log_evaluation(self, split: str, metrics: dict):
        self.logger.info(f"Evaluation on {split}:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_attack_results(self, attack_type: str, results: dict):
        self.logger.info(f"Attack Results - {attack_type}:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
    
    def save_metrics_history(self, filepath: Optional[str] = None):
        import json
        
        if filepath is None:
            filepath = self.log_dir / f'{self.experiment_name}_metrics.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics history saved to {filepath}")