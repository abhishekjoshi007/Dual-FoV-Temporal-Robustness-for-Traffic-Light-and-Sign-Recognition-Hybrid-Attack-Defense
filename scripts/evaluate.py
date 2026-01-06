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
import json

from data.dataset import DualFoVTrafficDataset, collate_fn
from data.transforms import get_val_transforms
from models.baseline_yolov8m import BaselineYOLOv8m
from models.unified_defense_stack import UnifiedDefenseStack
from models.temporal_voting import TemporalVoting
from attacks.natural_perturbations import NaturalPerturbationSuite
from attacks.hybrid_attacks import HybridAttackSuite
from evaluation.risk_weighted_metrics import RiskWeightedMetrics
from evaluation.odd_stratified_eval import ODDStratifiedEvaluator
from evaluation.statistical_tests import StatisticalTester
from utils.logger import setup_logger
from utils.visualization import Visualizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(config, model_path):
    logger = setup_logger()
    
    if 'unified' in config['experiment']['name'].lower():
        logger.info("Loading unified defense stack")
        
        base_model = BaselineYOLOv8m(
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size'],
            pretrained=False
        )
        
        defense_stack = UnifiedDefenseStack(
            base_model=base_model,
            bit_depth=config['defense']['feature_squeeze']['bit_depth'],
            median_kernel=config['defense']['feature_squeeze']['median_kernel'],
            temperature=config['defense']['temperature_scaling']['temperature'],
            entropy_threshold=config['defense']['entropy_gating']['entropy_threshold'],
        )
        
        if Path(model_path).exists():
            defense_stack.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        return defense_stack
    else:
        logger.info("Loading baseline model")
        model = BaselineYOLOv8m(
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size'],
            model_path=model_path if Path(model_path).exists() else None
        )
        return model


def evaluate_clean(model, dataloader, config, device):
    logger = setup_logger()
    logger.info("Evaluating on clean data")
    
    metrics_calc = RiskWeightedMetrics()
    odd_evaluator = ODDStratifiedEvaluator()
    
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_odds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mid_range = batch['mid_range'].to(device)
            long_range = batch['long_range'].to(device)
            annotations = batch['annotations']
            odds = batch['odd']
            
            B, T, C, H, W = mid_range.shape
            
            if isinstance(model, UnifiedDefenseStack):
                for t in range(T):
                    output = model(mid_range[:, t], long_range[:, t])
                    predictions = output['selected_results']
                    
                    for b in range(B):
                        gt_boxes = annotations[b][t]['boxes']
                        
                        if len(gt_boxes) > 0:
                            all_predictions.append(predictions[b])
                            all_ground_truth.append({
                                'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                'classes': gt_boxes[:, 0].astype(int),
                            })
                            all_odds.append(odds[b])
            else:
                mid_range_flat = mid_range.view(B * T, C, H, W)
                predictions = model.predict(mid_range_flat)
                
                for b in range(B):
                    for t in range(T):
                        idx = b * T + t
                        gt_boxes = annotations[b][t]['boxes']
                        
                        if len(gt_boxes) > 0:
                            all_predictions.append(predictions[idx])
                            all_ground_truth.append({
                                'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                'classes': gt_boxes[:, 0].astype(int),
                            })
                            all_odds.append(odds[b])
    
    map_results = metrics_calc.compute_map(all_predictions, all_ground_truth)
    rw_map_results = metrics_calc.compute_risk_weighted_map(all_predictions, all_ground_truth)
    cfr_results = metrics_calc.compute_critical_failure_rate(all_predictions, all_ground_truth)
    
    for pred, gt, odd in zip(all_predictions, all_ground_truth, all_odds):
        odd_evaluator.add_batch([pred], [gt], [odd])
    
    odd_results = odd_evaluator.evaluate_all()
    
    results = {
        'mAP': map_results['mAP'],
        'per_class_AP': map_results['per_class_AP'],
        'RW-mAP': rw_map_results['RW-mAP'],
        'CFR': cfr_results['CFR'],
        'odd_results': odd_results,
    }
    
    return results


def evaluate_with_perturbations(model, dataloader, config, device):
    logger = setup_logger()
    logger.info("Evaluating with natural perturbations")
    
    perturbation_suite = NaturalPerturbationSuite(
        perturbation_types=config['evaluation']['perturbation_suite']['types']
    )
    
    metrics_calc = RiskWeightedMetrics()
    results = {}
    
    for pert_type in perturbation_suite.perturbation_types:
        logger.info(f"Testing perturbation: {pert_type}")
        
        all_clean_pred = []
        all_pert_pred = []
        all_gt = []
        
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Perturbation: {pert_type}"):
                mid_range = batch['mid_range'].to(device)
                annotations = batch['annotations']
                
                B, T, C, H, W = mid_range.shape
                
                mid_range_np = mid_range.cpu().numpy()
                perturbed_np = np.zeros_like(mid_range_np)
                
                for b in range(B):
                    for t in range(T):
                        frame = (mid_range_np[b, t].transpose(1, 2, 0) * 255).astype(np.uint8)
                        pert_frame = perturbation_suite.apply_perturbation(frame, pert_type)
                        perturbed_np[b, t] = pert_frame.transpose(2, 0, 1) / 255.0
                
                perturbed = torch.from_numpy(perturbed_np).float().to(device)
                
                if isinstance(model, UnifiedDefenseStack):
                    for t in range(T):
                        clean_out = model(mid_range[:, t], mid_range[:, t])
                        pert_out = model(perturbed[:, t], perturbed[:, t])
                        
                        for b in range(B):
                            gt_boxes = annotations[b][t]['boxes']
                            if len(gt_boxes) > 0:
                                all_clean_pred.append(clean_out['selected_results'][b])
                                all_pert_pred.append(pert_out['selected_results'][b])
                                all_gt.append({
                                    'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                    'classes': gt_boxes[:, 0].astype(int),
                                })
                else:
                    clean_pred = model.predict(mid_range.view(B*T, C, H, W))
                    pert_pred = model.predict(perturbed.view(B*T, C, H, W))
                    
                    for b in range(B):
                        for t in range(T):
                            idx = b * T + t
                            gt_boxes = annotations[b][t]['boxes']
                            if len(gt_boxes) > 0:
                                all_clean_pred.append(clean_pred[idx])
                                all_pert_pred.append(pert_pred[idx])
                                all_gt.append({
                                    'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                    'classes': gt_boxes[:, 0].astype(int),
                                })
        
        asr = metrics_calc.compute_attack_success_rate(all_clean_pred, all_pert_pred, all_gt)
        map_results = metrics_calc.compute_map(all_pert_pred, all_gt)
        
        results[pert_type] = {
            'ASR': asr['ASR'],
            'mAP': map_results['mAP'],
        }
    
    return results


def evaluate_with_attacks(model, dataloader, config, device):
    logger = setup_logger()
    logger.info("Evaluating with adversarial attacks")
    
    attack_suite = HybridAttackSuite(
        epsilon=config['evaluation']['attack_suite']['pgd']['epsilon'],
        alpha=config['evaluation']['attack_suite']['pgd']['alpha'],
        num_iter=config['evaluation']['attack_suite']['pgd']['num_iter'],
    )
    
    metrics_calc = RiskWeightedMetrics()
    results = {}
    
    logger.info("This is a placeholder for attack evaluation")
    logger.info("Full implementation requires white-box model access")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate traffic sign/light detection models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--eval_mode', type=str, default='clean',
                       choices=['clean', 'perturbations', 'attacks', 'all'])
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['experiment']['seed'])
    
    logger = setup_logger(
        name=f"{config['experiment']['name']}_eval",
        log_file=f"{config['logging']['log_dir']}/eval_{args.split}.log"
    )
    
    logger.info(f"Evaluating: {config['experiment']['description']}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Evaluation mode: {args.eval_mode}")
    
    device = torch.device(f"cuda:{config['training']['device']}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = load_model(config, args.model_path)
    model = model.to(device)
    model.eval()
    
    transform = get_val_transforms(img_size=(config['model']['img_size'], config['model']['img_size']))
    
    dataset = DualFoVTrafficDataset(
        root_dir=config['data']['root_dir'],
        split=args.split,
        odd_list=config['data']['odd_list'],
        sequence_length=config['data']['sequence_length'],
        cameras=config['data']['cameras'],
        transform=transform,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
    )
    
    results = {}
    
    if args.eval_mode in ['clean', 'all']:
        clean_results = evaluate_clean(model, dataloader, config, device)
        results['clean'] = clean_results
        logger.info(f"Clean evaluation - mAP: {clean_results['mAP']:.4f}")
    
    if args.eval_mode in ['perturbations', 'all']:
        pert_results = evaluate_with_perturbations(model, dataloader, config, device)
        results['perturbations'] = pert_results
        for pert, metrics in pert_results.items():
            logger.info(f"{pert} - ASR: {metrics['ASR']:.2f}%, mAP: {metrics['mAP']:.4f}")
    
    if args.eval_mode in ['attacks', 'all']:
        attack_results = evaluate_with_attacks(model, dataloader, config, device)
        results['attacks'] = attack_results
    
    output_dir = Path(config['logging']['log_dir']) / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'eval_{args.split}_{args.eval_mode}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()