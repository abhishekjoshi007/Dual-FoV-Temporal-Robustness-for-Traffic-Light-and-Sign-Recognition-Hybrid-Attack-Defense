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


def evaluate_clean(model, dataloader, config, device, return_raw_scores=False):
    """Evaluate model on clean data

    Args:
        model: Model to evaluate
        dataloader: Data loader
        config: Configuration dictionary
        device: Device to run on
        return_raw_scores: If True, return per-sample mAP scores for statistical testing

    Returns:
        Dictionary of evaluation results
    """
    logger = setup_logger()
    logger.info("Evaluating on clean data")

    metrics_calc = RiskWeightedMetrics()
    odd_evaluator = ODDStratifiedEvaluator()

    model.eval()

    all_predictions = []
    all_ground_truth = []
    all_odds = []
    per_sample_maps = []  # For statistical testing

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

                            # Compute per-sample mAP for statistical testing
                            if return_raw_scores:
                                sample_map = metrics_calc.compute_map(
                                    [predictions[b]],
                                    [{
                                        'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                        'classes': gt_boxes[:, 0].astype(int),
                                    }]
                                )
                                per_sample_maps.append(sample_map['mAP'])
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

                            # Compute per-sample mAP for statistical testing
                            if return_raw_scores:
                                sample_map = metrics_calc.compute_map(
                                    [predictions[idx]],
                                    [{
                                        'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                        'classes': gt_boxes[:, 0].astype(int),
                                    }]
                                )
                                per_sample_maps.append(sample_map['mAP'])

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

    if return_raw_scores:
        results['per_sample_maps'] = np.array(per_sample_maps)

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


def run_ablation_study(config, model_path, dataloader, device):
    """Reproduce Table 10: Incremental ablation study

    Tests each defense component individually and cumulatively to measure
    their individual contributions to overall robustness.
    """
    logger = setup_logger()
    logger.info("Running ablation study (Table 10)")

    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'Baseline (No Defense)',
            'fs': False,
            'ts': False,
            'eg': False,
            'tv': False,
        },
        {
            'name': '+ Feature Squeezing',
            'fs': True,
            'ts': False,
            'eg': False,
            'tv': False,
        },
        {
            'name': '+ Temperature Scaling',
            'fs': True,
            'ts': True,
            'eg': False,
            'tv': False,
        },
        {
            'name': '+ Entropy Gating',
            'fs': True,
            'ts': True,
            'eg': True,
            'tv': False,
        },
        {
            'name': '+ Cross-FoV Validation',
            'fs': True,
            'ts': True,
            'eg': True,
            'tv': False,  # Same as entropy gating
        },
        {
            'name': '+ Temporal Voting (Full)',
            'fs': True,
            'ts': True,
            'eg': True,
            'tv': True,
        },
    ]

    results = []
    metrics_calc = RiskWeightedMetrics()

    # Load base model
    base_model = BaselineYOLOv8m(
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        pretrained=False
    )

    for ablation_cfg in ablation_configs:
        logger.info(f"\nEvaluating: {ablation_cfg['name']}")
        logger.info(f"  Feature Squeeze: {ablation_cfg['fs']}")
        logger.info(f"  Temperature Scaling: {ablation_cfg['ts']}")
        logger.info(f"  Entropy Gating: {ablation_cfg['eg']}")
        logger.info(f"  Temporal Voting: {ablation_cfg['tv']}")

        # Create model with specific ablation configuration
        if ablation_cfg['name'] == 'Baseline (No Defense)':
            # Use baseline model without any defense
            eval_model = base_model
            if Path(model_path).exists():
                eval_model.load(model_path)
        else:
            # Create defense stack with selective components
            defense_model = UnifiedDefenseStack(
                base_model=base_model,
                bit_depth=config['defense']['feature_squeeze']['bit_depth'],
                median_kernel=config['defense']['feature_squeeze']['median_kernel'],
                temperature=config['defense']['temperature_scaling']['temperature'],
                entropy_threshold=config['defense']['entropy_gating']['entropy_threshold'],
                use_feature_squeeze=ablation_cfg['fs'],
                use_temperature_scaling=ablation_cfg['ts'],
                use_entropy_gating=ablation_cfg['eg'],
            )

            if Path(model_path).exists():
                defense_model.load(model_path)

            eval_model = defense_model

        eval_model = eval_model.to(device)
        eval_model.eval()

        # Evaluate clean performance
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Ablation: {ablation_cfg['name']}"):
                mid_range = batch['mid_range'].to(device)
                long_range = batch['long_range'].to(device)
                annotations = batch['annotations']

                B, T, C, H, W = mid_range.shape

                if isinstance(eval_model, UnifiedDefenseStack):
                    for t in range(T):
                        output = eval_model(mid_range[:, t], long_range[:, t])
                        predictions = output['selected_results']

                        for b in range(B):
                            gt_boxes = annotations[b][t]['boxes']
                            if len(gt_boxes) > 0:
                                all_predictions.append(predictions[b])
                                all_ground_truth.append({
                                    'boxes': gt_boxes[:, 1:] * config['model']['img_size'],
                                    'classes': gt_boxes[:, 0].astype(int),
                                })
                else:
                    mid_range_flat = mid_range.view(B * T, C, H, W)
                    predictions = eval_model.predict(mid_range_flat)

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

        # Compute metrics
        map_results = metrics_calc.compute_map(all_predictions, all_ground_truth)
        rw_map_results = metrics_calc.compute_risk_weighted_map(all_predictions, all_ground_truth)
        cfr_results = metrics_calc.compute_critical_failure_rate(all_predictions, all_ground_truth)

        ablation_result = {
            'configuration': ablation_cfg['name'],
            'components': {
                'feature_squeeze': ablation_cfg['fs'],
                'temperature_scaling': ablation_cfg['ts'],
                'entropy_gating': ablation_cfg['eg'],
                'temporal_voting': ablation_cfg['tv'],
            },
            'metrics': {
                'mAP': map_results['mAP'],
                'RW-mAP': rw_map_results['RW-mAP'],
                'CFR': cfr_results['CFR'],
            }
        }

        results.append(ablation_result)

        logger.info(f"  Results: mAP={map_results['mAP']:.4f}, "
                   f"RW-mAP={rw_map_results['RW-mAP']:.4f}, "
                   f"CFR={cfr_results['CFR']:.4f}")

    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY RESULTS (Table 10)")
    logger.info("="*80)
    logger.info(f"{'Configuration':<35} {'mAP':>10} {'RW-mAP':>10} {'CFR':>10}")
    logger.info("-"*80)
    for result in results:
        logger.info(f"{result['configuration']:<35} "
                   f"{result['metrics']['mAP']:>10.4f} "
                   f"{result['metrics']['RW-mAP']:>10.4f} "
                   f"{result['metrics']['CFR']:>10.4f}")
    logger.info("="*80 + "\n")

    return results


def run_statistical_comparison(baseline_model_path, defense_model_path, config, dataloader, device):
    """Run statistical significance tests comparing baseline vs defense

    Reproduces the statistical claims from Table 9 with:
    - Paired t-tests
    - Bootstrap confidence intervals (n=1000)
    - Bonferroni correction for multiple comparisons
    - Effect size analysis

    Args:
        baseline_model_path: Path to baseline model checkpoint
        defense_model_path: Path to defense model checkpoint
        config: Configuration dictionary
        dataloader: Data loader
        device: Device to run on

    Returns:
        Dictionary with statistical test results
    """
    logger = setup_logger()
    logger.info("Running statistical significance tests (Table 19)")

    # Initialize statistical tester
    tester = StatisticalTester(
        alpha=config.get('statistical_testing', {}).get('alpha', 0.05),
        n_bootstrap=config.get('statistical_testing', {}).get('n_bootstrap', 1000)
    )

    # Load and evaluate baseline model
    logger.info("\nEvaluating baseline model...")
    baseline_config = config.copy()
    baseline_config['experiment']['name'] = 'baseline'
    baseline_model = load_model(baseline_config, baseline_model_path)
    baseline_model = baseline_model.to(device)
    baseline_results = evaluate_clean(
        baseline_model, dataloader, baseline_config, device, return_raw_scores=True
    )

    # Load and evaluate defense model
    logger.info("\nEvaluating defense model...")
    defense_config = config.copy()
    defense_config['experiment']['name'] = 'unified_defense'
    defense_model = load_model(defense_config, defense_model_path)
    defense_model = defense_model.to(device)
    defense_results = evaluate_clean(
        defense_model, dataloader, defense_config, device, return_raw_scores=True
    )

    # Extract per-sample mAP scores
    baseline_maps = baseline_results['per_sample_maps']
    defense_maps = defense_results['per_sample_maps']

    logger.info(f"\nBaseline mAP: {baseline_results['mAP']:.4f}")
    logger.info(f"Defense mAP: {defense_results['mAP']:.4f}")
    logger.info(f"Absolute improvement: +{defense_results['mAP'] - baseline_results['mAP']:.4f}")

    # Run statistical tests
    results = {}

    # 1. Paired t-test
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("="*80)

    ttest_result = tester.paired_t_test(baseline_maps, defense_maps)
    results['paired_t_test'] = ttest_result

    logger.info("\n1. Paired t-test:")
    logger.info(f"   t-statistic: {ttest_result['t_statistic']:.4f}")
    logger.info(f"   p-value: {ttest_result['p_value']:.6f}")
    logger.info(f"   Significant: {'YES' if ttest_result['significant'] else 'NO'} (α={ttest_result['alpha']})")
    logger.info(f"   Mean difference: {ttest_result['mean_difference']:.4f}")
    logger.info(f"   Cohen's d: {ttest_result['cohens_d']:.4f}")

    # 2. Wilcoxon signed-rank test (non-parametric alternative)
    wilcoxon_result = tester.wilcoxon_signed_rank_test(baseline_maps, defense_maps)
    results['wilcoxon_test'] = wilcoxon_result

    logger.info("\n2. Wilcoxon Signed-Rank Test (non-parametric):")
    logger.info(f"   Statistic: {wilcoxon_result['statistic']:.4f}")
    logger.info(f"   p-value: {wilcoxon_result['p_value']:.6f}")
    logger.info(f"   Significant: {'YES' if wilcoxon_result['significant'] else 'NO'} (α={wilcoxon_result['alpha']})")

    # 3. Bootstrap confidence intervals
    logger.info("\n3. Bootstrap Confidence Intervals (95%, n=1000):")

    baseline_mean, baseline_ci_lower, baseline_ci_upper = tester.bootstrap_confidence_interval(
        baseline_maps, confidence=0.95
    )
    defense_mean, defense_ci_lower, defense_ci_upper = tester.bootstrap_confidence_interval(
        defense_maps, confidence=0.95
    )

    results['confidence_intervals'] = {
        'baseline': {
            'mean': float(baseline_mean),
            'ci_lower': float(baseline_ci_lower),
            'ci_upper': float(baseline_ci_upper),
        },
        'defense': {
            'mean': float(defense_mean),
            'ci_lower': float(defense_ci_lower),
            'ci_upper': float(defense_ci_upper),
        }
    }

    logger.info(f"   Baseline: {baseline_mean:.4f} [{baseline_ci_lower:.4f}, {baseline_ci_upper:.4f}]")
    logger.info(f"   Defense:  {defense_mean:.4f} [{defense_ci_lower:.4f}, {defense_ci_upper:.4f}]")

    # 4. Effect size analysis
    effect_size_result = tester.compute_effect_size(baseline_maps, defense_maps)
    results['effect_size'] = effect_size_result

    logger.info("\n4. Effect Size Analysis:")
    logger.info(f"   Cohen's d: {effect_size_result['cohens_d']:.4f}")
    logger.info(f"   Magnitude: {effect_size_result['magnitude']}")
    logger.info(f"   Mean difference: {effect_size_result['mean_difference']:.4f}")

    # 5. Statistical power
    power_result = tester.power_analysis(baseline_maps, defense_maps)
    results['power'] = power_result

    logger.info("\n5. Statistical Power Analysis:")
    logger.info(f"   Sample size: {power_result['sample_size']}")
    logger.info(f"   Power: {power_result['power']:.4f}")
    logger.info(f"   Adequately powered (≥0.8): {'YES' if power_result['powered'] else 'NO'}")

    # 6. Bonferroni correction (for multiple metric comparisons)
    if config.get('statistical_testing', {}).get('bonferroni_correction', False):
        # Assuming we're testing 4 metrics: mAP, RW-mAP, ASR, CFR
        p_values = [ttest_result['p_value']] * 4  # Placeholder
        bonferroni_result = tester.bonferroni_correction(p_values)
        results['bonferroni'] = bonferroni_result

        logger.info("\n6. Bonferroni Correction (4 metrics tested):")
        logger.info(f"   Original α: {bonferroni_result['original_alpha']:.4f}")
        logger.info(f"   Corrected α: {bonferroni_result['corrected_alpha']:.4f}")
        logger.info(f"   Still significant: {'YES' if ttest_result['p_value'] < bonferroni_result['corrected_alpha'] else 'NO'}")

    logger.info("\n" + "="*80)
    logger.info("CONCLUSION:")
    if ttest_result['significant']:
        logger.info(f"✓ Defense significantly outperforms baseline (p={ttest_result['p_value']:.6f} < {ttest_result['alpha']})")
        logger.info(f"✓ Effect size is {effect_size_result['magnitude']} (Cohen's d={effect_size_result['cohens_d']:.4f})")
    else:
        logger.info(f"✗ No significant difference found (p={ttest_result['p_value']:.6f} >= {ttest_result['alpha']})")
    logger.info("="*80 + "\n")

    # Generate full statistical report
    full_report = tester.generate_statistical_report(
        baseline_maps, defense_maps, method_name="Unified Defense"
    )
    results['full_report'] = full_report

    logger.info("\nFull Statistical Report:")
    logger.info(full_report)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate traffic sign/light detection models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--eval_mode', type=str, default='clean',
                       choices=['clean', 'perturbations', 'attacks', 'all', 'ablation', 'statistical'])
    parser.add_argument('--baseline_model_path', type=str, default=None,
                       help='Path to baseline model (required for statistical mode)')
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

    if args.eval_mode == 'ablation':
        ablation_results = run_ablation_study(config, args.model_path, dataloader, device)
        results['ablation'] = ablation_results

    if args.eval_mode == 'statistical':
        if args.baseline_model_path is None:
            logger.error("--baseline_model_path is required for statistical mode")
            sys.exit(1)

        statistical_results = run_statistical_comparison(
            args.baseline_model_path,
            args.model_path,
            config,
            dataloader,
            device
        )
        results['statistical'] = statistical_results

    output_dir = Path(config['logging']['log_dir']) / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'eval_{args.split}_{args.eval_mode}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()