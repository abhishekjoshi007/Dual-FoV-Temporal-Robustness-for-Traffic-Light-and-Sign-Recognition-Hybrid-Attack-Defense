import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, chi2_contingency
import logging

logger = logging.getLogger(__name__)


class StatisticalTester:
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 1000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        
        logger.info(f"StatisticalTester initialized with alpha={alpha}, n_bootstrap={n_bootstrap}")
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        n_iterations: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        if n_iterations is None:
            n_iterations = self.n_bootstrap
        
        n = len(data)
        bootstrap_means = []
        
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        mean = np.mean(data)
        
        return mean, ci_lower, ci_upper
    
    def paired_t_test(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
    ) -> Dict:
        if len(baseline_scores) != len(method_scores):
            raise ValueError("Arrays must have the same length for paired t-test")
        
        t_statistic, p_value = ttest_rel(method_scores, baseline_scores)
        
        mean_diff = np.mean(method_scores - baseline_scores)
        std_diff = np.std(method_scores - baseline_scores)
        
        cohens_d = mean_diff / (std_diff + 1e-10)
        
        significant = p_value < self.alpha
        
        return {
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'mean_difference': float(mean_diff),
            'cohens_d': float(cohens_d),
            'significant': bool(significant),
            'alpha': self.alpha,
        }
    
    def wilcoxon_signed_rank_test(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
    ) -> Dict:
        if len(baseline_scores) != len(method_scores):
            raise ValueError("Arrays must have the same length for Wilcoxon test")
        
        differences = method_scores - baseline_scores
        non_zero_diff = differences[differences != 0]
        
        if len(non_zero_diff) == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'alpha': self.alpha,
            }
        
        statistic, p_value = wilcoxon(non_zero_diff)
        
        significant = p_value < self.alpha
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(significant),
            'alpha': self.alpha,
        }
    
    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: Optional[float] = None,
    ) -> Dict:
        if alpha is None:
            alpha = self.alpha
        
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests
        
        significant = [p < corrected_alpha for p in p_values]
        
        return {
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_tests': n_tests,
            'p_values': p_values,
            'significant': significant,
        }
    
    def compute_effect_size(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
    ) -> Dict:
        mean_baseline = np.mean(baseline_scores)
        mean_method = np.mean(method_scores)
        
        std_baseline = np.std(baseline_scores, ddof=1)
        std_method = np.std(method_scores, ddof=1)
        
        pooled_std = np.sqrt((std_baseline**2 + std_method**2) / 2)
        cohens_d = (mean_method - mean_baseline) / (pooled_std + 1e-10)
        
        if cohens_d < 0.2:
            magnitude = 'negligible'
        elif cohens_d < 0.5:
            magnitude = 'small'
        elif cohens_d < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        return {
            'cohens_d': float(cohens_d),
            'magnitude': magnitude,
            'mean_baseline': float(mean_baseline),
            'mean_method': float(mean_method),
            'mean_difference': float(mean_method - mean_baseline),
        }
    
    def power_analysis(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Dict:
        if alpha is None:
            alpha = self.alpha
        
        n = len(baseline_scores)
        
        effect_size_result = self.compute_effect_size(baseline_scores, method_scores)
        effect_size = effect_size_result['cohens_d']
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(0.8)
        
        noncentrality = effect_size * np.sqrt(n)
        
        power = 1 - norm.cdf(z_alpha - noncentrality) + norm.cdf(-z_alpha - noncentrality)
        
        return {
            'sample_size': n,
            'effect_size': float(effect_size),
            'alpha': alpha,
            'power': float(power),
            'powered': bool(power >= 0.8),
        }
    
    def compare_multiple_methods(
        self,
        baseline_scores: np.ndarray,
        methods_scores: Dict[str, np.ndarray],
    ) -> Dict:
        results = {}
        p_values = []
        
        for method_name, scores in methods_scores.items():
            test_result = self.paired_t_test(baseline_scores, scores)
            results[method_name] = test_result
            p_values.append(test_result['p_value'])
        
        bonferroni_result = self.bonferroni_correction(p_values)
        
        for i, (method_name, _) in enumerate(methods_scores.items()):
            results[method_name]['bonferroni_significant'] = bonferroni_result['significant'][i]
            results[method_name]['bonferroni_alpha'] = bonferroni_result['corrected_alpha']
        
        return results
    
    def mcnemar_test(
        self,
        baseline_correct: np.ndarray,
        method_correct: np.ndarray,
    ) -> Dict:
        if len(baseline_correct) != len(method_correct):
            raise ValueError("Arrays must have the same length")
        
        both_correct = np.sum((baseline_correct == 1) & (method_correct == 1))
        both_wrong = np.sum((baseline_correct == 0) & (method_correct == 0))
        baseline_only = np.sum((baseline_correct == 1) & (method_correct == 0))
        method_only = np.sum((baseline_correct == 0) & (method_correct == 1))
        
        contingency_table = np.array([
            [both_correct, baseline_only],
            [method_only, both_wrong]
        ])
        
        b = baseline_only
        c = method_only
        
        if b + c == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'contingency_table': contingency_table.tolist(),
            }
        
        statistic = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        
        significant = p_value < self.alpha
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': bool(significant),
            'contingency_table': contingency_table.tolist(),
            'alpha': self.alpha,
        }
    
    def compute_confidence_intervals_for_metrics(
        self,
        metric_values: Dict[str, np.ndarray],
        confidence: float = 0.95,
    ) -> Dict:
        results = {}
        
        for metric_name, values in metric_values.items():
            mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(
                values, confidence
            )
            
            results[metric_name] = {
                'mean': float(mean),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'confidence': confidence,
            }
        
        return results
    
    def permutation_test(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
        n_permutations: int = 10000,
    ) -> Dict:
        observed_diff = np.mean(method_scores) - np.mean(baseline_scores)
        
        combined = np.concatenate([baseline_scores, method_scores])
        n_baseline = len(baseline_scores)
        n_total = len(combined)
        
        permuted_diffs = []
        
        for _ in range(n_permutations):
            permuted = np.random.permutation(combined)
            perm_baseline = permuted[:n_baseline]
            perm_method = permuted[n_baseline:]
            
            perm_diff = np.mean(perm_method) - np.mean(perm_baseline)
            permuted_diffs.append(perm_diff)
        
        permuted_diffs = np.array(permuted_diffs)
        
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        significant = p_value < self.alpha
        
        return {
            'observed_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': bool(significant),
            'n_permutations': n_permutations,
            'alpha': self.alpha,
        }
    
    def analyze_significance_across_odds(
        self,
        baseline_by_odd: Dict[str, np.ndarray],
        method_by_odd: Dict[str, np.ndarray],
    ) -> Dict:
        results = {}
        p_values = []
        
        for odd in baseline_by_odd.keys():
            if odd not in method_by_odd:
                continue
            
            test_result = self.paired_t_test(
                baseline_by_odd[odd],
                method_by_odd[odd]
            )
            
            results[odd] = test_result
            p_values.append(test_result['p_value'])
        
        if len(p_values) > 0:
            bonferroni = self.bonferroni_correction(p_values)
            
            for i, odd in enumerate(results.keys()):
                results[odd]['bonferroni_significant'] = bonferroni['significant'][i]
                results[odd]['bonferroni_alpha'] = bonferroni['corrected_alpha']
        
        return results
    
    def test_variance_homogeneity(
        self,
        *groups: np.ndarray,
    ) -> Dict:
        statistic, p_value = stats.levene(*groups)
        
        homogeneous = p_value >= self.alpha
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'homogeneous': bool(homogeneous),
            'alpha': self.alpha,
        }
    
    def test_normality(
        self,
        data: np.ndarray,
    ) -> Dict:
        if len(data) < 3:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'normal': False,
                'alpha': self.alpha,
                'note': 'Sample size too small for normality test',
            }
        
        statistic, p_value = stats.shapiro(data)
        
        normal = p_value >= self.alpha
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'normal': bool(normal),
            'alpha': self.alpha,
        }
    
    def generate_statistical_report(
        self,
        baseline_scores: np.ndarray,
        method_scores: np.ndarray,
        method_name: str = "Method",
    ) -> str:
        report = []
        report.append(f"Statistical Analysis Report: {method_name} vs Baseline")
        report.append("=" * 60)
        report.append("")
        
        report.append("Descriptive Statistics:")
        report.append(f"  Baseline: mean={np.mean(baseline_scores):.4f}, std={np.std(baseline_scores):.4f}")
        report.append(f"  {method_name}: mean={np.mean(method_scores):.4f}, std={np.std(method_scores):.4f}")
        report.append("")
        
        normality_baseline = self.test_normality(baseline_scores)
        normality_method = self.test_normality(method_scores)
        report.append("Normality Tests (Shapiro-Wilk):")
        report.append(f"  Baseline: p={normality_baseline['p_value']:.4f}, normal={normality_baseline['normal']}")
        report.append(f"  {method_name}: p={normality_method['p_value']:.4f}, normal={normality_method['normal']}")
        report.append("")
        
        ttest_result = self.paired_t_test(baseline_scores, method_scores)
        report.append("Paired t-test:")
        report.append(f"  t-statistic: {ttest_result['t_statistic']:.4f}")
        report.append(f"  p-value: {ttest_result['p_value']:.4f}")
        report.append(f"  Significant: {ttest_result['significant']} (α={self.alpha})")
        report.append(f"  Cohen's d: {ttest_result['cohens_d']:.4f}")
        report.append("")
        
        wilcoxon_result = self.wilcoxon_signed_rank_test(baseline_scores, method_scores)
        report.append("Wilcoxon Signed-Rank Test:")
        report.append(f"  Statistic: {wilcoxon_result['statistic']:.4f}")
        report.append(f"  p-value: {wilcoxon_result['p_value']:.4f}")
        report.append(f"  Significant: {wilcoxon_result['significant']} (α={self.alpha})")
        report.append("")
        
        effect_size = self.compute_effect_size(baseline_scores, method_scores)
        report.append("Effect Size Analysis:")
        report.append(f"  Cohen's d: {effect_size['cohens_d']:.4f} ({effect_size['magnitude']})")
        report.append(f"  Mean difference: {effect_size['mean_difference']:.4f}")
        report.append("")
        
        power = self.power_analysis(baseline_scores, method_scores)
        report.append("Statistical Power:")
        report.append(f"  Sample size: {power['sample_size']}")
        report.append(f"  Power: {power['power']:.4f}")
        report.append(f"  Adequately powered (≥0.8): {power['powered']}")
        report.append("")
        
        mean, ci_lower, ci_upper = self.bootstrap_confidence_interval(method_scores)
        report.append("95% Confidence Interval (Bootstrap):")
        report.append(f"  Mean: {mean:.4f}")
        report.append(f"  CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return "\n".join(report)