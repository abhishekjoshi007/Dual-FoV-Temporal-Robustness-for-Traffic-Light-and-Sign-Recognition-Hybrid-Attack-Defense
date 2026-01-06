# Sequence-Preserving Dual-FoV Defense for Traffic Sign and Light Recognition

**Authors:** Abhishek Joshi¹, Janhavi Krishna Koda¹, Abhishek Phadke²
¹Texas A&M University-Corpus Christi | ²Christopher Newport University

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

This repository implements a novel **three-layer unified defense framework** for robust traffic light and sign recognition in autonomous vehicles (AVs). The system addresses both digital adversarial attacks and natural environmental perturbations (rain, fog, glare, occlusions) through:

1. **Dual Field-of-View (FoV) Architecture**: Synchronized mid-range (50° FoV) and long-range (25° FoV) cameras
2. **Sequence-Preserving Processing**: Temporal voting across 1-second frame sequences
3. **Unified Defense Stack**: Feature squeezing, temperature scaling, and entropy-based anomaly detection
4. **Multi-Source Benchmark**: 500 sequences (150k frames) across 4 Operational Design Domains (ODDs)

### 🎯 Key Results

| Metric | Baseline | Unified Defense | Improvement |
|--------|----------|-----------------|-------------|
| **mAP** | 70.2% | **79.8%** | +9.6% |
| **ASR** | 37.4% | **18.2%** | **-51%** ↓ |
| **Critical Failures** | 9.8% | **6.6%** | **-33%** ↓ |
| **Stability Score** | 0.65 | **0.85** | +31% |

---

## ✨ Key Features

### 🛡️ Defense Mechanisms

- **Feature Squeezing**: 5-bit quantization + 3×3 median filtering
- **Inference-Time Temperature Scaling**: τ=3.0 for confidence calibration
- **Entropy-Based FoV Gating**: Adaptive camera selection based on prediction uncertainty
- **Temporal Voting**: Quality-weighted aggregation across 5-frame windows

### 🌍 Environmental Robustness

Natural perturbation suite simulating real-world conditions:
- **Weather**: Rain (15-35% coverage, 50-120px streaks), fog (β=0.02 scattering), snow
- **Lighting**: Sun glare, headlight interference, lens flare
- **Sensor**: Dirt/mud occlusion (60-90% opacity), motion blur, defocus
- **Scene Complexity**: Sign clustering, background confusers, partial visibility

### 🎯 Attack Coverage

Comprehensive evaluation against:
- **Digital Attacks**: FGSM, PGD (α=2.5ε/10), Universal Adversarial Perturbations (UAP)
- **Black-Box Attacks**: SimBA, Square Attack (query-limited)
- **Hybrid Attacks**: Combined natural + adversarial perturbations
- **Adaptive Attacks**: Defense-aware PGD with Expectation over Transformation (EOT)

### 📊 Safety-Aware Evaluation

- **Risk-Weighted Metrics**: MUTCD-informed severity matrix (stop→go: 10×, red→green: 10×)
- **ODD-Stratified Analysis**: Highway, Night, Rainy, Urban performance breakdown
- **Temporal Stability**: Confidence volatility, label flip rate across sequences
- **Statistical Rigor**: Bootstrap CIs (n=1000), Bonferroni correction (α=0.002)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DUAL-FoV INPUT STREAMS                      │
│  ┌─────────────────────┐        ┌─────────────────────┐        │
│  │ Mid-Range Camera    │        │ Long-Range Camera   │        │
│  │ (50° FoV, 0.5-50m) │        │ (25° FoV, 10-200m)  │        │
│  └──────────┬──────────┘        └──────────┬──────────┘        │
└─────────────┼─────────────────────────────┼─────────────────────┘
              │                             │
              ▼                             ▼
    ┌─────────────────────────────────────────────────┐
    │         UNIFIED DEFENSE STACK (Layer 1-3)       │
    │  ┌──────────────────────────────────────────┐   │
    │  │  Layer 1: Feature Squeezing              │   │
    │  │  • 5-bit quantization                    │   │
    │  │  • 3×3 median filtering                  │   │
    │  └──────────────────────────────────────────┘   │
    │  ┌──────────────────────────────────────────┐   │
    │  │  Layer 2: Temperature Scaling (τ=3)      │   │
    │  │  • Confidence calibration                │   │
    │  └──────────────────────────────────────────┘   │
    │  ┌──────────────────────────────────────────┐   │
    │  │  Layer 3: Entropy Gating                 │   │
    │  │  • Cross-FoV validation                  │   │
    │  │  • Anomaly detection (threshold=0.5)     │   │
    │  └──────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  TEMPORAL VOTING      │
              │  • 5-frame window     │
              │  • Quality weighting  │
              │  • NMS (IoU=0.5)      │
              └───────────────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ Final Detection │
                │ (9 classes)     │
                └─────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.9 or higher
- **CUDA**: 11.8 (for GPU acceleration)
- **GPU**: NVIDIA GPU with 11GB+ VRAM (recommended: RTX 2080Ti or better)
- **OS**: Linux (Ubuntu 20.04+) or macOS

### Step 1: Clone Repository

```bash
git clone https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense.git
cd Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n dual-fov python=3.9
conda activate dual-fov

# OR using venv
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "
import torch
import ultralytics
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Ultralytics: {ultralytics.__version__}')
"
```

Expected output:
```
PyTorch: 2.0.1
CUDA Available: True
Ultralytics: 8.0.196
```

---

## 📦 Dataset Preparation

Our benchmark integrates **4 source datasets** into a unified dual-FoV format across 4 ODDs.

### Dataset Sources

| Dataset | License | Sequences | URL |
|---------|---------|-----------|-----|
| **aiMotive 3D** | Research-only | 210 
| **Waymo Open** | CC BY-NC 4.0 | 80 | 
| **Udacity SDC** | MIT | 45 | 
| **Texas (ours)** | By request | 165 | Contact authors |

### Combined License

⚠️ **Important**: Due to Waymo's CC BY-NC 4.0 license, the combined benchmark is **restricted to non-commercial research use only**.

### Reconstruction Instructions

We provide **preprocessing scripts and frame indices** to recreate our benchmark from original sources:

#### Step 1: Download Source Datasets

```bash
# 1. aiMotive 3D (requires Kaggle account)
# Visit: https://www.kaggle.com/datasets/tamasmatuszka/aimotive-3d-traffic-light-and-sign-dataset
# Accept license → Download → Extract to data/raw/aimotive/

# 2. Waymo Open Dataset
# Visit: https://waymo.com/open/
# Register → Download perception dataset → Extract to data/raw/waymo/

# 3. Udacity Self-Driving Car
git clone https://github.com/udacity/self-driving-car.git data/raw/udacity/

# 4. Texas sequences (contact: ajoshi5@islander.tamucc.edu)
# Available upon reasonable request via institutional DUA
```

#### Step 2: Run Preprocessing Pipeline

```bash
# Process aiMotive dataset
python preprocessing/av.py --input data/raw/aimotive/ --output data/processed/

# Filter sequences by ODD
python preprocessing/odd_frames.py --data data/processed/ --config configs/odd_filtering.yaml

# Extract dual-FoV frames
python preprocessing/camera_filter.py --cameras F_MIDRANGECAM_C,F_LONGRANGECAM_C

# Export YOLO format annotations
python preprocessing/bounding-box/export_yolo_aimotive_dual_fov.py \
    --input data/processed/ \
    --output data/unified/

# Harmonize all sources
python preprocessing/main.py --config configs/preprocessing.yaml
```

#### Step 3: Verify Dataset

```bash
python -c "
from data.dataset import DualFoVTrafficDataset
dataset = DualFoVTrafficDataset(root_dir='data/unified/', split='train')
print(f'Total sequences: {len(dataset)}')
print(f'Sample sequence shape: {dataset[0][\"mid_range\"].shape}')
"
```

Expected output:
```
Total sequences: 300
Sample sequence shape: torch.Size([30, 3, 640, 640])
```

### Dataset Statistics

| ODD | Sequences | Frames | Perturbations | Characteristics |
|-----|-----------|--------|---------------|-----------------|
| **Highway** | 120 | 3,600 | 7 types | High-speed, glare-prone |
| **Night** | 80 | 2,400 | 6 types | Low-light, headlight interference |
| **Rainy** | 70 | 2,100 | 8 types | Rain, fog, streaking |
| **Urban** | 230 | 6,900 | 9 types | Dense, occlusions, clustering |
| **Total** | **500** | **15,000** | **~150k** (w/ variants) | Dual-FoV, 1s sequences |

---

## ⚡ Quick Start

### Minimal Example: Run Defense on Single Image

```python
import torch
from models.baseline_yolov8m import BaselineYOLOv8m
from models.unified_defense_stack import UnifiedDefenseStack
from PIL import Image
import torchvision.transforms as T

# Load pre-trained model
base_model = BaselineYOLOv8m(num_classes=9)
base_model.load('checkpoints/det_natural/best.pt')

# Initialize defense stack
defense = UnifiedDefenseStack(
    base_model=base_model,
    bit_depth=5,
    median_kernel=3,
    temperature=3.0,
    entropy_threshold=0.5
)

# Load dual-FoV images
transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
mid_range = transform(Image.open('example_mid.jpg')).unsqueeze(0)
long_range = transform(Image.open('example_long.jpg')).unsqueeze(0)

# Run defense
with torch.no_grad():
    results = defense(mid_range, long_range, conf_threshold=0.25)

print(f"Selected FoV: {results['selected_fov']}")
print(f"Detections: {len(results['selected_results'])}")
print(f"High Uncertainty: {results['high_uncertainty']}")
```

### Evaluate on Test Set

```bash
python scripts/evaluate.py \
    --config configs/experiments/unified_defense.yaml \
    --checkpoint checkpoints/unified_defense/best.pt \
    --split test \
    --output results/unified_defense/
```

---

## 🔬 Reproducing Paper Results

To reproduce **Table 9** (main results) from the paper:

### Step 1: Download Pre-trained Checkpoints

```bash
# Download from GitHub Releases
wget https://github.com/abhishekjoshi007/.../releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip -d checkpoints/

# Verify checksums
md5sum -c checkpoints/checksums.md5
```

Available checkpoints:
- `det_clean.pth`: Baseline trained on clean data (68.4% mAP)
- `det_natural.pth`: Trained with natural perturbations (74.9% mAP)
- `unified_defense.pth`: Full defense stack (79.8% mAP)

### Step 2: Run Evaluation Suite

```bash
# Reproduce all tables and figures
bash scripts/reproduce_tables.sh

# This runs:
# 1. Baseline evaluation (Table 9, row 1)
# 2. Defense ablation studies (Table 10)
# 3. Per-class performance (Table 11)
# 4. ODD-stratified results (Table 12)
# 5. Attack robustness (Table 13)
# 6. Statistical significance tests (Table 19)
```

### Step 3: Compare Results

```bash
# Generate comparison report
python evaluation/compare_results.py \
    --ground_truth paper_results/table9.json \
    --predictions results/ \
    --output comparison_report.pdf
```

Expected output matches:
```
✓ Baseline mAP: 70.2% ± 1.3% (paper: 70.2%)
✓ Defense mAP: 79.8% ± 0.8% (paper: 79.8%)
✓ ASR reduction: 51.3% (paper: 51%)
✓ Critical failures: -32% (paper: -32%)
```

### Reproduction Notes

- **Random seeds**: Fixed to 42 for NumPy, PyTorch, CUDA
- **Hardware**: Results obtained on 4× NVIDIA A100 80GB GPUs
- **Tolerance**: ±0.5% mAP variation due to floating-point precision
- **Statistical tests**: Bootstrap CIs may vary slightly (±0.1% at 95% confidence)

---

## 🏋️ Training

### Train Baseline Model (No Defense)

```bash
python scripts/train.py \
    --config configs/experiments/baseline.yaml \
    --output checkpoints/baseline/ \
    --gpus 0,1,2,3 \
    --mixed_precision
```

### Train with Natural Perturbations

```bash
python scripts/train.py \
    --config configs/experiments/det_natural.yaml \
    --output checkpoints/det_natural/ \
    --augmentation natural_perturbations \
    --gpus 0,1,2,3
```

### Train Unified Defense Stack

```bash
python scripts/train.py \
    --config configs/experiments/unified_defense.yaml \
    --output checkpoints/unified_defense/ \
    --base_model checkpoints/det_natural/best.pt \
    --freeze_backbone \
    --freeze_epochs 10 \
    --gpus 0,1,2,3
```

### Training Configuration

Key hyperparameters (from `configs/experiments/unified_defense.yaml`):

```yaml
training:
  epochs: 50
  batch_size: 16
  optimizer: AdamW
  lr: 0.00005  # Fine-tuning learning rate
  weight_decay: 0.0001
  scheduler: cosine
  patience: 10
  mixed_precision: true

defense:
  feature_squeeze:
    bit_depth: 5
    median_kernel: 3
  temperature_scaling:
    temperature: 3.0
  entropy_gating:
    entropy_threshold: 0.5
  temporal_voting:
    window_size: 5
    persistence_threshold: 3
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/unified_defense/ --port 6006

# View in browser: http://localhost:6006
```

---

## 📊 Evaluation

### Comprehensive Evaluation Pipeline

```bash
python scripts/evaluate.py \
    --config configs/experiments/unified_defense.yaml \
    --checkpoint checkpoints/unified_defense/best.pt \
    --split test \
    --metrics mAP,RW-mAP,ASR,CFR,stability \
    --attacks fgsm,pgd,uap,hybrid \
    --perturbations rain,fog,glare,dirt \
    --odd_stratified \
    --statistical_tests \
    --output results/full_evaluation/
```

### Evaluation Options

| Flag | Description | Example |
|------|-------------|---------|
| `--metrics` | Metrics to compute | `mAP,RW-mAP,ASR,CFR` |
| `--attacks` | Adversarial attacks | `fgsm,pgd,uap,hybrid` |
| `--perturbations` | Natural perturbations | `rain,fog,glare` |
| `--odd_stratified` | Per-ODD breakdown | Flag (no value) |
| `--statistical_tests` | Bootstrap CIs + t-tests | Flag (no value) |
| `--visualize` | Save detection visualizations | Flag (no value) |

### Custom Evaluation Script

```python
from evaluation.risk_weighted_metrics import RiskWeightedMetrics
from evaluation.odd_stratified_eval import ODDStratifiedEvaluator

# Initialize evaluators
rw_metrics = RiskWeightedMetrics(num_classes=9, severity_matrix='mutcd')
odd_eval = ODDStratifiedEvaluator(odd_list=['highway', 'night', 'rainy', 'urban'])

# Evaluate
results = odd_eval.evaluate(
    model=defense,
    dataloader=test_loader,
    metrics=['mAP', 'RW-mAP', 'ASR', 'CFR']
)

# Print ODD-stratified results
for odd, metrics in results.items():
    print(f"{odd}: mAP={metrics['mAP']:.1f}%, ASR={metrics['ASR']:.1f}%")
```

### Attack Evaluation

```python
from attacks.hybrid_attacks import HybridAttackSuite

# Initialize attack suite
attack_suite = HybridAttackSuite(
    epsilon=8/255,
    alpha=None,  # Calculated as 2.5*ε/10 per paper
    num_iter=10
)

# PGD attack
perturbed = attack_suite.pgd_attack(
    model=base_model,
    images=clean_images,
    targets=labels
)

# Evaluate attack success
asr_results = attack_suite.evaluate_attack_success(
    model=base_model,
    clean_images=clean_images,
    perturbed_images=perturbed
)

print(f"Attack Success Rate: {asr_results['asr']:.2f}%")
```

---

## 📁 Project Structure

```
Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── configs/                           # Configuration files
│   └── experiments/
│       ├── baseline.yaml              # Baseline YOLOv8m config
│       ├── det_natural.yaml           # Natural perturbation training
│       └── unified_defense.yaml       # Full defense stack config
│
├── models/                            # Defense implementations
│   ├── baseline_yolov8m.py            # YOLOv8m wrapper
│   ├── unified_defense_stack.py       # 3-layer defense (Alg. 1)
│   └── temporal_voting.py             # Sequence voting (Alg. 5)
│
├── attacks/                           # Attack implementations
│   ├── natural_perturbations.py       # 9 physical perturbations
│   └── hybrid_attacks.py              # FGSM, PGD, UAP, hybrid
│
├── data/                              # Dataset handling
│   ├── dataset.py                     # DualFoVTrafficDataset
│   ├── transforms.py                  # Augmentation pipeline
│   └── aimotive-dataset-loader/       # aiMotive data utilities
│
├── preprocessing/                     # Dataset preprocessing
│   ├── main.py                        # Main preprocessing pipeline
│   ├── odd_frames.py                  # ODD-based frame filtering
│   ├── camera_filter.py               # Dual-FoV extraction
│   └── bounding-box/
│       └── export_yolo_aimotive_dual_fov.py  # YOLO format export
│
├── evaluation/                        # Evaluation metrics
│   ├── risk_weighted_metrics.py       # RW-mAP, RW-ASR, CFR
│   ├── odd_stratified_eval.py         # Per-ODD evaluation
│   └── statistical_tests.py           # Bootstrap CIs, t-tests
│
├── scripts/                           # Training/evaluation scripts
│   ├── train.py                       # Training pipeline
│   ├── evaluate.py                    # Evaluation pipeline
│   └── reproduce_tables.sh            # Reproduce paper results
│
├── utils/                             # Utility functions
│   ├── logger.py                      # Experiment logging
│   └── visualization.py               # Detection visualization
│
├── checkpoints/                       # Pre-trained models
│   ├── det_clean.pth                  # Baseline (68.4% mAP)
│   ├── det_natural.pth                # Natural training (74.9% mAP)
│   └── unified_defense.pth            # Full defense (79.8% mAP)
│
└── unified dataset/                   # Dataset examples and docs
    └── basic_loading.py               # Dataset loading API
```


## 📄 License

### Code License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

### Dataset License

⚠️ **Important**: The reconstructed benchmark follows the **most restrictive source license**:

- **Overall**: Non-commercial research use only (due to Waymo CC BY-NC 4.0)
- **aiMotive**: Research-only (Kaggle ToS)
- **Waymo**: CC BY-NC 4.0 (non-commercial)
- **Udacity**: MIT License
- **Texas (ours)**: Available upon request for research purposes

**Usage Restrictions:**
- ✅ Academic research
- ✅ Educational purposes
- ✅ Non-commercial publications
- ❌ Commercial applications
- ❌ Redistribution of source data

**Reconstruction Compliance:**
- We provide **preprocessing scripts** (MIT licensed)
- We provide **frame indices** (factual data, not copyrightable)
- Users must **download source datasets** from original providers
- Users must **accept original license terms**

This approach follows precedent from COCO, ImageNet, and similar benchmarks.

---

## 🙏 Acknowledgments

We acknowledge the use of publicly available datasets:

- **aiMotive 3D Traffic Light and Sign Dataset** - [Kaggle](https://www.kaggle.com/datasets/tamasmatuszka/aimotive-3d-traffic-light-and-sign-dataset)
- **Waymo Open Dataset** - [waymo.com/open](https://waymo.com/open/)
- **Udacity Self-Driving Car Dataset** - [GitHub](https://github.com/udacity/self-driving-car)

Special thanks to:
- Texas A&M University-Corpus Christi for computational resources
- Christopher Newport University for research support
- Ultralytics for the YOLOv8 implementation

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
vim configs/experiments/unified_defense.yaml
# Change: batch_size: 16 → batch_size: 8
```

**2. Dataset Not Found**
```bash
# Verify dataset structure
python -c "
from data.dataset import DualFoVTrafficDataset
dataset = DualFoVTrafficDataset(root_dir='data/unified/', split='train')
print(f'Found {len(dataset)} sequences')
"
```

**3. Missing Checkpoints**
```bash
# Download pre-trained models
wget https://github.com/abhishekjoshi007/.../releases/download/v1.0/checkpoints.zip
unzip checkpoints.zip -d checkpoints/
```

**4. Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**5. Slow Evaluation**
```bash
# Use smaller test set for quick validation
python scripts/evaluate.py --config configs/experiments/unified_defense.yaml --max_samples 100
```

### Performance Tips

- **Multi-GPU Training**: Use `--gpus 0,1,2,3` for data parallelism
- **Mixed Precision**: Enable with `--mixed_precision` for 2× speedup
- **Caching**: Set `num_workers: 8` in config for faster data loading
- **Validation Frequency**: Reduce `--val_interval 5` during training

---

## 📞 Contact

For questions, issues, or collaboration:

- **Abhishek Joshi**: [ajoshi5@islander.tamucc.edu](mailto:ajoshi5@islander.tamucc.edu)
- **Janhavi Krishna Koda**: [jkoda@islander.tamucc.edu](mailto:jkoda@islander.tamucc.edu)
- **Abhishek Phadke**: [abhishek.phadke@cnu.edu](mailto:abhishek.phadke@cnu.edu)

**GitHub Issues**: [Report bugs or request features](https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense/issues)

---

## 🔄 Updates

**v1.0.0** (January 2026)
- Initial release
- Paper accepted at MDPI
- Code-paper alignment verified
- Pre-trained checkpoints available

**Planned Updates:**
- [ ] Extended ODD coverage (construction zones, tunnels)
- [ ] Certified robustness guarantees
- [ ] Integration with planning/control modules
- [ ] Real-world deployment on test vehicle

---

<p align="center">
  <img src="docs/assets/dual_fov_demo.gif" alt="Dual-FoV Demo" width="600"/>
  <br>
  <em>Demo: Dual-FoV defense in action (Highway ODD with sun glare)</em>
</p>

---
