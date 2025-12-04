# 🚗 Dual-FoV Traffic Sign and Light Recognition Dataset - README

<div align="center">

![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-CC_BY_4.0-green?style=for-the-badge)
![Size](https://img.shields.io/badge/Size-85GB-orange?style=for-the-badge)
![Frames](https://img.shields.io/badge/Frames-150K-blue?style=for-the-badge)

**A comprehensive multi-source dataset for robust autonomous vehicle perception research**

[📖 Paper](https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense) | [💻 Code](https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense) | [📊 Kaggle Dataset](https://www.kaggle.com/datasets/joshi07abhishek/dual-fov-for-traffic-light-and-sign-recognition)

</div>


## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Statistics](#-dataset-statistics)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Data Structure](#-data-structure)
- [Annotation Format](#-annotation-format)
- [Data Splits](#-data-splits)
- [Visualization](#-visualization)
- [PyTorch Integration](#-pytorch-integration)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)


## 🎯 Overview

This dataset provides **500 synchronized sequences** (~150,000 frames) featuring dual field-of-view camera streams with precise 3D annotations for traffic lights and signs. Designed for temporal robustness analysis under real-world environmental perturbations in autonomous vehicle perception.

### Key Features

✅ **Dual-FoV Synchronized Streams** - Mid-range (50° FoV) + Long-range (25° FoV)  
✅ **Four Operational Design Domains** - Highway, Night, Rainy, Urban  
✅ **Comprehensive 3D Annotations** - Traffic lights & signs with occlusion scores  
✅ **Natural Perturbation Labels** - Dirt, glare, weather effects  
✅ **Multi-Source Integration** - aiMotive, Waymo, Udacity, Texas recordings  


## 📊 Dataset Statistics

| Metric | Value |
---|
| **Total Sequences** | 500 (15+ sec each) |
| **Total Frames** | ~150,000 |
| **Traffic Lights** | 45,000+ instances |
| **Traffic Signs** | 38,000+ instances |
| **ODDs** | Highway (120), Night (80), Rainy (70), Urban (230) |
| **Camera Streams** | 2 primary (Mid + Long range) |
| **Size** | 85 GB (62 GB compressed) |
| **License** | CC BY 4.0 |


## 🔧 Installation

### On Kaggle Notebooks

The dataset is automatically available when you add it to your Kaggle notebook. No installation required!

```python
# Dataset is available at:
# /kaggle/input/dual-fov-for-traffic-light-and-sign-recognition/
```

### For Local Development

```bash
# Install required packages
pip install pillow numpy matplotlib torch torchvision

# Download the dataset using Kaggle API
pip install kaggle
kaggle datasets download -d joshi07abhishek/dual-fov-for-traffic-light-and-sign-recognition

# Extract
unzip dual-fov-for-traffic-light-and-sign-recognition.zip -d ./dataset
```

### Requirements

```
Python >= 3.7
pillow >= 8.0.0
numpy >= 1.19.0
matplotlib >= 3.3.0
torch >= 1.8.0 (optional, for PyTorch integration)
```


## 🚀 Quick Start

### Step 1: Download the Loader Class

Save the `DualFoVDatasetLoader` class (provided in the full code section) or use this minimal loader:

```python
import json
from pathlib import Path
from PIL import Image

# For Kaggle Notebooks
DATASET_PATH = '/kaggle/input/dual-fov-for-traffic-light-and-sign-recognition'

# For local use
# DATASET_PATH = './dataset'
```

### Step 2: Load Your First Frame

```python
from dataset_loader import DualFoVDatasetLoader

# Initialize loader
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Load a complete frame (images + annotations)
frame_data = loader.load_complete_frame(
    odd='highway',
    sequence='20231006-114522-00.15.00-00.15.15@Sogun',
    frame_id='0000001'
)

# Access data
mid_img = frame_data['images']['mid_range']
long_img = frame_data['images']['long_range']
lights = frame_data['annotations']['traffic_lights']
signs = frame_data['annotations']['traffic_signs']

print(f"✓ Loaded frame {frame_data['frame_id']}")
print(f"✓ Traffic lights: {len(lights.get('objects', []))}")
print(f"✓ Traffic signs: {len(signs.get('objects', []))}")
```

**Expected Output:**
```
✓ Loaded frame 0000001
✓ Traffic lights: 3
✓ Traffic signs: 2
```


## 💡 Usage Examples

### Example 1: Load Single Frame with Dual FoV

```python
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Load both mid-range and long-range images
mid_img, long_img = loader.load_dual_fov_frame(
    odd='highway',
    sequence='20231006-114522-00.15.00-00.15.15@Sogun',
    frame_id='0000001'
)

# Display image sizes
print(f"Mid-range: {mid_img.size}")
print(f"Long-range: {long_img.size}")

# Load annotations separately
traffic_lights = loader.load_traffic_lights('highway', '20231006-114522-00.15.00-00.15.15@Sogun', '0000001')
traffic_signs = loader.load_traffic_signs('highway', '20231006-114522-00.15.00-00.15.15@Sogun', '0000001')

print(f"Traffic lights detected: {len(traffic_lights.get('objects', []))}")
print(f"Traffic signs detected: {len(traffic_signs.get('objects', []))}")
```


### Example 2: Iterate Through an Entire Sequence

```python
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Iterate through all frames in a sequence
sequence_name = '20231006-114522-00.15.00-00.15.15@Sogun'

for frame_data in loader.iterate_sequence('highway', sequence_name):
    frame_id = frame_data['frame_id']
    print(f"📍 Processing frame {frame_id}")
    
    # Access images
    mid_img = frame_data['images']['mid_range']
    
    # Process traffic lights
    for obj in frame_data['annotations']['traffic_lights'].get('objects', []):
        state = obj.get('state', 'unknown')
        distance = obj.get('distance', 0)
        print(f"  🚦 Light {obj['id']}: {state} at {distance:.1f}m")
    
    # Process traffic signs
    for obj in frame_data['annotations']['traffic_signs'].get('objects', []):
        sign_type = obj.get('type', 'unknown')
        text = obj.get('text', '')
        print(f"  🚸 Sign {obj['id']}: {sign_type} ({text})")
```

**Expected Output:**
```
📍 Processing frame 0000001
  🚦 Light TL_001: red at 45.3m
  🚦 Light TL_002: green at 67.8m
  🚸 Sign TS_042: us_stop (STOP)
📍 Processing frame 0000002
  🚦 Light TL_001: red at 43.1m
  ...
```


### Example 3: Load All Sequences from an ODD

```python
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Get all sequences from highway ODD
highway_sequences = loader.sequences['highway']

print(f"Found {len(highway_sequences)} highway sequences\n")

for seq in highway_sequences[:3]:  # Process first 3 sequences
    print(f"📂 Sequence: {seq}")
    
    # Get frame count
    frame_ids = loader.get_frame_ids('highway', seq)
    print(f"   Frames: {len(frame_ids)}")
    
    # Load calibration (once per sequence)
    calib = loader.load_calibration('highway', seq)
    print(f"   Cameras: {', '.join(calib.keys())}")
    
    # Sample first frame
    if frame_ids:
        lights = loader.load_traffic_lights('highway', seq, frame_ids[0])
        signs = loader.load_traffic_signs('highway', seq, frame_ids[0])
        print(f"   Sample: {len(lights.get('objects', []))} lights, {len(signs.get('objects', []))} signs")
    print()
```

**Expected Output:**
```
Found 120 highway sequences

📂 Sequence: 20231006-114522-00.15.00-00.15.15@Sogun
   Frames: 450
   Cameras: F_MIDRANGECAM_C, F_LONGRANGECAM_C, F_CTCAM_L, F_CTCAM_R
   Sample: 3 lights, 2 signs

📂 Sequence: 20231006-120334-00.15.00-00.15.15@Highway101
   Frames: 480
   Cameras: F_MIDRANGECAM_C, F_LONGRANGECAM_C, F_CTCAM_L, F_CTCAM_R
   Sample: 2 lights, 4 signs
```


### Example 4: Access Camera Calibration

```python
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Load calibration parameters
calib = loader.load_calibration('highway', '20231006-114522-00.15.00-00.15.15@Sogun')

# Access mid-range camera intrinsics
mid_cam = calib['F_MIDRANGECAM_C']
print("Mid-range Camera Intrinsics:")
print(f"  Focal Length: fx={mid_cam['fx']:.2f}, fy={mid_cam['fy']:.2f}")
print(f"  Principal Point: cx={mid_cam['cx']:.2f}, cy={mid_cam['cy']:.2f}")
print(f"  Resolution: {mid_cam['width']}x{mid_cam['height']}")

# Load extrinsic matrices
extrinsics = loader.load_extrinsics('highway', '20231006-114522-00.15.00-00.15.15@Sogun')
print(f"\nExtrinsic matrices available for: {', '.join(extrinsics.keys())}")
```


### Example 5: Filter Frames by Perturbation Type

```python
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

# Find all rainy frames with glare
rainy_sequences = loader.sequences['rainy']

glare_frames = []

for seq in rainy_sequences:
    frame_ids = loader.get_frame_ids('rainy', seq)
    
    for frame_id in frame_ids:
        signs = loader.load_traffic_signs('rainy', seq, frame_id)
        
        # Check for glare degradation
        for obj in signs.get('objects', []):
            if 'glare' in obj.get('degradation', []):
                glare_frames.append(('rainy', seq, frame_id))
                break

print(f"Found {len(glare_frames)} frames with glare perturbation")

# Load one example
if glare_frames:
    odd, seq, frame_id = glare_frames[0]
    frame_data = loader.load_complete_frame(odd, seq, frame_id)
    print(f"Example: {odd}/{seq}/frame_{frame_id}")
```


## 📁 Data Structure

```
/kaggle/input/dual-fov-for-traffic-light-and-sign-recognition/
│
├── highway/                                    # Highway ODD (120 sequences)
│   ├── 20231006-114522-00.15.00-00.15.15@Sogun/
│   │   ├── sensor/
│   │   │   ├── calibration/
│   │   │   │   ├── calibration.json           # Camera intrinsics
│   │   │   │   └── extrinsic_matrices.json    # Camera extrinsics
│   │   │   ├── camera/
│   │   │   │   ├── F_MIDRANGECAM_C/           # Mid-range images (50° FoV)
│   │   │   │   │   └── F_MIDRANGECAM_C_0000001.jpg
│   │   │   │   ├── F_LONGRANGECAM_C/          # Long-range images (25° FoV)
│   │   │   │   │   └── F_LONGRANGECAM_C_0000001.jpg
│   │   │   │   ├── F_CTCAM_L/                 # Context camera (optional)
│   │   │   │   └── F_CTCAM_R/                 # Context camera (optional)
│   │   │   └── gnssins/
│   │   │       └── egomotion2.json            # GNSS/INS trajectory
│   │   ├── traffic_light/box/3d_body/
│   │   │   └── frame_0000001.json             # Traffic light annotations
│   │   └── traffic_sign/box/3d_body/
│   │       └── frame_0000001.json             # Traffic sign annotations
│   └── [119 more sequences...]
│
├── night/                                      # Night ODD (80 sequences)
├── rainy/                                      # Rainy ODD (70 sequences)
└── urban/                                      # Urban ODD (230 sequences)
```


## 📝 Annotation Format

### Traffic Light JSON

```json
{
  "frame_id": "0000001",
  "timestamp": "2023-10-06T11:45:22.000000Z",
  "objects": [
    {
      "id": "TL_001",
      "type": "traffic_light",
      "state": "red",
      "bbox_3d": {
        "center": [12.5, 3.2, 45.3],
        "dimensions": [0.3, 0.8, 0.3],
        "rotation": [0.0, 0.0, 1.57]
      },
      "occlusion": 0.15,
      "distance": 45.3
    }
  ]
}
```

**States:** `red`, `green`, `yellow`, `arrow_left`, `arrow_right`, `pedestrian`

### Traffic Sign JSON

```json
{
  "frame_id": "0000001",
  "timestamp": "2023-10-06T11:45:22.000000Z",
  "objects": [
    {
      "id": "TS_042",
      "type": "us_stop",
      "text": "STOP",
      "bbox_3d": {
        "center": [8.2, 1.5, 28.7],
        "dimensions": [0.6, 0.6, 0.1],
        "rotation": [0.0, 0.0, 0.0]
      },
      "occlusion": 0.05,
      "distance": 28.7,
      "degradation": ["dirt", "glare"]
    }
  ]
}
```

**Sign Types:** `us_stop`, `us_yield`, `us_speedlimit_XX`, `us_oneway`, `us_do_not_enter`, etc.


## 🎲 Data Splits

### Creating Train/Val/Test Splits

```python
def create_data_splits(loader, train=0.6, val=0.2, test=0.2, seed=42):
    """
    Create reproducible train/val/test splits balanced across ODDs
    
    Returns: Dictionary with 'train', 'val', 'test' keys
    """
    import numpy as np
    np.random.seed(seed)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for odd in loader.odds:
        if odd not in loader.sequences:
            continue
            
        sequences = loader.sequences[odd]
        n_total = len(sequences)
        
        # Shuffle
        indices = np.random.permutation(n_total)
        
        # Calculate split points
        n_train = int(n_total * train)
        n_val = int(n_total * val)
        
        # Split sequences
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Add to splits
        splits['train'].extend([(odd, sequences[i]) for i in train_idx])
        splits['val'].extend([(odd, sequences[i]) for i in val_idx])
        splits['test'].extend([(odd, sequences[i]) for i in test_idx])
    
    print(f"✓ Train: {len(splits['train'])} sequences")
    print(f"✓ Val: {len(splits['val'])} sequences")
    print(f"✓ Test: {len(splits['test'])} sequences")
    
    return splits

# Usage
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)
splits = create_data_splits(loader)

# Save to JSON for reproducibility
import json
with open('data_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)
```


## 🎨 Visualization

### Visualize Frame with Annotations

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_frame(loader, odd, sequence, frame_id, camera='mid_range'):
    """Visualize a frame with bounding boxes"""
    
    # Load complete frame
    frame_data = loader.load_complete_frame(odd, sequence, frame_id)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Display image
    img = frame_data['images'][camera]
    ax.imshow(img)
    
    # Draw traffic lights (red boxes)
    for obj in frame_data['annotations']['traffic_lights'].get('objects', []):
        # Note: You may need to project 3D bbox to 2D or use bbox_2d if available
        state = obj.get('state', 'unknown')
        distance = obj.get('distance', 0)
        ax.text(50, 50, f"🚦 {state} @ {distance:.1f}m", 
                color='red', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Draw traffic signs (blue boxes)
    for obj in frame_data['annotations']['traffic_signs'].get('objects', []):
        sign_type = obj.get('type', 'unknown')
        text = obj.get('text', '')
        ax.text(50, 100, f"🚸 {text or sign_type}", 
                color='blue', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f"{odd.upper()} - {camera.replace('_', ' ').title()} - Frame {frame_id}", 
                 fontsize=14, weight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Usage
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)
visualize_frame(loader, 'highway', '20231006-114522-00.15.00-00.15.15@Sogun', '0000001')
```


## 🔥 PyTorch Integration

### PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DualFoVTorchDataset(Dataset):
    """PyTorch Dataset for Dual-FoV dataset"""
    
    def __init__(self, loader, odds, camera='F_MIDRANGECAM_C', transform=None):
        self.loader = loader
        self.camera = camera
        self.transform = transform or transforms.ToTensor()
        
        # Build sample index
        self.samples = []
        for odd in odds:
            if odd in loader.sequences:
                for seq in loader.sequences[odd]:
                    frame_ids = loader.get_frame_ids(odd, seq, camera)
                    self.samples.extend([(odd, seq, fid) for fid in frame_ids])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        odd, seq, frame_id = self.samples[idx]
        
        # Load image
        img = self.loader.load_image(odd, seq, frame_id, self.camera)
        
        # Load annotations
        lights = self.loader.load_traffic_lights(odd, seq, frame_id)
        signs = self.loader.load_traffic_signs(odd, seq, frame_id)
        
        # Transform
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'traffic_lights': lights,
            'traffic_signs': signs,
            'metadata': {'odd': odd, 'sequence': seq, 'frame_id': frame_id}
        }

# Usage
loader = DualFoVDatasetLoader(root_dir=DATASET_PATH)

dataset = DualFoVTorchDataset(
    loader=loader,
    odds=['highway', 'urban'],
    transform=transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Test
for batch in dataloader:
    images = batch['image']  # [B, 3, 640, 640]
    print(f"✓ Batch loaded: {images.shape}")
    break
```


## 🛠️ Troubleshooting

### Common Issues

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Check your dataset path:
```python
# For Kaggle
DATASET_PATH = '/kaggle/input/dual-fov-for-traffic-light-and-sign-recognition'

# For local
DATASET_PATH = './dataset'

# Verify path exists
import os
print(os.path.exists(DATASET_PATH))
```


**Problem:** `KeyError: 'objects'` when accessing annotations

**Solution:** Check if objects key exists:
```python
objects = annotations.get('objects', [])  # Returns [] if key doesn't exist
```


**Problem:** Memory error when loading many frames

**Solution:** Use generator/iterator pattern:
```python
# Instead of loading all frames at once
for frame_data in loader.iterate_sequence(odd, sequence):
    # Process one frame at a time
    pass
```


**Problem:** Slow loading on Kaggle

**Solution:** Enable GPU and increase workers:
```python
dataloader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
```

```

**Paper:** [Link to Paper](https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense)  
**Code:** [GitHub Repository](https://github.com/abhishekjoshi007/Dual-FoV-Temporal-Robustness-for-Traffic-Light-and-Sign-Recognition-Hybrid-Attack-Defense)


## 📜 License

This dataset is released under **CC BY 4.0 License**.

You are free to:
- ✅ Share and redistribute
- ✅ Adapt and build upon  
- ✅ Use commercially

With attribution required.


## 🙏 Acknowledgments

This dataset integrates sequences from:
- **aiMotive 3D Traffic Light and Sign Dataset** (Kunsági-Máté et al., ECCV 2024)
- **Waymo Open Dataset** (Waymo LLC)
- **Udacity Self-Driving Car Dataset** (Udacity Inc.)
- **Texas Driving Sequences** (Original contribution, 165 sequences)

