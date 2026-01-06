import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path
import random
from collections import defaultdict


def create_splits(root_dir, output_dir, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create train/val/test splits for dual-FoV traffic dataset

    Args:
        root_dir: Root directory containing ODD-organized sequences
        output_dir: Directory to save splits.json
        train_ratio: Proportion of data for training (default: 0.6)
        val_ratio: Proportion of data for validation (default: 0.2)
        seed: Random seed for reproducibility (default: 42)

    Dataset Structure Expected:
        root_dir/
            highway/
                seq_001/
                    sensor/camera/
                        F_MIDRANGECAM_C/
                        F_LONGRANGECAM_C/
                seq_002/
                ...
            night/
            rainy/
            urban/

    Output:
        splits.json with format:
        {
            "train": ["path/to/seq1", "path/to/seq2", ...],
            "val": [...],
            "test": [...]
        }
    """
    random.seed(seed)

    root_path = Path(root_dir)
    odd_sequences = defaultdict(list)

    # Scan each ODD for valid sequences
    for odd in ['highway', 'night', 'rainy', 'urban']:
        odd_path = root_path / odd
        if not odd_path.exists():
            print(f"Warning: ODD directory '{odd}' not found, skipping...")
            continue

        for seq_folder in sorted(odd_path.iterdir()):
            if not seq_folder.is_dir():
                continue

            # Check for dual-FoV camera data
            sensor_path = seq_folder / 'sensor' / 'camera'
            if not sensor_path.exists():
                continue

            mid_cam = sensor_path / 'F_MIDRANGECAM_C'
            long_cam = sensor_path / 'F_LONGRANGECAM_C'

            # Verify both cameras exist with sufficient frames
            if mid_cam.exists() and long_cam.exists():
                mid_frames = list(mid_cam.glob('*.jpg')) + list(mid_cam.glob('*.png'))
                long_frames = list(long_cam.glob('*.jpg')) + list(long_cam.glob('*.png'))

                # Require at least 30 frames (1 second @ 30fps)
                if len(mid_frames) >= 30 and len(long_frames) >= 30:
                    odd_sequences[odd].append(str(seq_folder.relative_to(root_path)))

    # Split sequences per ODD to maintain balanced representation
    train_sequences = []
    val_sequences = []
    test_sequences = []

    for odd, sequences in odd_sequences.items():
        random.shuffle(sequences)
        n = len(sequences)

        # Calculate split indices
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test = n - n_train - n_val (implicit)

        train_sequences.extend(sequences[:n_train])
        val_sequences.extend(sequences[n_train:n_train+n_val])
        test_sequences.extend(sequences[n_train+n_val:])

    # Create output structure
    splits = {
        'train': sorted(train_sequences),
        'val': sorted(val_sequences),
        'test': sorted(test_sequences),
        'metadata': {
            'seed': seed,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': 1.0 - train_ratio - val_ratio,
        }
    }

    # Save splits
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Dataset splits created successfully!")
    print(f"{'='*60}")
    print(f"  Train: {len(train_sequences):4d} sequences ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_sequences):4d} sequences ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_sequences):4d} sequences ({(1-train_ratio-val_ratio)*100:.1f}%)")
    print(f"  Total: {len(train_sequences) + len(val_sequences) + len(test_sequences):4d} sequences")
    print(f"{'-'*60}")

    # Per-ODD breakdown
    print("Per-ODD Distribution:")
    for odd in ['highway', 'night', 'rainy', 'urban']:
        train_count = sum(1 for s in train_sequences if f'{odd}/' in s)
        val_count = sum(1 for s in val_sequences if f'{odd}/' in s)
        test_count = sum(1 for s in test_sequences if f'{odd}/' in s)
        total_count = train_count + val_count + test_count

        if total_count > 0:
            print(f"  {odd.capitalize():8s}: {total_count:3d} sequences "
                  f"(train: {train_count:3d}, val: {val_count:3d}, test: {test_count:3d})")

    print(f"{'='*60}")
    print(f"Splits saved to: {output_path / 'splits.json'}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")


def verify_splits(splits_file):
    """Verify splits.json integrity"""
    with open(splits_file, 'r') as f:
        splits = json.load(f)

    train = set(splits['train'])
    val = set(splits['val'])
    test = set(splits['test'])

    # Check for overlaps
    train_val_overlap = train & val
    train_test_overlap = train & test
    val_test_overlap = val & test

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("ERROR: Splits have overlapping sequences!")
        if train_val_overlap:
            print(f"  Train-Val overlap: {train_val_overlap}")
        if train_test_overlap:
            print(f"  Train-Test overlap: {train_test_overlap}")
        if val_test_overlap:
            print(f"  Val-Test overlap: {val_test_overlap}")
        return False

    print("✓ Splits verified: No overlapping sequences")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create train/val/test splits for dual-FoV traffic dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default='../Unified/Dataset/',
        help='Root directory containing ODD-organized sequences'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits',
        help='Output directory for splits.json'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.6,
        help='Proportion of data for training'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Proportion of data for validation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing splits.json instead of creating new one'
    )

    args = parser.parse_args()

    if args.verify:
        splits_file = Path(args.output_dir) / 'splits.json'
        if not splits_file.exists():
            print(f"ERROR: Splits file not found: {splits_file}")
            sys.exit(1)
        verify_splits(splits_file)
    else:
        # Validate ratios
        if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1):
            print("ERROR: train_ratio and val_ratio must be between 0 and 1")
            sys.exit(1)

        if args.train_ratio + args.val_ratio >= 1.0:
            print("ERROR: train_ratio + val_ratio must be < 1.0")
            sys.exit(1)

        create_splits(
            args.root_dir,
            args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

        # Auto-verify created splits
        splits_file = Path(args.output_dir) / 'splits.json'
        if splits_file.exists():
            verify_splits(splits_file)
