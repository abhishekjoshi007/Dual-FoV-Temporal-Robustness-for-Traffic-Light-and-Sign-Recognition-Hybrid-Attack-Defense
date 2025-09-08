#!/usr/bin/env python3
"""
camera_filter.py

Prune sequences so that only F_LONGRANGECAM_C and F_MIDRANGECAM_C
remain under sensor/camera/. Safe against macOS '._*' sidecar files and
read-only files on external drives.
"""

from pathlib import Path
import os
import stat
import shutil

KEEP_DEFAULT = ("F_LONGRANGECAM_C", "F_MIDRANGECAM_C")

def _robust_unlink(path: Path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        # sidecar listed but already gone — ignore
        return
    except PermissionError:
        # clear read-only bit and retry
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
        try:
            os.unlink(path)
        except FileNotFoundError:
            return

def robust_rmtree(path: Path):
    """
    rmtree that tolerates missing '._*' AppleDouble files and read-only files.
    """
    if not path.exists() and not path.is_symlink():
        return
    # Walk bottom-up
    for root, dirs, files in os.walk(path, topdown=False):
        root_p = Path(root)
        # files
        for name in files:
            p = root_p / name
            _robust_unlink(p)
        # dirs
        for name in dirs:
            d = root_p / name
            try:
                os.rmdir(d)
            except FileNotFoundError:
                pass
            except PermissionError:
                os.chmod(d, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
                try:
                    os.rmdir(d)
                except FileNotFoundError:
                    pass
    # finally remove the root dir
    try:
        os.rmdir(path)
    except FileNotFoundError:
        pass
    except PermissionError:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        try:
            os.rmdir(path)
        except FileNotFoundError:
            pass

def filter_cameras_in_sequence(sequence_path: Path, keep=KEEP_DEFAULT):
    """
    Keep only specified camera folders inside `sensor/camera/` of one sequence.
    """
    cam_root = sequence_path / "sensor" / "camera"
    if not cam_root.exists():
        print(f"[warn] No camera folder at {cam_root}")
        return

    for cam in cam_root.iterdir():
        if cam.is_dir() and cam.name not in keep:
            print(f"[info] Removing unwanted camera: {cam}")
            robust_rmtree(cam)

    print(f"[done] {sequence_path.name}: kept only {', '.join(keep)}")

def filter_cameras_dataset(root_path: str, keep=KEEP_DEFAULT):
    """
    Batch mode: clean all sequences under AIMOTIVE_TL_TS_DATASET.
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    for odd_dir in root.iterdir():
        if not odd_dir.is_dir():
            continue
        for seq in odd_dir.iterdir():
            if seq.is_dir():
                filter_cameras_in_sequence(seq, keep=keep)

if __name__ == "__main__":
    # EDIT THIS to your dataset root
    dataset_root = "/Volumes/Untitled/Dataset"

    filter_cameras_dataset(
        dataset_root,
        keep=("F_LONGRANGECAM_C", "F_MIDRANGECAM_C")
    )
