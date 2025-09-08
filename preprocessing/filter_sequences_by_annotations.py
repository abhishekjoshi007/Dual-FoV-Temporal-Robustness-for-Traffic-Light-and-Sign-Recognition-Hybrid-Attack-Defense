#!/usr/bin/env python3
"""
filter_sequences_by_annotations.py

Copy/symlink only those aiMotive sequences that match annotation criteria.

Features:
- box_mode=auto: prefer 2d_body if present in the sequence, else fall back to 3d_body
- require=both|any: require TL & TS or TL OR TS
- min_frames_tl / min_frames_ts: require a minimum number of frames with boxes
- Verbose counting: prints per-sequence counts for easy debugging
- Symlink by default (falls back to copy if symlink not allowed)

Usage:
  python3 filter_sequences_by_annotations.py
  (Edit dataset_root and out_root at the bottom)
"""

import os, json, shutil
from pathlib import Path
from typing import Tuple, Optional

ODDS_DEFAULT = ("highway", "night", "rainy", "urban")

def count_nonempty_jsons(dir_path: Path) -> int:
    """
    Count files in dir_path with a non-empty list/dict JSON.
    """
    if not dir_path.exists():
        return 0
    count = 0
    for p in dir_path.iterdir():
        if p.suffix.lower() != ".json" or p.name.startswith("._"):
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if (isinstance(data, list) and len(data) > 0) or (isinstance(data, dict) and len(data) > 0):
                count += 1
        except Exception:
            # unreadable JSON -> skip
            continue
    return count

def pick_box_type_for_sequence(seq: Path, box_mode: str) -> Optional[str]:
    """
    Decide which box_type directory to use for this sequence.
    - auto: prefer 2d_body if present, else 3d_body if present
    - 2d_body / 3d_body: use that, return None if missing
    """
    if box_mode == "auto":
        d2 = seq / "traffic_light" / "box" / "2d_body"
        d3 = seq / "traffic_light" / "box" / "3d_body"
        if d2.exists():
            return "2d_body"
        if d3.exists():
            return "3d_body"
        return None
    else:
        # fixed choice
        d = seq / "traffic_light" / "box" / box_mode
        return box_mode if d.exists() else None

def compute_counts(seq: Path, box_type: str) -> Tuple[int, int]:
    """
    Return (#frames_with_TL_boxes, #frames_with_TS_boxes) for the given box_type.
    """
    tl_dir = seq / "traffic_light" / "box" / box_type
    ts_dir = seq / "traffic_sign"  / "box" / box_type
    tl_count = count_nonempty_jsons(tl_dir)
    ts_count = count_nonempty_jsons(ts_dir)
    return tl_count, ts_count

def sequence_qualifies(seq: Path,
                       box_mode: str = "auto",
                       require: str = "both",
                       min_frames_tl: int = 1,
                       min_frames_ts: int = 1) -> Tuple[bool, str, int, int]:
    """
    Decide if a sequence qualifies. Returns (qualifies, used_box_type, tl_count, ts_count).
    - box_mode: "auto" | "2d_body" | "3d_body"
    - require:  "both" | "any"
    """
    used = pick_box_type_for_sequence(seq, box_mode)
    if used is None:
        return False, "none", 0, 0

    tl_count, ts_count = compute_counts(seq, used)

    if require == "both":
        ok = (tl_count >= min_frames_tl) and (ts_count >= min_frames_ts)
    else:  # "any"
        ok = (tl_count >= min_frames_tl) or (ts_count >= min_frames_ts)

    return ok, used, tl_count, ts_count

def safe_symlink_dir(src: Path, dst: Path):
    """
    Mirror a directory using symlinks: create directories, and symlink files.
    Fallback to copy file-by-file if symlinks are not allowed.
    """
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (target_dir / d).mkdir(exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = target_dir / f
            if dp.exists() or dp.is_symlink():
                continue
            try:
                os.symlink(sp, dp)
            except OSError:
                shutil.copy2(sp, dp)

def copy_sequence(src_seq: Path, dst_seq: Path, mode: str = "symlink"):
    dst_seq.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        if dst_seq.exists():
            shutil.rmtree(dst_seq)
        shutil.copytree(src_seq, dst_seq)
    else:
        dst_seq.mkdir(parents=True, exist_ok=True)
        safe_symlink_dir(src_seq, dst_seq)

def filter_sequences(dataset_root: str,
                     out_root: str,
                     odds = ODDS_DEFAULT,
                     box_mode: str = "auto",
                     require: str = "both",
                     min_frames_tl: int = 1,
                     min_frames_ts: int = 1,
                     mode: str = "symlink",
                     dry_run: bool = False) -> None:
    """
    Main filter: scans ODDs, evaluates each sequence, saves qualifying ones.
    """
    root = Path(dataset_root)
    out = Path(out_root)
    if not root.exists():
        raise SystemExit(f"[error] dataset_root not found: {root}")

    scanned = kept = 0
    print(f"[info] scan root={root}")
    print(f"[info] options: box_mode={box_mode}, require={require}, min_frames_tl={min_frames_tl}, min_frames_ts={min_frames_ts}, mode={mode}, dry_run={dry_run}")

    for odd in odds:
        odd_dir = root / odd
        if not odd_dir.exists():
            print(f"[warn] ODD missing: {odd_dir}")
            continue

        for seq in sorted([p for p in odd_dir.iterdir() if p.is_dir()]):
            scanned += 1
            ok, used, tlc, tsc = sequence_qualifies(
                seq, box_mode=box_mode, require=require,
                min_frames_tl=min_frames_tl, min_frames_ts=min_frames_ts
            )
            used_disp = used if used in ("2d_body", "3d_body") else "none"
            if ok:
                kept += 1
                dst_seq = out / odd / seq.name
                if dry_run:
                    print(f"[keep] {seq.name}  TL={tlc} TS={tsc}  (box_type={used_disp})  ->  {dst_seq}")
                else:
                    copy_sequence(seq, dst_seq, mode=mode)
                    print(f"[saved] {odd}/{seq.name}  TL={tlc} TS={tsc}  (box_type={used_disp})")
            else:
                print(f"[skip]  {odd}/{seq.name}  TL={tlc} TS={tsc}  (box_type={used_disp})")

    print(f"[done] scanned={scanned}, kept={kept}, out_root={out}")

# ------------- entrypoint (edit paths below) -------------
if __name__ == "__main__":
    # EDIT THESE:
    dataset_root = "/Volumes/Untitled/aimotive_tl_ts_dataset"
    out_root     = "/Volumes/Untitled/aimotive_filter"

    # Behaviour knobs:
    box_mode = "auto"        # "auto" | "2d_body" | "3d_body"
    require  = "both"        # "both" | "any"
    min_frames_tl = 1        # require TL boxes in >= this many frames
    min_frames_ts = 1        # require TS boxes in >= this many frames
    mode     = "symlink"     # "symlink" | "copy"
    dry_run  = False         # True = print only

    filter_sequences(
        dataset_root=dataset_root,
        out_root=out_root,
        odds=ODDS_DEFAULT,
        box_mode=box_mode,
        require=require,
        min_frames_tl=min_frames_tl,
        min_frames_ts=min_frames_ts,
        mode=mode,
        dry_run=dry_run
    )
