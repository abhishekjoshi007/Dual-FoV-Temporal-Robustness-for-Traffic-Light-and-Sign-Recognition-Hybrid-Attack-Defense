#!/usr/bin/env python3
"""
prune_sequences_by_annotations.py

Filter IN-PLACE an aiMotive dataset so only sequences that match your
annotation criteria remain. Non-qualifying sequences are deleted or moved.

Features:
- box_mode=auto: prefer 2d_body if present; otherwise fall back to 3d_body
- require=both|any: TL∧TS or TL∨TS
- min_frames_tl / min_frames_ts thresholds
- Robust deletion that tolerates macOS `._*` sidecars and read-only files
- Optional quarantine: move non-qualifiers to a backup folder instead of deleting
- Dry run ON by default (safety)

Usage:
  python3 prune_sequences_by_annotations.py
  (Edit dataset_root and options in the entrypoint below)
"""

import os, json, shutil, stat
from pathlib import Path
from typing import Tuple, Optional

ODDS_DEFAULT = ("highway", "night", "rainy", "urban")

# ---------- annotation scanning ----------
def count_nonempty_jsons(dir_path: Path) -> int:
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
    if box_mode == "auto":
        d2 = seq / "traffic_light" / "box" / "2d_body"
        d3 = seq / "traffic_light" / "box" / "3d_body"
        if d2.exists(): return "2d_body"
        if d3.exists(): return "3d_body"
        return None
    d = seq / "traffic_light" / "box" / box_mode
    return box_mode if d.exists() else None

def compute_counts(seq: Path, box_type: str) -> Tuple[int, int]:
    tl_dir = seq / "traffic_light" / "box" / box_type
    ts_dir = seq / "traffic_sign"  / "box" / box_type
    return count_nonempty_jsons(tl_dir), count_nonempty_jsons(ts_dir)

def sequence_qualifies(seq: Path,
                       box_mode: str = "auto",
                       require: str = "both",
                       min_frames_tl: int = 1,
                       min_frames_ts: int = 1) -> Tuple[bool, str, int, int]:
    used = pick_box_type_for_sequence(seq, box_mode)
    if used is None:
        return False, "none", 0, 0
    tl_count, ts_count = compute_counts(seq, used)
    if require == "both":
        ok = (tl_count >= min_frames_tl) and (ts_count >= min_frames_ts)
    else:
        ok = (tl_count >= min_frames_tl) or (ts_count >= min_frames_ts)
    return ok, used, tl_count, ts_count

# ---------- robust deletion / move ----------
def _chmod_w(p: Path):
    try:
        mode = p.stat().st_mode
        p.chmod(mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
    except Exception:
        pass

def robust_rmtree(path: Path):
    """Remove a directory tree, tolerating macOS '._*' files and read-only entries."""
    if not path.exists() and not path.is_symlink():
        return
    for root, dirs, files in os.walk(path, topdown=False):
        root_p = Path(root)
        for name in files:
            p = root_p / name
            if name.startswith("._"):
                # Ignore ghost AppleDouble entries
                try:
                    p.unlink(missing_ok=True)
                except TypeError:
                    try:
                        if p.exists():
                            p.unlink()
                    except FileNotFoundError:
                        pass
                continue
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except PermissionError:
                _chmod_w(p)
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        for name in dirs:
            d = root_p / name
            try:
                d.rmdir()
            except FileNotFoundError:
                pass
            except PermissionError:
                _chmod_w(d)
                try:
                    d.rmdir()
                except FileNotFoundError:
                    pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass
    except PermissionError:
        _chmod_w(path)
        try:
            path.rmdir()
        except FileNotFoundError:
            pass

def move_to_quarantine(seq: Path, quarantine_root: Path):
    dst = quarantine_root / seq.parent.name / seq.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use shutil.move to handle cross-device moves
    shutil.move(str(seq), str(dst))
    return dst

# ---------- main logic ----------
def prune_sequences(dataset_root: str,
                    odds = ODDS_DEFAULT,
                    box_mode: str = "auto",
                    require: str = "both",
                    min_frames_tl: int = 1,
                    min_frames_ts: int = 1,
                    action: str = "delete",           # "delete" | "move"
                    quarantine_root: Optional[str] = None,
                    dry_run: bool = True) -> None:
    """
    Filter in place. Remove or move sequences that DO NOT meet criteria.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise SystemExit(f"[error] dataset_root not found: {root}")

    print(f"[info] root={root}")
    print(f"[info] options: box_mode={box_mode}, require={require}, min_tl={min_frames_tl}, min_ts={min_frames_ts}, action={action}, dry_run={dry_run}")

    if action == "move":
        if not quarantine_root:
            raise SystemExit("[error] action='move' requires quarantine_root")
        quarantine_root = Path(quarantine_root)
        quarantine_root.mkdir(parents=True, exist_ok=True)

    scanned = kept = removed = 0

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
                print(f"[keep] {odd}/{seq.name}  TL={tlc} TS={tsc}  (box_type={used_disp})")
                continue

            # Non-qualifier: delete/move
            removed += 1
            if dry_run:
                print(f"[would remove] {odd}/{seq.name}  TL={tlc} TS={tsc}  (box_type={used_disp})")
            else:
                if action == "delete":
                    print(f"[remove] {odd}/{seq.name}")
                    robust_rmtree(seq)
                elif action == "move":
                    dst = move_to_quarantine(seq, quarantine_root)
                    print(f"[moved] {odd}/{seq.name} -> {dst}")
                else:
                    raise ValueError("action must be 'delete' or 'move'")

    print(f"[done] scanned={scanned}, kept={kept}, removed={removed}")
    if dry_run:
        print("[note] dry run: no changes were made. Set dry_run=False to apply.")

# ------------- entrypoint (edit paths below) -------------
if __name__ == "__main__":
    # EDIT THIS to your dataset root
    dataset_root = "/Volumes/Untitled/Dataset"

    # Behaviour knobs
    box_mode = "auto"        # "auto" | "2d_body" | "3d_body"
    require  = "both"        # "both" | "any"
    min_frames_tl = 1
    min_frames_ts = 1

    # What to do with non-qualifying sequences:
    #   action="delete"  -> permanently delete (fastest, frees space)
    #   action="move"    -> move to a quarantine folder for review/undo
    action = "move"
    quarantine_root = "/Volumes/Untitled/aimotive_quarantine"  # used only if action="move"

    # Safety:
    dry_run = True  # <-- start with True; switch to False when you're sure

    prune_sequences(
        dataset_root=dataset_root,
        odds=ODDS_DEFAULT,
        box_mode=box_mode,
        require=require,
        min_frames_tl=min_frames_tl,
        min_frames_ts=min_frames_ts,
        action=action,
        quarantine_root=quarantine_root,
        dry_run=dry_run
    )
