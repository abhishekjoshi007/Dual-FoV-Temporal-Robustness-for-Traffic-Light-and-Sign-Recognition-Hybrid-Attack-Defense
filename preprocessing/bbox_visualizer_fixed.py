import argparse
import os
import shlex
import subprocess
from pathlib import Path

def run(cmd: str):
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def main():
    p = argparse.ArgumentParser(description="Render TL/TS boxes on camera frames")
    p.add_argument("data_root", help="Path to your local aiMotive TL/TS dataset root")
    p.add_argument("--aimotive_repo", default="../aimotive-dataset-loader",
                   help="Path to the cloned aimotive-dataset-loader repo")
    p.add_argument("--odd", nargs="+", default=["highway"],
                   help="ODD subset(s): highway night rainy urban")
    p.add_argument("--sequence", default=None,
                   help="Optional single sequence name, e.g. 20231006-114522-00.15.00-00.15.15@Sogun")
    p.add_argument("--cameras", nargs="+",
                   default=["F_LONGRANGECAM_C", "F_MIDRANGECAM_C"],
                   help="Camera names to render")
    p.add_argument("--objects", nargs="+", default=["traffic_light", "traffic_sign"],
                   choices=["traffic_light", "traffic_sign"],
                   help="Which object types to draw")
    p.add_argument("--save_dir", default="out_renders",
                   help="Where to put the rendered PNGs")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing outputs")
    args = p.parse_args()

    repo = Path(args.aimotive_repo).resolve()
    example = repo / "examples" / "example_render.py"
    if not example.exists():
        raise FileNotFoundError(f"Could not find example_render.py at {example}")

    # Ensure output folder exists
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build base command that we’ll extend
    base = (
        f"PYTHONPATH={shlex.quote(str(repo))} "
        f"python {shlex.quote(str(example))} "
        f"--data-root {shlex.quote(args.data_root)} "
        + ("--sequence " + shlex.quote(args.sequence) + " " if args.sequence else "")
        + ("--overwrite " if args.overwrite else "")
        + ("--cameras " + " ".join(map(shlex.quote, args.cameras)) + " ")
    )

    # Loop ODD x object type
    for odd in args.odd:
        for obj in args.objects:
            out_dir = Path(args.save_dir) / f"{odd}_{obj}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = (
                base
                + f"--odd {shlex.quote(odd)} "
                + f"--object-type {shlex.quote(obj)} "
                + f"--save-dir {shlex.quote(str(out_dir))}"
            )
            run(cmd)

if __name__ == "__main__":
    main()
