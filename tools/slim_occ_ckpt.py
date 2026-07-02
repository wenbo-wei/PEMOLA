"""Slim occlusion-classifier checkpoints down to just the model weights.

Each training-time .pth saved by `occ_cls_train.save_checkpoint` contains:
    {'model', 'optimizer', 'lr_scheduler', 'scaler', 'epoch', 'max_accuracy', 'config'}

For release / inference only `model` is needed. This script rewrites the
checkpoint(s) to keep just `{'model': state_dict}`.

Usage:
    # default: scan output/occ_classifier/*/ep*.pth, write *_slim.pth alongside
    python tools/slim_occ_ckpt.py

    # in-place (overwrites originals, after creating .bak):
    python tools/slim_occ_ckpt.py --inplace

    # explicit files:
    python tools/slim_occ_ckpt.py path/to/a.pth path/to/b.pth -o slim_dir/

    # drop the outer dict and save the raw state_dict instead:
    python tools/slim_occ_ckpt.py --raw
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = str(PROJECT_ROOT / "output" / "occ_classifier" / "*" / "ep*.pth")


def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def slim_one(src: Path, dst: Path, raw: bool, dry_run: bool) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        print(f"[skip] {src}: no 'model' key (keys={list(ckpt.keys())})")
        return

    state_dict = ckpt["model"]
    slim = state_dict if raw else {"model": state_dict}

    orig_size = src.stat().st_size
    if dry_run:
        print(f"[dry] {src} ({human_size(orig_size)}) -> {dst} "
              f"keys_kept={'<raw state_dict>' if raw else list(slim.keys())}")
        return

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    torch.save(slim, tmp)
    tmp.replace(dst)
    new_size = dst.stat().st_size
    saved = orig_size - new_size
    print(f"[done] {src.name}: {human_size(orig_size)} -> {human_size(new_size)} "
          f"(saved {human_size(saved)}, {saved / orig_size * 100:.1f}%)  ->  {dst}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*",
                   help="Checkpoint files. If omitted, defaults to "
                        f"{DEFAULT_GLOB}")
    p.add_argument("-o", "--output-dir", type=str, default=None,
                   help="Write slimmed checkpoints into this directory "
                        "(filename appended with '_slim' unless --inplace).")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite the original .pth files. A .bak copy is "
                        "made first unless --no-backup is given.")
    p.add_argument("--no-backup", action="store_true",
                   help="With --inplace, skip creating .bak copies.")
    p.add_argument("--raw", action="store_true",
                   help="Save the raw state_dict directly (drop the outer "
                        "{'model': ...} wrapper). Loaders that expect "
                        "checkpoint['model'] will need updating.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen without writing files.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = [Path(p) for p in sorted(glob.glob(DEFAULT_GLOB))]

    if not files:
        print(f"No checkpoints found (looked at: {DEFAULT_GLOB})")
        return

    if args.inplace and args.output_dir:
        raise SystemExit("--inplace and --output-dir are mutually exclusive.")

    for src in files:
        if not src.exists():
            print(f"[skip] {src}: does not exist")
            continue

        if args.inplace:
            if not args.no_backup and not args.dry_run:
                bak = src.with_suffix(src.suffix + ".bak")
                if not bak.exists():
                    shutil.copy2(src, bak)
                    print(f"[bak]  {src} -> {bak}")
            dst = src
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / src.name
        else:
            dst = src.with_name(src.stem + "_slim" + src.suffix)

        slim_one(src, dst, raw=args.raw, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
