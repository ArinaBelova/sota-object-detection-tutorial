#!/usr/bin/env python
"""Render annotated test-image predictions for an Ultralytics YOLO model.

Inference-only (no training): loads the fine-tuned best.pt for each dataset and
draws predicted boxes on the first K test images, saving them to a shared folder
so compare_models.ipynb can display a models x images grid.

Run from THIS directory with THIS directory's venv, after fine-tuning:

    .venv/bin/python render_predictions.py --samples 6

The same first-K test images (sorted by filename) are used across all models, so
the grids line up for side-by-side comparison.
"""
import argparse
from pathlib import Path

import cv2

DEFAULT_DATASETS = ["cable-damage", "bone-fracture-7fylg", "soda-bottles"]
DEFAULT_RF100_ROOT = "/home/arina_belova_jetbrains_com/roboflow-100-benchmark/rf100"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26m.pt", help="Only used to derive the run-dir name stem.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--samples", type=int, default=6, help="How many test images per dataset to render.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for drawing boxes.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--rf100-root", default=DEFAULT_RF100_ROOT)
    p.add_argument("--project", default="runs/detect/rf100_runs_10ep", help="Where fine-tuned runs live.")
    p.add_argument("--out", default=None, help="Defaults to ../comparison_results/pred_viz")
    return p.parse_args()


def sample_test_images(rf100_root: Path, ds: str, k: int) -> list[Path]:
    img_dir = rf100_root / ds / "test" / "images"
    if not img_dir.is_dir():
        return []
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return imgs[:k]


def main():
    args = parse_args()
    from ultralytics import YOLO

    model_stem = Path(args.model).stem
    out_root = Path(args.out or Path(__file__).resolve().parent.parent / "comparison_results" / "pred_viz")
    rf100_root = Path(args.rf100_root)

    for ds in args.datasets:
        best = Path(args.project) / f"{model_stem}_{ds}" / "weights" / "best.pt"
        if not best.exists():
            print(f"!! skipping {ds}: no fine-tuned weights at {best} (train first)")
            continue
        images = sample_test_images(rf100_root, ds, args.samples)
        if not images:
            print(f"!! skipping {ds}: no test images found")
            continue

        out_dir = out_root / model_stem / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(best))
        print(f"[{ds}] rendering {len(images)} images with {best} -> {out_dir}")
        for img_path in images:
            res = model.predict(str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
            annotated = res[0].plot()  # BGR ndarray with boxes+labels drawn
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), annotated)
    print(f"\nDone. Annotated images under {out_root}")


if __name__ == "__main__":
    main()
