#!/usr/bin/env python
"""Render annotated test-image predictions for RF-DETR.

Inference-only: loads each dataset's fine-tuned checkpoint and draws predicted
boxes on the first K test images, saving them to the shared pred_viz folder so
compare_models.ipynb can display a models x images grid.

Run from THIS directory with THIS directory's venv:

    .venv/bin/python render_predictions.py --samples 6

Uses the same first-K test images (sorted by filename) as the YOLO renderers, so
the grids line up across models.
"""
import argparse
from pathlib import Path

import cv2
import supervision as sv
from PIL import Image

DEFAULT_DATASETS = ["cable-damage", "bone-fracture-7fylg", "soda-bottles"]
DEFAULT_RF100_ROOT = "/home/arina_belova_jetbrains_com/roboflow-100-benchmark/rf100"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Must match finetune_rf100.py's VARIANTS.
VARIANT_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANT_CLASSES), default="base", help="RF-DETR size variant.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--rf100-root", default=DEFAULT_RF100_ROOT)
    p.add_argument(
        "--output-root",
        default="./rf100_rfdetr_runs",
        help="Runs are read from <output-root>/<variant>/<dataset>.",
    )
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
    import rfdetr

    ModelClass = getattr(rfdetr, VARIANT_CLASSES[args.variant])
    model_stem = f"rfdetr_{args.variant}"

    out_root = Path(args.out or Path(__file__).resolve().parent.parent / "comparison_results" / "pred_viz")
    rf100_root = Path(args.rf100_root)
    runs_root = Path(args.output_root) / args.variant

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for ds in args.datasets:
        checkpoint = runs_root / ds / "checkpoint_best_total.pth"
        if not checkpoint.exists():
            print(f"!! skipping {ds}: no checkpoint at {checkpoint} (train first)")
            continue
        images = sample_test_images(rf100_root, ds, args.samples)
        if not images:
            print(f"!! skipping {ds}: no test images found")
            continue

        out_dir = out_root / model_stem / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        model = ModelClass(pretrain_weights=str(checkpoint))
        print(f"[{ds}] rendering {len(images)} images with {checkpoint} -> {out_dir}")
        for img_path in images:
            det = model.predict(Image.open(img_path).convert("RGB"), threshold=args.conf)
            names = det.data.get("class_name") if det.data else None
            if names is not None:
                labels = [f"{n} {c:.2f}" for n, c in zip(names, det.confidence)]
            else:
                labels = [f"{int(cid)} {c:.2f}" for cid, c in zip(det.class_id, det.confidence)]
            scene = cv2.imread(str(img_path))
            scene = box_annotator.annotate(scene=scene, detections=det)
            scene = label_annotator.annotate(scene=scene, detections=det, labels=labels)
            cv2.imwrite(str(out_dir / f"{img_path.stem}.png"), scene)
    print(f"\nDone. Annotated images under {out_root}")


if __name__ == "__main__":
    main()
