#!/usr/bin/env python
"""Unified RF100 benchmark for RF-DETR.

Same four-step pipeline as the YOLO scripts, writing a standardized results JSON
shared with the other models in ``three_od_models/comparison_results``:

  1. zero-shot baseline validation stats (before fine-tuning)
  2. fine-tune for N epochs (``--epochs``, default 10)
  3. per-epoch validation curve (parsed from RF-DETR's ``metrics.csv``)
  4. final stats on the ``test`` split after fine-tuning

RF-DETR trains on COCO-format folders, so RF100 (YOLO format) is converted first
(cached under ``--coco-workdir``). RF-DETR has no ``.val()``, so baseline and test
mAP are computed with pycocotools over model predictions.

Pick the size variant with ``--variant`` (nano/small/medium/base/large). Each variant
has its own valid input resolution, so ``--resolution`` defaults to the variant's native
value (nano 384, small 512, medium 576, base 560, large 704) unless overridden. Runs and
results are namespaced by variant, so variants never overwrite each other:
  - runs   -> <output-root>/<variant>/<dataset>/
  - results-> ../comparison_results/rfdetr_<variant>_results.json

Run from THIS directory with THIS directory's venv, e.g.::

    cd three_od_models/rf-detr
    .venv/bin/python finetune_rf100.py --variant nano --epochs 10
    .venv/bin/python finetune_rf100.py --variant base --epochs 10 --skip-existing

``--skip-existing`` reuses any run whose ``metrics.csv`` already exists, computing only
the baseline + test eval.

Baseline mAP is expected ~0: COCO-pretrained class ids don't match the relabeled
RF100 class ids. Labeled a "zero-shot baseline", not an error.
"""
import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

DEFAULT_DATASETS = ["cable-damage", "bone-fracture-7fylg", "soda-bottles"]
DEFAULT_RF100_ROOT = Path("/home/arina_belova_jetbrains_com/roboflow-100-benchmark/rf100")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")

# RF-DETR metrics.csv column -> standardized curve key.
CURVE_COLUMNS = {
    "val/mAP_50": "mAP50",
    "val/mAP_50_95": "mAP50_95",
    "val/precision": "precision",
    "val/recall": "recall",
}

# Selectable RF-DETR size variants: name -> (rfdetr class name, native resolution).
# Each variant has its own valid resolution (patch_size * num_windows differs), so the
# default resolution follows the variant unless --resolution overrides it.
VARIANTS = {
    "nano": ("RFDETRNano", 384),
    "small": ("RFDETRSmall", 512),
    "medium": ("RFDETRMedium", 576),
    "base": ("RFDETRBase", 560),
    "large": ("RFDETRLarge", 704),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANTS), default="base", help="RF-DETR size variant.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override input resolution. Default = the variant's native resolution "
        "(nano 384, small 512, medium 576, base 560, large 704).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rf100-root", default=str(DEFAULT_RF100_ROOT))
    p.add_argument("--coco-workdir", default="./rf100_coco")
    p.add_argument(
        "--output-root",
        default="./rf100_rfdetr_runs",
        help="Runs are saved under <output-root>/<variant>/<dataset>.",
    )
    p.add_argument("--eval-threshold", type=float, default=0.05, help="Min confidence kept for COCO eval.")
    p.add_argument("--skip-existing", action="store_true", help="Reuse runs that already have metrics.csv.")
    p.add_argument(
        "--results-json",
        default=None,
        help="Defaults to ../comparison_results/rfdetr_<variant>_results.json",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# YOLO -> COCO conversion (adapted from rf-detr/test.ipynb)
# --------------------------------------------------------------------------- #
def read_yolo_names(data_yaml_path: Path) -> list[str]:
    lines = data_yaml_path.read_text().splitlines()
    names, in_names = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            after = stripped.split(":", 1)[1].strip()
            if after.startswith("[") and after.endswith("]"):
                return [x.strip().strip("'\"") for x in after[1:-1].split(",") if x.strip()]
            continue
        if in_names:
            if stripped.startswith("-"):
                names.append(stripped[1:].strip().strip("'\""))
            elif stripped and not line.startswith(" "):
                break
    if not names:
        raise ValueError(f"Could not parse class names from {data_yaml_path}")
    return names


def yolo_split_to_coco(dataset_dir: Path, split: str, out_split_dir: Path, categories: list[dict]) -> dict:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    out_split_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = image_id = 1
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        target = out_split_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for raw in label_path.read_text().splitlines():
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                class_idx = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                abs_w, abs_h = bw * width, bh * height
                x_min = max(0.0, min((xc * width) - abs_w / 2, float(width)))
                y_min = max(0.0, min((yc * height) - abs_h / 2, float(height)))
                abs_w = max(0.0, min(abs_w, float(width) - x_min))
                abs_h = max(0.0, min(abs_h, float(height) - y_min))
                if abs_w <= 0 or abs_h <= 0:
                    continue
                annotations.append({
                    "id": ann_id, "image_id": image_id, "category_id": class_idx + 1,
                    "bbox": [x_min, y_min, abs_w, abs_h], "area": abs_w * abs_h,
                    "iscrowd": 0, "segmentation": [],
                })
                ann_id += 1
        image_id += 1

    coco = {
        "info": {"description": f"{dataset_dir.name} {split} converted from RF100 YOLO"},
        "licenses": [], "categories": categories, "images": images, "annotations": annotations,
    }
    (out_split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    return {"images": len(images), "annotations": len(annotations)}


def convert_rf100_dataset_to_coco(dataset_name: str, rf100_root: Path, coco_workdir: Path) -> Path:
    dataset_dir = rf100_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing RF100 dataset: {dataset_dir}")
    out_dir = coco_workdir / dataset_name
    if (out_dir / ".conversion_complete").exists():
        return out_dir

    class_names = read_yolo_names(dataset_dir / "data.yaml")
    categories = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(class_names)]
    summary = {}
    for split in SPLITS:
        if not (dataset_dir / split).exists():
            print(f"  skipping missing split: {dataset_name}/{split}")
            continue
        summary[split] = yolo_split_to_coco(dataset_dir, split, out_dir / split, categories)
    (out_dir / ".conversion_complete").write_text(json.dumps(summary, indent=2))
    print(f"  converted {dataset_name} -> {out_dir}: {summary}")
    return out_dir


# --------------------------------------------------------------------------- #
# COCO evaluation via pycocotools
# --------------------------------------------------------------------------- #
def coco_eval(model, coco_json: Path, images_dir: Path, threshold: float,
              optimize: bool = True, warmup: int = 3) -> dict:
    """Run model.predict over a split, score with pycocotools, and time inference.

    RF-DETR predict returns 0-indexed class_id into the model's class list; our
    COCO categories are 1-indexed, so category_id = class_id + 1 (verified against
    a fine-tuned checkpoint). For the COCO-pretrained baseline the emitted ids are
    unrelated to the RF100 ids, so mAP resolves to ~0 as expected.

    optimize=True calls model.optimize_for_inference() before the predict loop
    (inference-only; do NOT use on a model you will train) so the measured speed
    reflects the optimized model. Per-image end-to-end predict() latency is timed
    at batch=1 with CUDA sync + warmup, and returned alongside mAP so speed and
    accuracy can be benchmarked together.
    """
    import time
    import statistics
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if optimize:
        model.optimize_for_inference()

    coco_gt = COCO(str(coco_json))
    images = coco_gt.dataset["images"]
    cuda = torch.cuda.is_available()

    # Warm up (trace / cudnn autotune) so timing excludes one-off startup costs.
    if images and warmup:
        warm_img = Image.open(images_dir / images[0]["file_name"]).convert("RGB")
        for _ in range(warmup):
            model.predict(warm_img, threshold=threshold)
        if cuda:
            torch.cuda.synchronize()

    detections = []
    latencies = []
    for img in images:
        # Force RGB: some images are grayscale/CMYK and RF-DETR.predict() rejects
        # non-3-channel inputs when given a raw path.
        image = Image.open(images_dir / img["file_name"]).convert("RGB")
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        det = model.predict(image, threshold=threshold)
        if cuda:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        for (x1, y1, x2, y2), conf, cls in zip(det.xyxy, det.confidence, det.class_id):
            detections.append({
                "image_id": img["id"],
                "category_id": int(cls) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(conf),
            })

    speed = {}
    if latencies:
        mean_s = statistics.mean(latencies)
        speed = {
            "latency_ms_mean": 1000 * mean_s,
            "latency_ms_median": 1000 * statistics.median(latencies),
            "fps": (1.0 / mean_s) if mean_s > 0 else float("inf"),
            "optimized": bool(optimize),
        }

    if not detections:
        return {"mAP50": 0.0, "mAP50_95": 0.0, **speed}
    coco_dt = coco_gt.loadRes(detections)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]), **speed}


# --------------------------------------------------------------------------- #
# metrics.csv -> per-epoch curve
# --------------------------------------------------------------------------- #
def _as_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_val_curve(metrics_csv: Path) -> list[dict]:
    """Collapse RF-DETR's sparse metrics.csv rows to one entry per epoch."""
    if not metrics_csv.exists():
        return []
    by_epoch = defaultdict(dict)
    with metrics_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            ep = _as_float(row.get("epoch"))
            if ep is None:
                continue
            ep = int(ep)
            for src in CURVE_COLUMNS:
                val = _as_float(row.get(src))
                if val is not None:
                    by_epoch[ep][src] = val
    curve = []
    for ep in sorted(by_epoch):
        entry = {"epoch": ep + 1}  # RF-DETR epochs are 0-indexed; +1 to match YOLO
        for src, dst in CURVE_COLUMNS.items():
            entry[dst] = by_epoch[ep].get(src)
        curve.append(entry)
    return curve


def load_results_json(path: Path, model_name: str, epochs: int, resolution: int) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            data.setdefault("datasets", {})
            data["model"] = model_name
            data["epochs"] = epochs
            data["imgsz"] = resolution
            return data
        except json.JSONDecodeError:
            pass
    return {
        "model": model_name,
        "framework": "rfdetr",
        "epochs": epochs,
        "imgsz": resolution,
        "datasets": {},
    }


def main():
    args = parse_args()
    import rfdetr

    class_name, native_res = VARIANTS[args.variant]
    ModelClass = getattr(rfdetr, class_name)
    model_name = f"rfdetr_{args.variant}"
    resolution = args.resolution or native_res  # variant-native resolution if not overridden

    rf100_root = Path(args.rf100_root)
    coco_workdir = Path(args.coco_workdir)
    # Runs live under <output-root>/<variant>/<dataset> so variants never collide.
    runs_root = Path(args.output_root) / args.variant
    results_path = Path(
        args.results_json
        or Path(__file__).resolve().parent.parent / "comparison_results" / f"{model_name}_results.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_results_json(results_path, model_name, args.epochs, resolution)

    def save():
        results_path.write_text(json.dumps(results, indent=2))

    print(f"variant={args.variant} ({class_name})  resolution={resolution}  runs_root={runs_root}")

    for ds in args.datasets:
        print(f"\n{'=' * 70}\n=== {model_name} :: {ds} ===\n{'=' * 70}")
        entry = {}

        # 0. Ensure COCO conversion exists.
        print(f"[{ds}] preparing COCO data ...")
        coco_dir = convert_rf100_dataset_to_coco(ds, rf100_root, coco_workdir)
        val_json, val_imgs = coco_dir / "valid" / "_annotations.coco.json", coco_dir / "valid"
        test_json, test_imgs = coco_dir / "test" / "_annotations.coco.json", coco_dir / "test"

        # 1. Zero-shot baseline on val (expected ~0).
        print(f"[{ds}] 1/4 zero-shot baseline val ...")
        entry["baseline_val"] = coco_eval(ModelClass(), val_json, val_imgs, args.eval_threshold)
        print(f"    baseline {entry['baseline_val']}")

        # 2. Fine-tune (or reuse existing run).
        output_dir = runs_root / ds
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = output_dir / "metrics.csv"
        if args.skip_existing and metrics_csv.exists():
            print(f"[{ds}] 2/4 reusing existing run at {output_dir}")
        else:
            print(f"[{ds}] 2/4 fine-tuning {args.epochs} epochs ...")
            ModelClass().train(
                dataset_dir=str(coco_dir),
                output_dir=str(output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
                lr=args.lr,
                resolution=resolution,
                early_stopping=True,
                checkpoint_interval=1,
            )
            metrics_csv = next(output_dir.rglob("metrics.csv"), metrics_csv)
        entry["run_dir"] = str(output_dir.resolve())

        # 3. Per-epoch validation curve from metrics.csv.
        print(f"[{ds}] 3/4 parsing val curve from {metrics_csv} ...")
        entry["val_curve"] = parse_val_curve(metrics_csv)
        print(f"    parsed {len(entry['val_curve'])} epochs")

        # 4. Final test-split eval with the best checkpoint.
        checkpoint = output_dir / "checkpoint_best_total.pth"
        print(f"[{ds}] 4/4 final test-split eval ({checkpoint.name}) ...")
        best_model = ModelClass(pretrain_weights=str(checkpoint))
        entry["final_test"] = coco_eval(best_model, test_json, test_imgs, args.eval_threshold)
        print(f"    test {entry['final_test']}")

        results["datasets"][ds] = entry
        save()
        print(f"[{ds}] done -> {results_path}")

    print(f"\nAll datasets complete. Results: {results_path}")


if __name__ == "__main__":
    main()
