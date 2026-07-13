#!/usr/bin/env python
"""Unified RF100 benchmark for an Ultralytics YOLO model (YOLOv12 / YOLO26).

For every dataset (treated separately, no combined model) this runs the same
four-step pipeline and writes a standardized results JSON shared with the other
models in ``three_od_models/comparison_results``:

  1. zero-shot baseline validation stats (before fine-tuning)
  2. fine-tune for N epochs (``--epochs``, default 10)
  3. per-epoch validation curve (parsed from the run's ``results.csv``)
  4. final stats on the ``test`` split after fine-tuning

Run it from THIS directory with THIS directory's venv, e.g.::

    cd three_od_models/yolov12
    .venv/bin/python finetune_rf100.py --epochs 10

Note: baseline mAP is expected to be ~0. The COCO-pretrained weights emit COCO
class ids, while each RF100 dataset reuses those ids for different classes, so
zero-shot predictions never match the ground-truth class ids. This is labeled a
"zero-shot baseline", not an error.
"""
import argparse
import csv
import json
import os
from pathlib import Path

DEFAULT_DATASETS = ["cable-damage", "bone-fracture-7fylg", "soda-bottles"]
DEFAULT_RF100_ROOT = "/home/arina_belova_jetbrains_com/roboflow-100-benchmark/rf100"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"val": "valid", "test": "test"}  # our split name -> RF100 folder name

# Ultralytics results.csv column names -> our standardized keys.
CURVE_COLUMNS = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo26m.pt", help="Pretrained weights to fine-tune from.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="RF100 dataset folder names.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--rf100-root", default=DEFAULT_RF100_ROOT)
    p.add_argument("--project", default="rf100_runs_10ep", help="Ultralytics project dir for fresh runs.")
    p.add_argument(
        "--results-json",
        default=None,
        help="Where to write the standardized results JSON. "
        "Defaults to ../comparison_results/<model_stem>_results.json",
    )
    p.add_argument("--skip-existing", action="store_true", help="Reuse a run dir that already has results.csv.")
    return p.parse_args()


def load_results_json(path: Path, model_stem: str, epochs: int, imgsz: int) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            data.setdefault("datasets", {})
            data["epochs"] = epochs
            data["imgsz"] = imgsz
            return data
        except json.JSONDecodeError:
            pass
    return {
        "model": model_stem,
        "framework": "ultralytics",
        "epochs": epochs,
        "imgsz": imgsz,
        "datasets": {},
    }


def parse_val_curve(results_csv: Path) -> list[dict]:
    """Parse an Ultralytics results.csv into one standardized dict per epoch."""
    curve = []
    if not results_csv.exists():
        return curve
    with results_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            entry = {"epoch": int(float(row["epoch"]))}
            for src, dst in CURVE_COLUMNS.items():
                val = row.get(src, "")
                entry[dst] = float(val) if val not in ("", None) else None
            curve.append(entry)
    return curve


def collect_split_images(rf100_root: str, ds: str, split: str) -> list[Path]:
    d = Path(rf100_root) / ds / SPLIT_DIRS[split] / "images"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def measure_speed(model, images: list[Path], imgsz: int, device: str, warmup: int = 3) -> dict:
    """Time per-image predict() at batch=1 with warmup + CUDA sync.

    Matches the RF-DETR benchmark's timing (end-to-end predict latency) so FPS is
    comparable across models. Returns {} if there are no images.
    """
    import time
    import statistics
    import torch

    if not images:
        return {}
    cuda = str(device) != "cpu" and torch.cuda.is_available()
    for _ in range(warmup):
        model.predict(str(images[0]), imgsz=imgsz, device=device, verbose=False)
    if cuda:
        torch.cuda.synchronize()

    latencies = []
    for p in images:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=imgsz, device=device, verbose=False)
        if cuda:
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    mean_s = statistics.mean(latencies)
    return {
        "latency_ms_mean": 1000 * mean_s,
        "latency_ms_median": 1000 * statistics.median(latencies),
        "fps": (1.0 / mean_s) if mean_s > 0 else float("inf"),
    }


def main():
    args = parse_args()
    from ultralytics import YOLO

    model_stem = Path(args.model).stem
    results_path = Path(
        args.results_json
        or Path(__file__).resolve().parent.parent / "comparison_results" / f"{model_stem}_results.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_results_json(results_path, model_stem, args.epochs, args.imgsz)

    def save():
        results_path.write_text(json.dumps(results, indent=2))

    for ds in args.datasets:
        yaml_path = Path(args.rf100_root) / ds / "data.yaml"
        if not yaml_path.exists():
            print(f"!! skipping {ds}: missing {yaml_path}")
            continue
        print(f"\n{'=' * 70}\n=== {model_stem} :: {ds} ===\n{'=' * 70}")
        entry = {}

        # 1. Zero-shot baseline on the val split (expected ~0).
        print(f"[{ds}] 1/4 zero-shot baseline val ...")
        base_model = YOLO(args.model)
        base = base_model.val(
            data=str(yaml_path), split="val", imgsz=args.imgsz, device=args.device, verbose=False
        )
        entry["baseline_val"] = {"mAP50": float(base.box.map50), "mAP50_95": float(base.box.map)}
        entry["baseline_val"].update(
            measure_speed(base_model, collect_split_images(args.rf100_root, ds, "val"), args.imgsz, args.device)
        )
        print(f"    baseline mAP50={base.box.map50:.4f}  mAP50-95={base.box.map:.4f}"
              f"  ({entry['baseline_val'].get('fps', float('nan')):.1f} FPS)")

        # 2. Fine-tune for N epochs.
        run_name = f"{model_stem}_{ds}"
        run_dir = Path(args.project) / run_name
        results_csv = run_dir / "results.csv"
        if args.skip_existing and results_csv.exists():
            print(f"[{ds}] 2/4 reusing existing run at {run_dir}")
        else:
            print(f"[{ds}] 2/4 fine-tuning {args.epochs} epochs ...")
            train_model = YOLO(args.model)
            train_res = train_model.train(
                data=str(yaml_path),
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=run_name,
                exist_ok=True,
                verbose=False,
            )
            run_dir = Path(train_res.save_dir)
            results_csv = run_dir / "results.csv"
        entry["run_dir"] = str(run_dir.resolve())

        # 3. Per-epoch validation curve.
        print(f"[{ds}] 3/4 parsing val curve from {results_csv.name} ...")
        entry["val_curve"] = parse_val_curve(results_csv)
        print(f"    parsed {len(entry['val_curve'])} epochs")

        # 4. Final stats on the test split with the best checkpoint.
        best = run_dir / "weights" / "best.pt"
        print(f"[{ds}] 4/4 final test-split eval ({best.name}) ...")
        best_model = YOLO(str(best))
        test = best_model.val(
            data=str(yaml_path), split="test", imgsz=args.imgsz, device=args.device, verbose=False
        )
        entry["final_test"] = {"mAP50": float(test.box.map50), "mAP50_95": float(test.box.map)}
        entry["final_test"].update(
            measure_speed(best_model, collect_split_images(args.rf100_root, ds, "test"), args.imgsz, args.device)
        )
        print(f"    test mAP50={test.box.map50:.4f}  mAP50-95={test.box.map:.4f}"
              f"  ({entry['final_test'].get('fps', float('nan')):.1f} FPS)")

        results["datasets"][ds] = entry
        save()
        print(f"[{ds}] done -> {results_path}")

    print(f"\nAll datasets complete. Results: {results_path}")


if __name__ == "__main__":
    main()
