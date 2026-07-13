#!/usr/bin/env python
"""Prepare COCO val2017 once, shared by all three models' notebooks.

Downloads ONLY the val split (not the 18 GB train split):
  - val2017 images         (~780 MB)  -> {root}/images/val2017/
  - instances_val2017.json (~240 MB zip) -> {root}/annotations/instances_val2017.json

Then derives Ultralytics-style YOLO labels and a val-only data yaml so YOLO's
native `.val()` can run without triggering a full-COCO download:
  - {root}/labels/val2017/*.txt   (class cx cy w h, normalized; class = 0..79)
  - {root}/coco-val.yaml          (val: images/val2017, 80 class names)

RF-DETR instead uses images/val2017 + annotations/instances_val2017.json directly
with pycocotools.

Idempotent: each step is skipped if its output already exists. Run once from
anywhere with any python3 (stdlib only):

    python3 prepare_coco_val.py
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
DEFAULT_ROOT = Path.home() / "datasets" / "coco"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n           -> {dest}")

    def _hook(count, block_size, total_size):
        if total_size > 0 and count % 200 == 0:
            done = count * block_size
            print(f"    {done / 1e6:7.1f} / {total_size / 1e6:7.1f} MB", end="\r")

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def ensure_images(root: Path):
    val_dir = root / "images" / "val2017"
    if val_dir.is_dir() and any(val_dir.glob("*.jpg")):
        print(f"[images] present: {val_dir}")
        return val_dir
    zip_path = root / "val2017.zip"
    if not zip_path.exists():
        _download(IMAGES_URL, zip_path)
    print(f"[images] extracting {zip_path} ...")
    (root / "images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root / "images")  # creates images/val2017/
    zip_path.unlink(missing_ok=True)
    print(f"[images] ready: {val_dir}")
    return val_dir


def ensure_annotations(root: Path):
    ann_path = root / "annotations" / "instances_val2017.json"
    if ann_path.exists():
        print(f"[annotations] present: {ann_path}")
        return ann_path
    zip_path = root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        _download(ANN_URL, zip_path)
    print(f"[annotations] extracting instances_val2017.json ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extract("annotations/instances_val2017.json", root)
    zip_path.unlink(missing_ok=True)
    print(f"[annotations] ready: {ann_path}")
    return ann_path


def generate_yolo_labels(root: Path, ann_path: Path):
    """Convert instances_val2017.json to Ultralytics YOLO txt labels (class 0..79)."""
    labels_dir = root / "labels" / "val2017"
    marker = labels_dir / ".complete"
    coco = json.loads(ann_path.read_text())

    # Contiguous 0..79 index in COCO category-id order (the standard COCO80 order).
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    catid_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    if marker.exists():
        print(f"[labels] present: {labels_dir}")
        return labels_dir, names

    labels_dir.mkdir(parents=True, exist_ok=True)
    img_by_id = {img["id"]: img for img in coco["images"]}
    lines_by_img = {img_id: [] for img_id in img_by_id}

    for ann in coco["annotations"]:
        if ann.get("iscrowd"):
            continue
        img = img_by_id[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        nw, nh = bw / w, bh / h
        idx = catid_to_idx[ann["category_id"]]
        lines_by_img[ann["image_id"]].append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    for img_id, img in img_by_id.items():
        stem = Path(img["file_name"]).stem
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines_by_img[img_id]))
    marker.write_text("ok")
    print(f"[labels] wrote {len(img_by_id)} label files -> {labels_dir}")
    return labels_dir, names


def write_yaml(root: Path, names: list[str]):
    yaml_path = root / "coco-val.yaml"
    lines = [
        f"path: {root}",
        "val: images/val2017",
        "",
        "names:",
    ]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"[yaml] wrote {yaml_path}")
    return yaml_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT), help="COCO root (default ~/datasets/coco).")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Preparing COCO val2017 under {root}\n")
    val_dir = ensure_images(root)
    ann_path = ensure_annotations(root)
    labels_dir, names = generate_yolo_labels(root, ann_path)
    yaml_path = write_yaml(root, names)

    n_imgs = len(list(val_dir.glob("*.jpg")))
    print(f"\nReady: {n_imgs} val images, {len(names)} classes")
    print(f"  images     : {val_dir}")
    print(f"  annotations: {ann_path}")
    print(f"  yolo yaml  : {yaml_path}")


if __name__ == "__main__":
    main()
