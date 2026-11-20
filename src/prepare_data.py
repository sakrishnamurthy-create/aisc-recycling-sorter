"""
Dataset preparation for the AISC Smart Recycling Sorter.

Downloads the TrashNet dataset (Yang & Thung, Stanford CS229, 2016),
https://github.com/garythung/trashnet, filters it down to the charter's
4 target classes (plastic, paper, glass, metal), and splits it into
train/val/test folders ready for torchvision.datasets.ImageFolder.

Usage:
    python src/prepare_data.py --out data/

If your team collects its own photos instead of (or in addition to)
TrashNet, drop them into data/raw/<class_name>/*.jpg using the same
4 class names before running this script with --skip-download, and
it will split+organize whatever is in data/raw/.
"""
import argparse
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

TRASHNET_REPO = "https://github.com/garythung/trashnet.git"
TARGET_CLASSES = ["plastic", "paper", "glass", "metal"]
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def download_trashnet(tmp_dir: Path) -> Path:
    """Clone TrashNet and unzip its resized image set. Returns the path
    to the extracted dataset-resized/ folder (6 class subfolders)."""
    repo_dir = tmp_dir / "trashnet"
    if not repo_dir.exists():
        print(f"Cloning {TRASHNET_REPO} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", TRASHNET_REPO, str(repo_dir)],
            check=True,
        )
    zip_path = repo_dir / "data" / "dataset-resized.zip"
    extract_dir = tmp_dir / "trashnet_extracted"
    resized_dir = extract_dir / "dataset-resized"
    if not resized_dir.exists():
        print("Unzipping dataset-resized.zip ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return resized_dir


def collect_raw(raw_dir: Path) -> dict[str, list[Path]]:
    """Return {class_name: [image paths]} for every TARGET_CLASSES
    subfolder found under raw_dir."""
    by_class = {}
    for cls in TARGET_CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"  WARNING: no '{cls}' folder found under {raw_dir}, skipping.")
            continue
        imgs = sorted(
            p
            for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        by_class[cls] = imgs
        print(f"  {cls}: {len(imgs)} images")
    return by_class


def split_and_copy(by_class: dict[str, list[Path]], out_dir: Path, seed: int = 42):
    random.seed(seed)
    for cls, imgs in by_class.items():
        imgs = imgs.copy()
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train : n_train + n_val],
            "test": imgs[n_train + n_val :],
        }
        for split_name, split_imgs in splits.items():
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_imgs:
                shutil.copy2(src, dest_dir / src.name)
        print(
            f"  {cls}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output data/ directory")
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="skip TrashNet download, use data/raw/<class>/*.jpg you already collected",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    raw_dir = out_dir / "raw"

    if not args.skip_download:
        tmp_dir = out_dir / "_download_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        resized_dir = download_trashnet(tmp_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for cls in TARGET_CLASSES:
            src_cls_dir = resized_dir / cls
            dest_cls_dir = raw_dir / cls
            if src_cls_dir.exists() and not dest_cls_dir.exists():
                shutil.copytree(src_cls_dir, dest_cls_dir)
        print(f"TrashNet images copied into {raw_dir}")
        print(
            "NOTE: TrashNet is lab photos on a white board, not real bin/curbside "
            "shots. Supplement with your own photos in data/raw/<class>/ for a "
            "classifier that generalizes to the live demo setup."
        )

    print(f"\nCollecting images from {raw_dir} ...")
    by_class = collect_raw(raw_dir)
    if not by_class:
        raise SystemExit(
            f"No images found under {raw_dir}. Either let this script download "
            "TrashNet, or add your own data/raw/<class>/*.jpg first."
        )

    print(f"\nSplitting into {out_dir}/train, /val, /test ...")
    split_and_copy(by_class, out_dir, seed=args.seed)
    print("\nDone. Point train.py at this data/ directory.")


if __name__ == "__main__":
    main()
