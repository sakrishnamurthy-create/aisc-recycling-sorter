# AISC Smart Recycling Sorter

AI-powered recycling sorter that classifies waste images into 4 material
classes (plastic, paper, glass, metal) in real time, built for AISC's
product competition (Dec 2, 2026).

Team: Sanjit Krishnamurthy (lead, AI/ML), Samuel Kidane (ML/CV), Micah Zahir
(sorting mechanism), Wilson Lo (data pipeline), Ria Kanwar (pitch/demo).

## Project status

See the project charter for full scope, success criteria, and risks.
Working backward from the Dec 2 presentation with 13 days from Nov 19:

| Deliverable | Target |
|---|---|
| Dataset collected/labeled (4 classes) | done via `prepare_data.py` (TrashNet) + team photos |
| First classifier trained and evaluated | this week |
| Sorting mechanism (simulated first, physical if time allows) | Nov 24 |
| Pitch deck drafted | Nov 28 |
| Dry run / rehearsal | Nov 30 |
| AISC product competition presentation | Dec 2 |

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the dataset

Downloads TrashNet and filters it to the 4 target classes, then splits
70/15/15 into train/val/test:

```bash
python src/prepare_data.py --out data/
```

To add your own photos (recommended, TrashNet is lab-lit objects on a
white background, not real bin/curbside conditions), drop them into
`data/raw/<class_name>/*.jpg` for `plastic`, `paper`, `glass`, `metal`,
then re-run with `--skip-download` to just re-split:

```bash
python src/prepare_data.py --out data/ --skip-download
```

## 2. Train the classifier

Transfer learning on MobileNetV2 (frozen backbone, only the final layer
trains), fast on CPU, appropriate for a few hundred images per class:

```bash
python src/train.py --data data/ --epochs 15 --out models/classifier.pt
```

Prints train/val accuracy per epoch, saves the best checkpoint, and
reports final held-out test accuracy against the charter's 85% bar.

## 3. Run the end-to-end demo

Single image:

```bash
python src/infer.py --model models/classifier.pt --image some_photo.jpg
```

Live webcam (the actual competition demo path, camera feed to
classification to simulated sorting action):

```bash
python src/infer.py --model models/classifier.pt --webcam
```

`src/sorter_sim.py` is the simulated sorting mechanism (prints/logs
which bin each item routes to). If Micah gets a physical sorter working,
it is designed to be swapped in without touching `infer.py`, see the
`PhysicalSorter` stub in that file.

## Repo layout

```
src/
  prepare_data.py   # TrashNet download + split, or organize your own photos
  train.py           # transfer-learning classifier training
  infer.py           # single-image or webcam inference + sorting demo
  sorter_sim.py      # simulated (and stubbed physical) sorting mechanism
data/                # created by prepare_data.py (gitignored, regenerate locally)
models/              # trained checkpoints (gitignored, too big for git)
```

## Open items

- Confirm current status: has data actually been collected and has a classifier actually been trained yet? Both were due before today.
- Decide now whether the sorting mechanism is simulated-only or physical+simulated fallback, do not leave this decision for later.
- Supplement TrashNet with team-collected photos for better real-world accuracy (TrashNet alone may not generalize to competition lighting and camera setup).
