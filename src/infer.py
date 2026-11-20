"""
End-to-end demo: camera feed (or a single image) -> classification ->
simulated sorting action.

This is the script to run live at the AISC product competition if the
team goes with (or falls back to) a simulated sorter instead of a
physical one, per the charter's risk note ("a software or simulated
fallback should be ready early, not as a last resort").

Usage:
    # single image
    python src/infer.py --model models/classifier.pt --image path/to/photo.jpg

    # live webcam demo (press 'q' to quit)
    python src/infer.py --model models/classifier.pt --webcam
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from sorter_sim import SimulatedSorter

IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

eval_tfms = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes


def classify_pil_image(model, classes, device, pil_img) -> tuple[str, float]:
    x = eval_tfms(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def run_on_image(model, classes, device, image_path: Path, sorter: SimulatedSorter):
    img = Image.open(image_path)
    label, conf = classify_pil_image(model, classes, device, img)
    print(f"{image_path.name}: {label} (confidence {conf:.1%})")
    sorter.sort(label, conf)


def run_on_webcam(model, classes, device, sorter: SimulatedSorter, interval_s: float = 1.5):
    import cv2  # local import: only needed for the webcam path

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions.")

    # Right after permission is granted (or on some drivers generally), the
    # camera can take a moment to warm up -- the first few reads may fail
    # even though the device opened successfully. Retry briefly instead of
    # silently exiting on the first failed frame.
    warm_up_deadline = time.time() + 5.0
    ok, frame = cap.read()
    while not ok and time.time() < warm_up_deadline:
        time.sleep(0.2)
        ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            "Webcam opened but never returned a frame after 5s. Try unplugging/"
            "reopening the app using the camera elsewhere, or check System "
            "Settings > Privacy & Security > Camera again."
        )

    print("Webcam demo running. Press 'q' in the preview window to quit.")
    last_infer = 0.0
    label, conf = "-", 0.0
    consecutive_failures = 0
    first_loop = True
    try:
        while True:
            if first_loop:
                # already have a good frame from the warm-up read above
                first_loop = False
            else:
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures > 30:  # ~a few seconds of no frames
                        print("Lost the camera feed (no frames for a while). Stopping.")
                        break
                    time.sleep(0.1)
                    continue
            consecutive_failures = 0
            now = time.time()
            if now - last_infer >= interval_s:
                import cv2 as _cv2

                rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                label, conf = classify_pil_image(model, classes, device, pil_img)
                sorter.sort(label, conf)
                last_infer = now
            overlay = f"{label} ({conf:.0%})"
            cv2.putText(frame, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Smart Recycling Sorter - live demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/classifier.pt", type=Path)
    ap.add_argument("--image", type=Path, help="classify a single image file")
    ap.add_argument("--webcam", action="store_true", help="run the live webcam demo")
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=0.6,
        help="below this confidence, route to the 'uncertain' bin instead of guessing",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.model, device)
    sorter = SimulatedSorter(classes, conf_threshold=args.conf_threshold)

    if args.image:
        run_on_image(model, classes, device, args.image, sorter)
    elif args.webcam:
        run_on_webcam(model, classes, device, sorter)
    else:
        raise SystemExit("Pass either --image <path> or --webcam")


if __name__ == "__main__":
    main()
