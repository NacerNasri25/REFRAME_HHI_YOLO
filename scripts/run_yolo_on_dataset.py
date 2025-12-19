import json
from pathlib import Path
from ultralytics import YOLO

def pick_weights(model_root: Path) -> Path:
    weights = sorted(model_root.rglob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No .pt weights found under {model_root}")
    for w in weights:
        if w.name == "best.pt":
            return w
    return weights[0]

def find_images(images_root: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    exts = exts + tuple(e.upper() for e in exts)  # include .JPG etc
    return sorted([p for p in images_root.rglob("*") if p.is_file() and p.suffix in exts])

def export_split(split: str, conf: float, max_images: int | None):
    project_root = Path(".")
    dataset_root = project_root / "data" / "Dataset_YOLO"
    model_root = project_root / "model" / "YOLO11_Model"
    out_root = project_root / "outputs" / f"yolo_{split}"

    weights_path = pick_weights(model_root)
    model = YOLO(str(weights_path))

    images_dir = dataset_root / "images" / split
    image_paths = find_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"Using weights: {weights_path}")
    print(f"Split: {split} | images: {len(image_paths)} | conf: {conf}")
    out_root.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_out = out_root / img_path.stem
        img_out.mkdir(parents=True, exist_ok=True)

        results = model(str(img_path), conf=conf, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for i, (xyxy, score, cid) in enumerate(zip(boxes_xyxy, confs, clss)):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                dets.append({
                    "id": i,
                    "class_id": int(cid),
                    "score": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        info = {
            "image": str(img_path),
            "weights": str(weights_path),
            "num_detections": len(dets),
            "detections": dets,
        }

        (img_out / "info.json").write_text(json.dumps(info, indent=2))
        print(f"{img_path.name}: {len(dets)} detections -> {img_out}/info.json")

def main():
    # Adjust split and max_images as needed
    export_split(split="val", conf=0.25, max_images=20)

if __name__ == "__main__":
    main()
