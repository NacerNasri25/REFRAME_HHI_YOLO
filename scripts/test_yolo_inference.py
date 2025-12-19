from ultralytics import YOLO
from pathlib import Path

def pick_weights(model_root: Path) -> Path:
    weights = sorted(model_root.rglob("*.pt"))
    if not weights:
        raise FileNotFoundError(f"No .pt weights found under {model_root}")
    for w in weights:
        if w.name == "best.pt":
            return w
    return weights[0]

def pick_first_image(images_root: Path) -> Path:
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP", "*.tif", "*.TIF", "*.tiff", "*.TIFF", "*.webp", "*.WEBP")
    for pat in patterns:
        matches = list(images_root.rglob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No images found under {images_root}")

def main():
    model_root = Path("model/YOLO11_Model")
    images_root = Path("data/Dataset_YOLO/images")

    model_path = pick_weights(model_root)
    image_path = pick_first_image(images_root)

    print("MODEL_PATH:", model_path)
    print("IMAGE_PATH:", image_path)

    model = YOLO(str(model_path))
    results = model(str(image_path), verbose=True)

    r = results[0]
    n = 0 if r.boxes is None else len(r.boxes)
    print("Detected boxes:", n)
    print("Classes:", r.boxes.cls if r.boxes is not None else None)
    print("Scores:", r.boxes.conf if r.boxes is not None else None)

if __name__ == "__main__":
    main()
