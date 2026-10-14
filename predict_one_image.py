import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def find_latest_checkpoint():
    """Find the newest trained model file (model.ckpt) inside the 'results' folder."""
    ckpts = list(Path("results").rglob("model.ckpt"))
    if not ckpts:
        raise SystemExit("No model.ckpt found in 'results'. Run train_patchcore_bottle.py first.")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def to_numpy(x):
    """Turn a tensor (or number) into a plain numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict_one_image.py /full/path/to/photo.png")

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    ckpt_path = find_latest_checkpoint()
    print(f"Using model: {ckpt_path}")

    # 1. Load model + run prediction on the one image
    model = Patchcore()
    engine = Engine(accelerator="cpu")
    dataset = PredictDataset(path=image_path, image_size=(256, 256))
    predictions = engine.predict(model=model, dataset=dataset, ckpt_path=ckpt_path)

    # 2. Read the result (predictions come in batches; we take the first image)
    batch = predictions[0]
    score = float(to_numpy(batch.pred_score).reshape(-1)[0])
    is_defect = bool(to_numpy(batch.pred_label).reshape(-1)[0])
    heat = np.squeeze(to_numpy(batch.anomaly_map)[0])  # 2D map

    verdict = "DEFECT" if is_defect else "GOOD"
    print(f"\nResult: {verdict}   (score = {score:.3f}, higher = more likely defect)")

    # 3. Show original photo and heat-map on top
    original = Image.open(image_path).convert("RGB")
    heat_img = Image.fromarray(heat.astype(np.float32)).resize(original.size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original photo")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(np.array(heat_img), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"{verdict}  (score {score:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()