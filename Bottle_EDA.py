from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# >>> CHANGE THIS to the folder where you extracted the bottle dataset <<<
DATA_DIR = Path(r"/Users/rishikora/projects/projects/Parts Defect/bottle")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MASK_DIR = DATA_DIR / "ground_truth"


def list_images(folder):
    """Return all .png images in a folder, sorted by name."""
    return sorted(folder.glob("*.png"))



def count_images():
    print("\n=== STEP 1: IMAGE COUNTS ===")
    for split_dir in [TRAIN_DIR, TEST_DIR, MASK_DIR]:
        if not split_dir.exists():
            print(f"Missing folder: {split_dir}")
            continue
        for sub in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            print(f"{split_dir.name:13s} / {sub.name:15s}: {len(list_images(sub))} images")



def check_sizes():
    print("\n=== STEP 2: IMAGE SIZE AND COLOUR ===")
    all_images = list(TRAIN_DIR.rglob("*.png")) + list(TEST_DIR.rglob("*.png"))
    sizes = {}
    modes = {}
    for path in all_images:
        with Image.open(path) as img:
            sizes[img.size] = sizes.get(img.size, 0) + 1
            modes[img.mode] = modes.get(img.mode, 0) + 1
    print("Sizes (width, height):", sizes)
    print("Colour modes:", modes, "(RGB = colour, L = grey)")



def show_samples():
    print("\n=== STEP 3: SAMPLE IMAGES (close the window to continue) ===")
    groups = sorted(p for p in TEST_DIR.iterdir() if p.is_dir())
    per_row = 4
    fig, axes = plt.subplots(len(groups), per_row, figsize=(3 * per_row, 3 * len(groups)))
    for row, group in enumerate(groups):
        for col, path in enumerate(list_images(group)[:per_row]):
            ax = axes[row, col]
            ax.imshow(Image.open(path))
            ax.set_title(f"{group.name}\n{path.name}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    plt.show()



def show_masks():
    print("\n=== STEP 4: DEFECTS WITH MASK (close the window to finish) ===")
    defect_groups = sorted(p for p in TEST_DIR.iterdir() if p.is_dir() and p.name != "good")
    fig, axes = plt.subplots(len(defect_groups), 3, figsize=(9, 3 * len(defect_groups)))
    for row, group in enumerate(defect_groups):
        img_path = list_images(group)[0]
        mask_path = MASK_DIR / group.name / f"{img_path.stem}_mask.png"
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0

        # Paint defect pixels red on a copy of the image
        overlay = image.copy()
        overlay[mask] = [255, 0, 0]

        defect_percent = 100 * mask.sum() / mask.size
        for col, (pic, title) in enumerate([
            (image, f"{group.name} - image"),
            (mask, "mask (white = defect)"),
            (overlay, f"overlay ({defect_percent:.2f}% area)"),
        ]):
            axes[row, col].imshow(pic, cmap="gray" if col == 1 else None)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if not DATA_DIR.exists():
        raise SystemExit(f"Folder not found: {DATA_DIR}\nPlease set DATA_DIR at the top of the script.")
    count_images()
    check_sizes()
    show_samples()
    show_masks()