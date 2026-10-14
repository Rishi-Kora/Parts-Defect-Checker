# Parts-Defect-Checker

Find defects on factory parts (bottles, screws) from a single photo, using **PatchCore** anomaly detection from the [anomalib](https://github.com/open-edge-platform/anomalib) library and the **MVTec AD** dataset.

The model learns **only from good parts**. When it sees a new photo, it says **GOOD** or **DEFECT**, gives a score, and shows a **heat-map** of where the problem is.

---

## Features

- Train a PatchCore model on any MVTec AD category (tested on `bottle` and `screw`)
- Check one photo from the command line, with a side-by-side heat-map
- Simple web page (Gradio) to upload a photo and see the result
- Runs on CPU, including Apple Silicon Macs

---

## Results

Measured on the MVTec AD test set for each category:

| Category | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|----------|-------------|----------|-------------|----------|
| Bottle   | 1.000       | 0.992    | 0.986       | 0.727    |
| Screw    | 0.970       | 0.959    | 0.989       | 0.370    |

- **Image** scores measure whether the model can tell good parts from defective ones.
- **Pixel** scores measure whether it marks the right area of the defect.
- Screw defects are very small (scratches on threads, bent tips), so the exact outline is harder to find (lower Pixel F1).

---

## Project Structure

```
Parts Defect/
├── train_patchcore_bottle.py   # Train + test PatchCore on one MVTec category
├── predict_one_image.py        # Check one photo from the command line
├── app.py                      # Gradio web page for uploading photos
├── bottle/                     # MVTec AD category (downloaded separately)
├── screw/                      # MVTec AD category (downloaded separately)
└── results/                    # Trained models and test images (created by training)
```

---

## Requirements

- **Python 3.10 or newer** (current anomalib does not support Python 3.9)
- macOS, Linux or Windows

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the libraries
python -m pip install "anomalib[cpu]" matplotlib gradio
```

---

## Dataset

1. Download one or more categories from the [MVTec AD download page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads).
2. Extract them into the project folder, so each category sits in its own folder (`bottle/`, `screw/`).

Each category contains:

- `train/good/`: good images only (used for training)
- `test/`: good and defective images
- `ground_truth/`: masks that show the exact defect area

> **Licence note:** MVTec AD is released under **CC BY-NC-SA 4.0** and **may not be used for commercial purposes**. Use your own images for any commercial product.

---

## Usage

### 1. Train and test a model

Open `train_patchcore_bottle.py` and set:

- `DATA_ROOT`: the folder that **contains** the category folders (not the category folder itself)
- `category`: for example `"bottle"` or `"screw"`

Then run:

```bash
python train_patchcore_bottle.py
```

The script prints the test scores (AUROC, F1). It saves the model under `results/Patchcore/MVTecAD/<category>/`.

### 2. Check one photo (command line)

```bash
python predict_one_image.py "/full/path/to/photo.png"
```

- Uses the **most recently trained** model in `results/`
- Prints `GOOD` or `DEFECT` with a score
- Opens a window with the original photo and the heat-map

### 3. Web page (Gradio)

```bash
python app.py
```

Open the link shown in the terminal (usually http://127.0.0.1:7860), then:

1. Upload a photo
2. Choose the model for that part (bottle or screw)
3. Click **Check photo**

---

## How to Read the Output

- **Score:** from 0 to 1. Higher means more likely a defect.
- **GOOD / DEFECT:** decided by the threshold the model learned during testing.
- **Heat-map:** blue = normal, red = likely defect. Colours use a fixed 0–1 scale, so red always means a high score.

---

## Limitations

- The model only knows the **exact photo setup** it was trained on (camera angle, lighting, background).
- A photo taken differently (for example, a bottle from the side or an internet image) will look "unusual" and be marked **DEFECT**, even if the part is fine.
- For real factory use, train on photos taken with a **fixed camera, angle and lighting** that match production.

---

## Roadmap

- [x] Explore the data
- [x] Train PatchCore (bottle)
- [x] Check a single photo with a heat-map
- [x] Test a harder category (screw)
- [x] Gradio web page
- [ ] Train on our own factory photos
- [ ] Live camera inspection
- [ ] Defect dashboard (per day, shift, machine)

---

## Acknowledgements

- [anomalib](https://github.com/open-edge-platform/anomalib) by Intel (Apache 2.0)
- MVTec AD dataset:
  - Bergmann et al., *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, IJCV 2021.
  - Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- PatchCore: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022.

---

## Licence

The code in this repository is released under the **MIT** licence. See [LICENSE](LICENSE) for more information.
The MVTec AD dataset is **not included** and keeps its own non-commercial licence.
