import tempfile
from pathlib import Path
import gradio as gr
import numpy as np
from matplotlib import colormaps
from PIL import Image
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


def list_checkpoints():
    """All trained models (model.ckpt) in the 'results' folder, newest first."""
    ckpts = sorted(Path("results").rglob("model.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in ckpts]


def to_numpy(x):
    """Turn a tensor (or number) into a plain numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def check_photo(image, ckpt_path):
    if image is None:
        return "Please upload a photo.", None
    if not ckpt_path:
        return "No trained model found. Run the training script first.", None

    image = image.convert("RGB")

    
    with tempfile.TemporaryDirectory() as tmp:
        photo_path = Path(tmp) / "upload.png"
        image.save(photo_path)
        dataset = PredictDataset(path=photo_path, image_size=(256, 256))
        engine = Engine(accelerator="cpu")
        predictions = engine.predict(model=Patchcore(), dataset=dataset, ckpt_path=ckpt_path)

    
    batch = predictions[0]
    score = float(to_numpy(batch.pred_score).reshape(-1)[0])
    is_defect = bool(to_numpy(batch.pred_label).reshape(-1)[0])
    heat = np.squeeze(to_numpy(batch.anomaly_map)[0]).astype(np.float32)

    
    heat = np.array(Image.fromarray(heat).resize(image.size))
    heat = np.clip(heat, 0, 1)
    colour = colormaps["jet"](heat)[..., :3] * 255
    overlay = (0.5 * np.array(image) + 0.5 * colour).astype(np.uint8)

    verdict = "DEFECT" if is_defect else "GOOD"
    return f"{verdict}  (score {score:.3f}, higher = more likely defect)", overlay


checkpoints = list_checkpoints()

with gr.Blocks(title="Parts Defect Checker") as demo:
    gr.Markdown("# Parts Defect Checker\nUpload a photo of a part. The model says GOOD or DEFECT and shows where the problem is.")
    photo = gr.Image(type="pil", label="Upload a photo")
    model_choice = gr.Dropdown(
        choices=checkpoints,
        value=checkpoints[0] if checkpoints else None,
        label="Model (pick the one for this part: bottle or screw)",
    )
    check_btn = gr.Button("Check photo", variant="primary")
    result = gr.Textbox(label="Result")
    heatmap = gr.Image(label="Heat-map (red = likely defect)")
    check_btn.click(check_photo, inputs=[photo, model_choice], outputs=[result, heatmap])

if __name__ == "__main__":
    demo.launch()