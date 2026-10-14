from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
DATA_ROOT = "/Users/rishikora/projects/projects/Parts Defect/"
datamodule = MVTecAD(
    root=DATA_ROOT,
    category="screw",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,  
)

model = Patchcore()
engine = Engine(max_epochs=1, accelerator="cpu")
engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)
print("\nTEST RESULTS:", results)
print("Heat-map images and the saved model are in the 'results' folder.")