# YOLOv12 venv setup with `uv`

Target stack: Python **3.11**, torch **2.2.2+cu121**, torchvision **0.17.2+cu121**, numpy **1.26.4**, Ubuntu + A100 (CUDA 12.x driver). The `ultralytics` package is provided by the local YOLOv12 repo (editable install), not from PyPI.

## 1. Install `uv` (skip if already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

cd ~/three_od_models/yolov12
rm -rf .venv
uv venv --python 3.11 .venv
source .venv/bin/activate

--extra-index-url https://download.pytorch.org/whl/cu121

# core DL
torch==2.2.2+cu121
torchvision==0.17.2+cu121
torchaudio==2.2.2+cu121

# numerical stack (numpy<2 is critical)
numpy==1.26.4
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
pillow>=10.0
opencv-python>=4.8
pyyaml>=6.0
tqdm>=4.66
psutil
py-cpuinfo
requests
scikit-learn>=1.3

# ultralytics runtime deps (repo's setup.py also lists some; explicit is fine)
ultralytics-thop

# ONNX (optional)
onnx>=1.15
onnxruntime-gpu==1.17.1

# Jupyter
jupyterlab
ipywidgets
ipykernel

# misc
roboflow


uv pip install -r requirements.txt
uv pip install setuptools wheel       # only if -e . complains about build deps
uv pip install -e . --no-deps

python -c "import ultralytics, os; print(ultralytics.__version__); print(os.path.dirname(ultralytics.__file__))"

Path must point into your repo (.../three_od_models/yolov12/ultralytics), not site-packages. If it points to site-packages, run uv pip uninstall ultralytics and redo step 4.

python -m ipykernel install --user --name yolov12 --display-name "Python 3.11 (yolov12)"

python - <<'PY'
import numpy, torch, torchvision
from ultralytics import YOLO
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY


Expected: numpy 1.26.4, torch 2.2.2+cu121 cuda? True, A100 listed, no NumPy 1.x/2.x warnings.

from ultralytics import YOLO

m = YOLO("/home/arina_belova_jetbrains_com/rf100_runs/yolov12n_4-fold-defect/weights/best.pt")
r = m.val(
    data="/home/arina_belova_jetbrains_com/roboflow-100-benchmark/rf100/4-fold-defect/data.yaml",
    split="val", workers=0, verbose=True,
)
print(r.box.map50, r.box.map)



 Lock for reproducibility: 
uv pip freeze > requirements.lock.txt

Rebuild identically:

uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install -e . --no-deps

