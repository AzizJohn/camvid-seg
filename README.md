# Segmentation with CNNs and Transformers

Final project, Deep Vision. A controlled comparison of a CNN-based model
(U-Net, ResNet34 encoder) and a Transformer-based model (SegFormer MiT-B2)
for semantic segmentation on CamVid (11 classes), analyzing accuracy,
data efficiency, robustness to corruptions, and boundary quality.

## Repository layout

```
src/
  dataset.py         CamVid dataset, transforms, palette, class constants
  visualize_data.py  Day 1 sanity check: sample grids + label validation
  class_stats.py     Day 1: class pixel statistics and loss weights
scripts/
  download_camvid.sh Downloads the 11-class CamVid (SegNet-Tutorial)
requirements.txt
```

`data/`, `outputs/` and `checkpoints/` are git-ignored; the dataset is
downloaded directly on each machine.

## Setup (on the GPU server)

```bash
git clone <YOUR-GITLAB-URL> camvid-seg && cd camvid-seg

python3 -m venv .venv && source .venv/bin/activate
# install torch matched to the CUDA version shown by nvidia-smi, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

bash scripts/download_camvid.sh
```

## Day 1 checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python src/class_stats.py --root data/CamVid
python src/visualize_data.py --root data/CamVid --split train --num 6
python src/visualize_data.py --root data/CamVid --split val --num 4
```

Expected: CUDA available; split sizes 367/101/233; all label checks pass;
`outputs/data_check_*.png` shows correctly aligned image/mask overlays;
`outputs/class_stats.json` contains pixel shares and class weights.
