# Segmentation with CNNs and Transformers: U-Net vs SegFormer on CamVid

Final project, Deep Vision (M.Sc. Artificial Intelligence, OTH Amberg-Weiden).

A controlled comparison of a convolutional model (**U-Net**, ResNet-34 encoder)
and a transformer model (**SegFormer**, MiT-B2 encoder) for semantic
segmentation, designed to isolate the effect of the feature-mixing mechanism
(local convolution vs. global self-attention) while holding data, training
pipeline, parameter budget, and ImageNet pretraining constant. The models are
compared along seven axes: accuracy, per-class behaviour, data efficiency,
robustness to corruption, boundary quality, cross-dataset generalization, and
test-time augmentation.

## Headline results (validation, CamVid)

| Axis | U-Net | SegFormer | Notes |
|------|-------|-----------|-------|
| mIoU (3 seeds) | 0.765 +/- 0.007 | **0.798 +/- 0.002** | gap = 6x pooled std, no overlap |
| Data efficiency @25% | 0.703 | **0.752** | SegFormer@25% ~ U-Net@100% |
| Robustness (retained @sev.5) | 54.3% | **66.5%** | U-Net collapses under noise |
| Boundary IoU (3px) | 0.416 | **0.453** | SegFormer wins 10/11 classes |
| Cross-dataset (Cityscapes) | 0.470 | **0.598** | gap widens to +0.128 |
| Best mIoU (flip+multiscale TTA) | 0.771 | **0.808** | gap stable under TTA |

Single theme: the transformer's advantage grows monotonically with the
difficulty of the condition -- marginal in-domain, largest under domain shift.

## Repository layout

```
src/
  # --- data ---
  dataset.py          CamVid dataset, transforms, palette, class constants
  visualize_data.py   sanity-check: sample grids + label validation
  class_stats.py      per-class pixel stats + median-frequency loss weights
  make_subsets.py     fixed nested 25/50/100% train subsets (seeded)
  # --- training core ---
  losses.py           combined CE (class-weighted) + Dice loss, void-ignored
  metrics.py          per-class IoU / mIoU / pixel-acc (torchmetrics)
  models.py           model factory: U-Net (smp) and SegFormer (HF)
  train.py            shared training script for BOTH models (AMP, poly LR)
  # --- evaluation & analysis ---
  evaluate.py         per-class IoU table + prediction grid for a run
  aggregate_results.py summary tables + data-efficiency plot
  compare_models.py   side-by-side qualitative comparison (disagreement mode)
  robustness.py       corruption evaluation (noise/blur/fog/brightness x5)
  boundary_iou.py     boundary-band IoU (edge-quality metric)
  tta_evaluate.py     test-time augmentation (flip + multi-scale)
  cityscapes_eval.py  zero-shot cross-dataset eval on Cityscapes
  interpretability.py encoder-focus heatmaps (qualitative, caveated)
scripts/
  download_camvid.sh  downloads the 11-class CamVid (SegNet-Tutorial)
requirements.txt
```

`data/`, `outputs/` and checkpoints are git-ignored; datasets are downloaded
on each machine.

## Setup

```bash
git clone <this-repo> camvid-seg && cd camvid-seg
conda create -n camvid-seg python=3.12 -y && conda activate camvid-seg
pip install torch torchvision          
pip install -r requirements.txt
bash scripts/download_camvid.sh        # verifies 367/101/233 split
```

Verify the GPU is visible:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Reproducing the experiments

All commands run from the repo root with the environment active. Training was
done on a single 8 GB GPU (RTX 3060 Ti); AMP is on by default.

### 1. Data preparation
```bash
python src/class_stats.py --root data/CamVid           # -> outputs/class_stats.json
python src/visualize_data.py --root data/CamVid --split train --num 6
python src/make_subsets.py --root data/CamVid --seed 42  # -> 25/50/100% lists
```

### 2. Training (six headline + subset runs)
```bash
# U-Net (lr 3e-4): 100/50/25%
python src/train.py --model unet --epochs 120 --lr 3e-4 --batch-size 8 --run-name unet_100
python src/train.py --model unet --epochs 120 --lr 3e-4 --batch-size 8 --subset outputs/subsets/train_50.txt --run-name unet_50
python src/train.py --model unet --epochs 120 --lr 3e-4 --batch-size 8 --subset outputs/subsets/train_25.txt --run-name unet_25
# SegFormer (lr 1e-4): 100/50/25%
python src/train.py --model segformer --epochs 120 --lr 1e-4 --batch-size 6 --run-name segformer_100
python src/train.py --model segformer --epochs 120 --lr 1e-4 --batch-size 6 --subset outputs/subsets/train_50.txt --run-name segformer_50
python src/train.py --model segformer --epochs 120 --lr 1e-4 --batch-size 6 --subset outputs/subsets/train_25.txt --run-name segformer_25
```

Seed reruns for variance (headline models, seeds 1 and 2):
```bash
for s in 1 2; do
  python src/train.py --model unet --epochs 120 --lr 3e-4 --batch-size 8 --seed $s --run-name unet_100_s$s
  python src/train.py --model segformer --epochs 120 --lr 1e-4 --batch-size 6 --seed $s --run-name segformer_100_s$s
done
```

### 3. Evaluation and analysis
```bash
python src/evaluate.py --run unet_100 --split val       # repeat per run
python src/aggregate_results.py --split val             # tables + data-eff plot
python src/compare_models.py --unet-run unet_100 --segformer-run segformer_100 --split val --mode disagree --num 6
python src/robustness.py --runs unet_100 segformer_100 --split val
python src/boundary_iou.py --runs unet_100 segformer_100 --split val --dilation 3
python src/tta_evaluate.py --run segformer_100 --split val --scales 0.75 1.0 1.25
python src/interpretability.py --unet-run unet_100 --segformer-run segformer_100 --split val --num 4
```

### 4. Cross-dataset generalization (Cityscapes)
Requires an approved Cityscapes account. Download the `gtFine` and
`leftImg8bit` val splits into `data/cityscapes/` (the `csDownload` tool from
`pip install cityscapesscripts` is the most reliable method), then:
```bash
python src/cityscapes_eval.py --runs unet_100 segformer_100 --cs-root data/cityscapes --num-vis 6
```
Cityscapes labels are remapped to the 11 CamVid classes (see `CS_TO_CAMVID` in
`cityscapes_eval.py`). The mapping is approximate, so the relative gap and
retention are the meaningful quantities, not absolute mIoU.

## Fairness protocol

The comparison's validity rests on changing only the architecture. Both models
share the same data pipeline, augmentations, combined CE+Dice loss,
median-frequency class weighting, optimizer (AdamW), poly LR schedule, epoch
count, and AMP. The single intentional asymmetry is the per-model learning
rate (U-Net 3e-4, SegFormer 1e-4), each tuned on validation; forcing an
identical LR would handicap one model. Any enhancement (TTA, longer training)
was applied identically to both. The test split was used only once, after all
tuning was frozen. A 200-epoch run confirmed both models converge by 120
epochs (U-Net +0.0003, SegFormer +0.006).

## Environment notes / gotchas

- **Install torch matched to your CUDA version** (check `nvidia-smi`).
- **albumentations is pinned to 1.4.18.** Version 2.x renamed padding
  arguments used in `dataset.py`; do not upgrade. The update-available warning
  it prints is harmless (set `NO_ALBUMENTATIONS_UPDATE=1` to silence).
- **imagecorruptions + NumPy 2.0:** the corruption library predates NumPy 2.0
  and references removed aliases (`np.float_`). `robustness.py` includes a
  compatibility shim at the top, so no downgrade is needed.
- The SegFormer "UNEXPECTED/MISSING" load report on first use is expected: the
  pretrained head is discarded and a fresh 11-class head is initialized.

## Results files

Generated under `outputs/`:
- `runs/<name>/` -- per run: `best.pt`, `last.pt`, `metrics.csv`,
  `config.json`, `eval_val.json`, `pred_val.png`.
- `analysis/` -- `summary_val.csv`, `per_class_val.csv`,
  `data_efficiency.png`, `compare_disagree_val.png`,
  `robustness_val.{csv,png}`, `boundary_iou_val.csv`,
  `cityscapes_summary.csv`, `cityscapes_qualitative.png`,
  `interpretability_val.png`.
