# Land-cover segmentation on OpenEarthMap

Semantic segmentation of aerial and satellite images into the 9 OpenEarthMap land-cover classes, with PyTorch Lightning and SegFormer (`segmentation_models_pytorch`).

## Goals and motivation

Land-cover maps describe what covers the ground: buildings, roads, trees, water, farmland, bare soil and so on. Many applications rely on them:

- **Urban planning**: tracking how cities expand.
- **Environmental monitoring**: deforestation, water bodies, land degradation.
- **Agriculture**: estimating cultivated areas.
- **Disaster response**: comparing the situation before and after an event.

Producing these maps by hand does not scale. Semantic segmentation automates it by assigning a class to every pixel of an image. Doing this reliably across very different places is hard, because cities, vegetation and building styles vary from one region to another.

This project builds and evaluates a complete segmentation pipeline on OpenEarthMap, with images from 75 regions around the world. It has three goals:

1. **Train a model** that labels every pixel with one of 9 classes, and measure it with the mean Intersection-over-Union (mIoU) on whole test images.
2. **Find out which design choices actually matter**, through an ablation study that adds one change at a time:
   - data augmentation;
   - a class-weighted loss for the strong class imbalance;
   - a tuned learning-rate schedule with a longer training;
   - a larger encoder.
3. **Understand the remaining errors**, with per-class scores, a confusion matrix and visual examples.

## Contents

| File | Content |
|---|---|
| [`L115_B_2.ipynb`](L115_B_2.ipynb) | Executed notebook: data pipeline, model, exploratory analysis, ablation study (E0–E4), results and discussion |
| [`Relazione_L115_B_2.pdf`](Relazione_L115_B_2.pdf) | Short report (in Italian) |
| [`results/`](results) | Metrics, per-class IoU, confusion matrix and learning curves of each experiment (JSON) |
| `pyproject.toml`, `uv.lock` | Exact Python environment |

## Results (test set, whole images)

| Run | Configuration | mIoU (9 classes) | mIoU (8 classes, no Background) |
|---|---|---|---|
| E0 | Baseline: SegFormer `mit_b0`, cross-entropy, lr 1e-4 | 0.628 | 0.588 |
| E3 | + augmentation, weighted CE + Dice, LR finder + warm-up/cosine, 2.4× longer training | 0.660 | 0.622 |
| E4 | E3 with the `mit_b2` encoder | **0.688** | **0.653** |

## Conclusions

The best model, SegFormer `mit_b2`, reaches a test mIoU of **0.688** (0.653 over the 8 land-cover classes), 0.06 above the baseline. The results are useful beyond the final score:

- **Where to spend effort.** The baseline was limited by its training budget, not by its architecture. A learning rate chosen with an LR finder, a warm-up + cosine schedule and a longer training gave the largest gain (+0.032). Augmentation and loss re-weighting made no measurable difference at the short budget.
- **An accuracy/cost trade-off.**
  - The lightweight `mit_b0` (3.7 M parameters) reaches 0.660 in 45 minutes of training, a good fit for limited hardware or fast inference.
  - The larger `mit_b2` adds +0.027 and improves every class, for 1.7× the training time.
- **Where the model still fails.** Most remaining errors are between visually similar classes: bare soil vs. developed open space, roads vs. developed space, grassland vs. trees. Rare classes are not the main problem, and Water reaches 0.77 IoU while covering only 2% of the pixels. Better results are more likely to come from more context (larger crops) or clearer class definitions than from re-weighting.
- **A trustworthy evaluation setup.** Validation and test mIoU differ by at most 0.011 in every run, so the validation split can be used reliably for model selection when extending this work.

**Limitations.**
- Each configuration was trained with a single seed.
- Validation and test come from the same regions, so generalization to unseen cities was not measured.

The [report](Relazione_L115_B_2.pdf) and the notebook discuss both points.

## Reproduce

```bash
uv sync                      # create the environment
uv run jupyter lab           # open L115_B_2.ipynb
```

The first cell downloads the dataset (about 6.4 GB) from the Hugging Face Hub into `./data`; on Google Colab it uses `/content`. Training all five experiments took about 3 hours on an RTX 3080 Laptop GPU (16 GB). Results are cached in `results/`, so experiments already present are skipped when the notebook is re-run.
