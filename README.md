# Land-cover segmentation on OpenEarthMap

Semantic segmentation of aerial and satellite images into the 9 OpenEarthMap land-cover classes, with PyTorch Lightning and SegFormer (`segmentation_models_pytorch`). This is a course assignment (notebook `L115_B_2`).

| File | Content |
|---|---|
| [`L115_B_2.ipynb`](L115_B_2.ipynb) | Executed notebook: solved template TODOs, exploratory analysis, ablation study (E0–E4), results and discussion |
| [`Relazione_L115_B_2.pdf`](Relazione_L115_B_2.pdf) | Short report (in Italian) |
| [`results/`](results) | Metrics, per-class IoU, confusion matrix and learning curves of each experiment (JSON) |
| `pyproject.toml`, `uv.lock` | Exact Python environment |

## Results (test set, whole images)

| Run | Configuration | mIoU (9 classes) | mIoU (8 classes, no Background) |
|---|---|---|---|
| E0 | Baseline: SegFormer `mit_b0`, cross-entropy, lr 1e-4 | 0.628 | 0.588 |
| E3 | + augmentation, weighted CE + Dice, LR finder + warm-up/cosine, 2.4× longer training | 0.660 | 0.622 |
| E4 | E3 with the `mit_b2` encoder | **0.688** | **0.653** |

## Reproduce

```bash
uv sync                      # create the environment
uv run jupyter lab           # open L115_B_2.ipynb
```

The first cell downloads the dataset (about 6.4 GB) from the Hugging Face Hub into `./data`; on Google Colab it uses `/content`. Training all five experiments took about 3 hours on an RTX 3080 Laptop GPU (16 GB). Results are cached in `results/`, so experiments already present are skipped when the notebook is re-run.
