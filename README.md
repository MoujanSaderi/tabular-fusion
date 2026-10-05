# Multimodal csPCa Detection

Detecting clinically significant prostate cancer (csPCa) by fusing 3D MRI
(T2 / ADC / B1500) with structured clinical features (PSA, PI-RADS, lesion
location, etc.) using a frozen ResNet3D imaging backbone.

## Architectures

| Architecture | Model | Config | Writeup |
|---|---|---|---|
| CBAM (channel + spatial clinical attention) | `aug11cbam.py` | `configs/cbam.yaml` | `cbam.md` |
| Early-scalar clinical attention | `aug11earlyscalar.py` | `configs/earlyscalar.yaml` | `earlyscalar.md` |
| Late fusion (flat clinical encoder) | `aug11latefusionflat.py` | `configs/latefusion_flat.yaml` | `latefusion.md` |
| DAFT (clinical affine transform in last layer4 block) | `daftfusion.py` | `configs/daft.yaml` | docstring in `daftfusion.py` |
| DAFT + late fusion, initialized from the baseline | `daftfusion.py` | `configs/daft_late.yaml` | docstring in `daftfusion.py` |
| DAFT + late fusion, initialized from the late-only model | `daftfusion.py` | `configs/daft_late_from_latefusion.yaml` | docstring in `daftfusion.py` |
| Probability-level late fusion | `late_fusion.py` | CLI args (see below) | — |

The first six fuse imaging + clinical *embeddings* inside one joint
network, trained via `train.py`; the first three each have a `.md` writeup (with matching
`.png` diagram) explaining the design. Probability-level late fusion is
different — it blends the output *probabilities* of two independently
trained models instead — so it runs as its own script, covered further down.

## Setup

```bash
pip install -r requirements.txt
```

This runs on the lab's HPC cluster (`prostatelab`) — the H5 imaging data and
CSV splits live under `/gpfs/data/prostatelab/...`. You'll need access to
that data (or an equivalent copy) for any of this to do anything.

## Data format

Each exam is one H5 file with:
```
['axt2'], ['adc'], ['b1500']        # imaging volumes
.attrs['maxPIRADS'], .attrs['psa'], .attrs['prostate_volume'], ...
```
Clinical features come from a separate CSV, keyed by `AccessionNumber`, with
37 columns matching what the clinical encoder expects.

## Running training

```bash
python train.py --config configs/cbam.yaml
python train.py --config configs/earlyscalar.yaml
python train.py --config configs/latefusion_flat.yaml
python train.py --config configs/daft.yaml
python train.py --config configs/daft_late.yaml
python train.py --config configs/daft_late_from_latefusion.yaml  # set model_ckpt first
```

Before running, check each config for:
- `paths.train_csv` / `paths.valid_csv` / `paths.data_dirs`
- `data.tabular_csv`
- `model_weights.model_ckpt` — a pretrained baseline checkpoint the frozen
  backbone initializes from (training freezes most/all of it in
  `on_train_start`, so this matters)
- `model_weights.clinical_ckpt` — optional pretrained clinical MLP checkpoint

Training uses early stopping + checkpointing on `best_val_pirads_auc`.

## Probability-level late fusion

Combine the output probabilities of a separately-trained imaging model and a
separately-trained tabular-only model (simple average / weighted average /
learned logistic regression), and compare each against the blend.

```bash
python late_fusion.py \
    --imaging-preds /path/to/preds_epoch_N.csv \
    --tabular-features /path/to/cspca_features_v4.csv \
    --tabular-model /path/to/cspca_model_v4_trainonly.pkl \
    --val-split-csv /path/to/val_split.csv \
    --output-csv late_fusion_preds.csv
```

- `--imaging-preds` expects the per-patient predictions CSV that `train.py`'s
  `trainer.test()` writes via `src/metrics/plot.py`'s `save_preds()` when a
  config's `debugging.debug` is `true`.
- `--tabular-model` expects a joblib-dumped `{"model", "feature_columns"}`
  dict from a tabular-only classifier trained separately on the clinical
  features CSV (not covered by anything else in this repo).

## `src/`

The `src/` package (data loading, ResNet3D backbone, metrics) mirrors the
lab's shared module of the same name, included here so training actually
runs from this repo rather than assuming everyone already has it on their
`PYTHONPATH`.
