"""
Config-driven training entrypoint for the three final frozen-encoder fusion
architectures, plus the tabular-only clinical MLP that pretrains their
clinical encoder:

    cbam            -> aug11cbam.TriSeriesModel
    earlyscalar     -> aug11earlyscalar.TriSeriesModelFrozen
    latefusion_flat -> aug11latefusionflat.LateFusionFlatFrozen
    clinical_mlp    -> mlpclinical.ClinicalMLPModel

All four subclass src.models.ResNet3D.base_3Dresnet.Base3DResNet. The fusion
models expect a frozen imaging backbone (loaded from model_weights.model_ckpt)
plus an optional frozen clinical MLP encoder (model_weights.clinical_ckpt).

clinical_mlp uses only the tabular features. After training, its best
checkpoint is converted into the {fc1.*, fc2.*} state dict the fusion models'
FrozenClinicalEncoder loads, and saved as clinical_encoder.pt in
save_weights_dir; point model_weights.clinical_ckpt at that file.

Usage:
    python train.py --config configs/clinical_mlp.yaml   # step 1 (optional)
    python train.py --config configs/cbam.yaml           # step 2

    # Convert an existing clinical_mlp Lightning checkpoint without retraining:
    python train.py --export-clinical-encoder path/to/clinical_mlp_*.ckpt
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, default_collate

from src.data.loader import ExamH5Dataset
from src.models.ResNet3D.lakshita_earlyfusion_construct_resnet3d import (
    load_saved_resnet3d_weights,
)
from src.utils.data_enums import SeriesType

from aug11cbam import TriSeriesModel as CBAMModel
from aug11earlyscalar import TriSeriesModelFrozen as EarlyScalarModel
from aug11latefusionflat import LateFusionFlatFrozen as LateFusionFlatModel
from mlpclinical import ClinicalMLPModel


MODEL_REGISTRY = {
    "cbam": CBAMModel,
    "earlyscalar": EarlyScalarModel,
    "latefusion_flat": LateFusionFlatModel,
    "clinical_mlp": ClinicalMLPModel,
}

# Models that never look at the imaging volumes, so an imaging checkpoint
# (model_weights.model_ckpt) has nothing to initialize.
TABULAR_ONLY_MODELS = {"clinical_mlp"}

# ClinicalMLPModel's nn.Sequential layer names -> the attribute names
# FrozenClinicalEncoder reads in aug11cbam / aug11earlyscalar /
# aug11latefusionflat. The final 64 -> 2 classifier (mlp.5) is dropped: the
# encoders only reuse the 37 -> 128 -> 64 feature extractor.
CLINICAL_ENCODER_KEY_MAP = {
    "mlp.0.weight": "fc1.weight",
    "mlp.0.bias": "fc1.bias",
    "mlp.2.weight": "fc2.weight",
    "mlp.2.bias": "fc2.bias",
}
CLINICAL_ENCODER_FILENAME = "clinical_encoder.pt"

# Input width of ClinicalMLPModel and every FrozenClinicalEncoder (nn.Linear(37, ...)).
CLINICAL_FEATURE_DIM = 36

SERIES_NAME_TO_ENUM = {
    "axt2": SeriesType.AXT2,
    "adc": SeriesType.ADC,
    "b1500": SeriesType.B1500,
    "dce": SeriesType.DCE,
}

TARGET_FROM_MODE = {
    "pirads": "pirads",
    "gleason": "gleason",
    "tstage": "tstage",
    "cspca": "cspca",
}


def collate_with_tabular_alias(batch):
    """ExamH5Dataset returns the clinical feature tensor as 'TabularFeatures',
    but the aug11 fusion models' training/validation/test steps read
    batch['tabular_features']. Alias it here instead of editing either the
    recovered loader or the recovered model files."""
    collated = default_collate(batch)
    collated["tabular_features"] = collated["TabularFeatures"]
    return collated


def build_dataset(csv_path, config, mode, series):
    data_cfg = config["data"]
    return ExamH5Dataset(
        metadata_csv=csv_path,
        data_dirs=config["paths"]["data_dirs"],
        series=series,
        model_type="3D",
        augment=data_cfg.get("augment", "none") if mode == "train" else "none",
        noise_sigma_range=data_cfg.get("noise_sigma_range", (0.0, 0.15)),
        downsample_factors=data_cfg.get("downsample_factors"),
        pirads_cutoff=data_cfg["pirads_cutoff"],
        mask_prostate=data_cfg.get("mask_prostate", False),
        device="cpu",
        mode=mode,
        normalize=data_cfg.get("normalize", True),
        target=TARGET_FROM_MODE[config["training"].get("mode", "pirads")],
        axt2_key=data_cfg.get("axt2_key", "axt2"),
        dwi_suffices=data_cfg.get("dwi_suffices"),
        dce_dirs=config["paths"].get("dce_dirs"),
        tabular_csv=data_cfg.get("tabular_csv"),
        # Tabular-only models never look at the MRI volumes, so don't read them.
        load_volumes=config["model"]["model"].lower() not in TABULAR_ONLY_MODELS,
    )


def build_dataloader(dataset, config, shuffle):
    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 4),
        shuffle=shuffle,
        collate_fn=collate_with_tabular_alias,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(config):
    model_name = config["model"]["model"].lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of {list(MODEL_REGISTRY)}."
        )
    model = MODEL_REGISTRY[model_name](config)

    weights_cfg = config.get("model_weights", {})
    if model_name in TABULAR_ONLY_MODELS:
        if weights_cfg.get("load_weights") and weights_cfg.get("model_ckpt"):
            print(
                f"[{model_name}] ignoring model_weights.model_ckpt: this model "
                "has no imaging backbone to initialize."
            )
        return model

    if weights_cfg.get("load_weights") and weights_cfg.get("model_ckpt"):
        checkpoint = torch.load(
            weights_cfg["model_ckpt"],
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # Non-matching keys (the attention/clinical-fusion layers, which the
        # checkpoint - trained on the plain baseline branched model - doesn't
        # have) are silently skipped, leaving them at their random init.
        model = load_saved_resnet3d_weights(
            model,
            checkpoint,
            disable_gradient=weights_cfg.get("disable_gradient", False),
        )
    return model


def export_clinical_encoder(lightning_ckpt_path, out_path):
    """Convert a ClinicalMLPModel Lightning checkpoint into the plain
    {fc1.weight, fc1.bias, fc2.weight, fc2.bias} dict that the fusion models'
    FrozenClinicalEncoder reads via model_weights.clinical_ckpt."""
    checkpoint = torch.load(lightning_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    missing = [k for k in CLINICAL_ENCODER_KEY_MAP if k not in state_dict]
    if missing:
        raise KeyError(
            f"{lightning_ckpt_path} has no {missing}; is it a clinical_mlp checkpoint?"
        )

    encoder_state = {
        new_key: state_dict[old_key].detach().cpu().clone()
        for old_key, new_key in CLINICAL_ENCODER_KEY_MAP.items()
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, out_path)
    print(f"Saved clinical encoder weights to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train one of the final fusion architectures "
            "(cbam / earlyscalar / latefusion_flat) or the clinical_mlp encoder"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to a YAML config (see configs/)")
    source.add_argument(
        "--export-clinical-encoder",
        metavar="CKPT",
        help="Convert an existing clinical_mlp Lightning checkpoint to a "
        "clinical_ckpt file, then exit (no training)",
    )
    parser.add_argument(
        "--out",
        help=f"Output path for --export-clinical-encoder "
        f"(default: {CLINICAL_ENCODER_FILENAME} next to the checkpoint)",
    )
    args = parser.parse_args()

    if args.export_clinical_encoder:
        ckpt = Path(args.export_clinical_encoder)
        export_clinical_encoder(ckpt, args.out or ckpt.parent / CLINICAL_ENCODER_FILENAME)
        return

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("paths", {}).setdefault("extra_valid_csv", None)
    config.setdefault("paths", {}).setdefault("dce_dirs", [])

    torch.manual_seed(config["training"].get("seed", 42))

    model_name = config["model"]["model"].lower()
    if model_name in TABULAR_ONLY_MODELS and not config["data"].get("tabular_csv"):
        raise ValueError(
            f"{model_name} trains on clinical features only; set data.tabular_csv."
        )

    series = [SERIES_NAME_TO_ENUM[s] for s in config["data"]["series"]]

    train_dataset = build_dataset(config["paths"]["train_csv"], config, "train", series)
    val_dataset = build_dataset(config["paths"]["valid_csv"], config, "val", series)

    n_features = train_dataset.num_tabular_features
    if config["data"].get("tabular_csv") and n_features != CLINICAL_FEATURE_DIM:
        raise ValueError(
            f"{config['data']['tabular_csv']} has {n_features} feature columns "
            f"(excluding AccessionNumber and split), but the clinical MLP/encoders "
            f"take {CLINICAL_FEATURE_DIM} inputs."
        )

    if config["training"].get("imbalance_strategy") == "weighted_loss" and not config[
        "training"
    ].get("class_weights"):
        config["training"]["class_weights"] = train_dataset.class_weights

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    val_loader = build_dataloader(val_dataset, config, shuffle=False)

    # Flat list, one maxPIRADS per validation exam in loader order (val_loader
    # isn't shuffled, so it lines up with the concatenated val predictions).
    # It used to be wrapped in an extra list, which made np.array(pirads) shape
    # (1, N) and crashed plot_pirads_cm with an IndexError.
    config["logging"]["val_pirads"] = val_dataset.pirads

    model = build_model(config)

    use_gpu = config["training"].get("gpu", False) and torch.cuda.is_available()
    if use_gpu:
        model.to(torch.device("cuda:0"))

    save_dir = Path(config["model_weights"]["save_weights_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    # Always create preds_dir: on_test_epoch_end writes a predictions CSV there
    # regardless of debugging.debug, so trainer.test() would otherwise crash
    # after a full training run if the directory doesn't exist yet.
    Path(config["debugging"]["preds_dir"]).mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["debugging"]["preds_dir"]) / "logs"
    config["logging"]["log_dir"] = str(log_dir)
    for flag, subdir in (
        ("plot_roc", "roc"),
        ("plot_confusion_matrix", "cm"),
        ("plot_pirads_breakdown", "pirads_breakdown"),
    ):
        if config["logging"].get(flag):
            (log_dir / subdir).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="best_val_pirads_auc",
        dirpath=save_dir,
        filename=config["model"]["model"] + "_{epoch:02d}_{best_val_pirads_auc:.3f}",
        mode="max",
        save_top_k=1,
    )
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="best_val_pirads_auc",
            patience=int(config["hyperparameters"].get("max_patience", 20)),
            mode="max",
        ),
        checkpoint_callback,
    ]

    trainer_kwargs = dict(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=callbacks,
        max_epochs=int(config["training"]["epochs"]),
        default_root_dir=save_dir,
        num_sanity_val_steps=0,
    )
    if config["logging"].get("log_run"):
        trainer_kwargs["logger"] = WandbLogger(
            name=config["logging"]["run_name"],
            project=config["logging"]["project_name"],
            save_dir=str(save_dir),
        )

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, val_dataloaders=[val_loader])

    if model_name in TABULAR_ONLY_MODELS:
        best_ckpt = checkpoint_callback.best_model_path
        if best_ckpt:
            encoder_path = export_clinical_encoder(
                best_ckpt, save_dir / CLINICAL_ENCODER_FILENAME
            )
            print(
                "Use it in a fusion config with:\n"
                f"  model_weights:\n    clinical_ckpt: {encoder_path}"
            )
        else:
            print("No checkpoint was saved, so no clinical encoder was exported.")

    test_csv = config["paths"].get("test_csv")
    if test_csv:
        test_dataset = build_dataset(test_csv, config, "test", series)
        test_loader = build_dataloader(test_dataset, config, shuffle=False)
    else:
        test_loader = val_loader
    trainer.test(model, dataloaders=[test_loader])


if __name__ == "__main__":
    main()
