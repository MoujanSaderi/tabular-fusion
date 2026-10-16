"""
Run a trained tabular-fusion model on a validation CSV and save per-exam
predictions to a CSV.

Works with any model in train.py's MODEL_REGISTRY (cbam, earlyscalar,
latefusion_flat, latefusion_flat_llrd, daft, crossattn, crossattn_late,
clinical_mlp, pirads_rule). Use the same YAML config the checkpoint was trained with: it
defines the architecture, series, tabular CSV and preprocessing.

Usage:
    python eval.py \
        --config configs/latefusion_flat.yaml \
        --ckpt output/latefusion_pjsplits_v1/latefusion_flat_epoch=12_best_val_pirads_auc=0.871.ckpt \
        --out preds_val.csv

    # evaluate on a different split than paths.valid_csv:
    python eval.py --config ... --ckpt ... --val-csv /path/to/other_split.csv --out preds.csv

    # rule-based models (pirads_rule) have no weights, so no --ckpt:
    python eval.py --config configs/pirads_rule.yaml --out preds_pirads_rule.csv

The output CSV has one row per exam with AccessionNumber, PatientID, target,
maxPIRADS, logits, softmax probabilities and a thresholded prediction. It also
includes "Predictions" / "Targets" columns in the same format as
src/metrics/plot.py:save_preds, so it can be passed straight to
late_fusion.py --imaging-preds.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from train import (
    MODEL_REGISTRY,
    RULE_BASED_MODELS,
    SERIES_NAME_TO_ENUM,
    build_dataloader,
    build_dataset,
)


def to_device(batch, device):
    """Move the tensors the models need onto device."""
    volumes = {k: v.to(device, non_blocking=True) for k, v in batch["volume_data_dict"].items()}
    tabular = batch["tabular_features"].to(device, non_blocking=True)
    return volumes, tabular


def load_model(config, ckpt_path, device):
    model_name = config["model"]["model"].lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of {list(MODEL_REGISTRY)}.")

    # Build the bare architecture. We deliberately skip train.build_model's
    # pretrained-backbone loading (model_weights.model_ckpt): every weight is
    # overwritten by the trained checkpoint below anyway.
    model = MODEL_REGISTRY[model_name](config)

    if model_name in RULE_BASED_MODELS:
        # Fixed rule with no weights: nothing to load.
        if ckpt_path:
            print(f"[{model_name}] has no learned weights; ignoring --ckpt {ckpt_path}")
        model.to(device)
        model.eval()
        print(f"Using rule-based model {model_name}")
        return model
    if not ckpt_path:
        raise ValueError(f"--ckpt is required for {model_name}.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # With imbalance_strategy: weighted_loss, the class weights are stored as a
    # buffer of the loss module (criterion.weight). They only matter for the
    # training loss, and the eval model is built without them, so drop them.
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith(("criterion.", "unweighted_loss."))
    }
    # strict=True so a config/checkpoint mismatch (wrong model type, different
    # tabular_dims, different daft settings, ...) fails loudly instead of
    # silently leaving layers at random init.
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    print(f"Loaded {model_name} from {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained fusion model on a CSV split.")
    parser.add_argument("--config", required=True, help="YAML config used to train the checkpoint")
    parser.add_argument("--ckpt", help="Lightning .ckpt file to evaluate "
                        "(required except for rule-based models such as pirads_rule)")
    parser.add_argument("--val-csv", help="Split CSV to evaluate on (default: paths.valid_csv)")
    parser.add_argument("--out", required=True, help="Where to write the predictions CSV")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--num-workers", type=int, help="Override training.num_workers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for the pred_label column (default 0.5)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    # Same defaults train.py fills in.
    config.setdefault("paths", {}).setdefault("extra_valid_csv", None)
    config["paths"].setdefault("dce_dirs", [])
    config["training"].setdefault("class_weights", None)  # loss is unused here
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers

    val_csv = args.val_csv or config["paths"]["valid_csv"]
    series = [SERIES_NAME_TO_ENUM[s] for s in config["data"]["series"]]

    # mode="val" -> no augmentation. build_dataset still reads paths.train_csv,
    # which the loader uses for its train-set-derived tabular preprocessing,
    # so keep it pointing at the original training split.
    dataset = build_dataset(val_csv, config, "val", series)
    if config["data"].get("tabular_csv"):
        config["data"]["tabular_dims"] = dataset.num_tabular_features
    loader = build_dataloader(dataset, config, shuffle=False)
    print(f"Evaluating on {len(dataset)} exams from {val_csv}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(config, args.ckpt, device)

    rows = []
    with torch.no_grad():
        for batch in loader:
            volumes, tabular = to_device(batch, device)
            if getattr(model, "uses_max_pirads", False):
                # Rule-based models decide from maxPIRADS (from the split CSV).
                logits = model(volumes, tabular, batch["maxPIRADS"].to(device))
            else:
                logits = model(volumes, tabular)
            logits = logits.float().cpu()
            probs = torch.softmax(logits, dim=1)

            for i in range(logits.shape[0]):
                p = probs[i].tolist()
                rows.append({
                    "AccessionNumber": int(batch["AccessionNumber"][i]),
                    "PatientID": int(batch["PatientID"][i]),
                    "target": int(batch["label"][i]),
                    "maxPIRADS": int(batch["maxPIRADS"][i]),
                    "logit_0": logits[i, 0].item(),
                    "logit_1": logits[i, 1].item(),
                    "prob_0": p[0],
                    "prob_1": p[1],
                    "pred_label": int(p[1] >= args.threshold),
                    # save_preds-compatible columns (for late_fusion.py)
                    "Predictions": str(p),
                    "Targets": int(batch["label"][i]),
                })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} predictions to {out_path}")

    if df["target"].nunique() == 2:
        auc = roc_auc_score(df["target"], df["prob_1"])
        acc = (df["pred_label"] == df["target"]).mean()
        print(f"AUC: {auc:.4f} | accuracy @ {args.threshold}: {acc:.4f}")
    else:
        print("Only one class present in targets; skipping AUC.")


if __name__ == "__main__":
    main()