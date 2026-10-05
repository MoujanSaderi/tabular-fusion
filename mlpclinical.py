"""
Tabular-only clinical MLP (tabular_dims -> 128 -> 64 -> 2), restored from commit 2db1e00.

Trained via `python train.py --config configs/clinical_mlp.yaml`. Its first two
layers (mlp.0, mlp.2) are what the frozen clinical encoders in aug11cbam.py,
aug11earlyscalar.py and aug11latefusionflat.py load as `fc1` / `fc2` through
model_weights.clinical_ckpt; train.py exports them in that format after training.
"""
import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet

# Hidden widths of the clinical MLP. The fusion models reuse everything up to
# the last hidden layer as their frozen clinical encoder, so CLINICAL_HIDDEN_DIMS[-1]
# is also the size of the clinical embedding they fuse.
CLINICAL_HIDDEN_DIMS = (128, 64)


def build_clinical_mlp(in_dims, dropout, num_classes=2):
    """Single source of truth for the clinical MLP architecture.

    Layer indices are part of the checkpoint format: mlp.0 and mlp.2 are the
    two hidden Linear layers (exported as fc1 / fc2 by train.py), mlp.5 is the
    classifier. Keep the order unchanged so old checkpoints still load.
    """
    h1, h2 = CLINICAL_HIDDEN_DIMS
    return nn.Sequential(
        nn.Linear(in_dims, h1),   # 0
        nn.ReLU(),                # 1
        nn.Linear(h1, h2),        # 2
        nn.ReLU(),                # 3
        nn.Dropout(p=dropout),    # 4
        nn.Linear(h2, num_classes),  # 5
    )


class ClinicalMLPModel(Base3DResNet):

    def __init__(self, config):
        super().__init__(config)

        # Base3DResNet always builds a single-branch ResNet3D + fc head. This
        # model never uses them, so drop them rather than carry ~46M unused
        # parameters through the optimizer and into every checkpoint.
        del self.resnet_single_branch
        del self.fc

        self.mlp = build_clinical_mlp(
            config["data"]["tabular_dims"], config["hyperparameters"]["dropout"]
        )
        self.dropout = self.mlp[4]  # kept as an attribute for backwards compatibility

    def forward(self, data_dict, tabular_features):
        return self.mlp(tabular_features)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:  # check only the first batch
            for k, v in batch.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    print(k, tuple(v.shape),
                        "nan:", torch.isnan(v).sum().item(),
                        "inf:", torch.isinf(v).sum().item(),
                        "min/max:", v.nan_to_num().min().item(), v.nan_to_num().max().item())
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.criterion(logits, target)
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            if self.log_configs["log_run"]:
                self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(batch["volume_data_dict"], tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            print("Test Loss", loss)
        return {"test_loss": loss}