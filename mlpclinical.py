"""
Tabular-only clinical MLP (36 -> 128 -> 64 -> 2), restored from commit 2db1e00.

Trained via `python train.py --config configs/clinical_mlp.yaml`. Its first two
layers (mlp.0, mlp.2) are what the frozen clinical encoders in aug11cbam.py,
aug11earlyscalar.py and aug11latefusionflat.py load as `fc1` / `fc2` through
model_weights.clinical_ckpt; train.py exports them in that format after training.
"""
import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet


class ClinicalMLPModel(Base3DResNet):

    def __init__(self, config):
        super().__init__(config)

        # Base3DResNet always builds a single-branch ResNet3D + fc head. This
        # model never uses them, so drop them rather than carry ~46M unused
        # parameters through the optimizer and into every checkpoint.
        del self.resnet_single_branch
        del self.fc

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.mlp = nn.Sequential(
            nn.Linear(config["data"]["tabular_dims"], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2),
        )

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