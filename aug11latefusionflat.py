import torch
import torch.nn as nn
from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch

from mlpclinical import build_clinical_mlp

# Index of the last hidden ReLU in build_clinical_mlp; the encoder keeps
# layers [0, CLINICAL_ENCODER_END) and drops dropout + the 64 -> 2 classifier.
CLINICAL_ENCODER_END = 4


def _load_clinical_encoder_state(clinical_ckpt):
    """Read a clinical-MLP checkpoint in any of the formats this repo produces
    and return it keyed like the encoder's nn.Sequential ("0.weight", "2.bias", ...):

      * clinical_encoder.pt exported by train.py   -> {fc1.*, fc2.*}
      * a raw ClinicalMLPModel Lightning checkpoint -> {"state_dict": {mlp.0.*, ...}}
      * a plain ClinicalMLPModel state dict         -> {mlp.0.*, mlp.2.*, ...}
    """
    state = torch.load(clinical_ckpt, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)

    renames = {"fc1.": "0.", "fc2.": "2.", "mlp.0.": "0.", "mlp.2.": "2."}
    out = {}
    for key, value in state.items():
        for old, new in renames.items():
            if key.startswith(old):
                out[new + key[len(old):]] = value
                break
    expected = {"0.weight", "0.bias", "2.weight", "2.bias"}
    if set(out) != expected:
        raise KeyError(
            f"{clinical_ckpt} doesn't look like a clinical MLP checkpoint: "
            f"found keys {sorted(state)[:8]}..."
        )
    return out


class FrozenClinicalEncoder(nn.Module):
    """The feature-extractor half of mlpclinical.ClinicalMLPModel
    (Linear -> ReLU -> Linear -> ReLU), built from the same build_clinical_mlp
    so the two can never drift apart. Fully frozen and locked in eval mode.

    in_dims is the number of clinical features. If it isn't given it's read
    from the checkpoint; if both are given they must agree.
    """

    def __init__(self, in_dims=None, clinical_ckpt=None):
        super().__init__()

        state = _load_clinical_encoder_state(clinical_ckpt) if clinical_ckpt else None
        ckpt_dims = state["0.weight"].shape[1] if state is not None else None

        if in_dims is None:
            if ckpt_dims is None:
                raise ValueError(
                    "FrozenClinicalEncoder needs the number of clinical features: "
                    "set data.tabular_dims / data.tabular_csv, or model_weights.clinical_ckpt."
                )
            in_dims = ckpt_dims
        elif ckpt_dims is not None and ckpt_dims != in_dims:
            raise ValueError(
                f"Clinical checkpoint {clinical_ckpt} was trained on {ckpt_dims} features "
                f"but the tabular data has {in_dims}. Use the same tabular_csv as the "
                f"clinical_mlp run that produced the checkpoint."
            )

        # Dropout is a no-op in a frozen, eval-only encoder, so its value is irrelevant.
        self.mlp = build_clinical_mlp(in_dims, dropout=0.0)[:CLINICAL_ENCODER_END]
        self.in_dims = in_dims
        self.out_dim = self.mlp[-2].out_features

        if state is not None:
            self.mlp.load_state_dict(state, strict=True)
            print(f"[FrozenClinicalEncoder] Loaded {in_dims} -> {self.out_dim} encoder from {clinical_ckpt}")
        else:
            print(
                "[FrozenClinicalEncoder] WARNING: no clinical_ckpt given, so the frozen "
                "clinical encoder is randomly initialized and will never be trained."
            )

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x):
        return self.mlp(x)


class LateFusionFlatFrozen(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")
    
    def __init__(self, config):
        super().__init__(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        # Accept series either as SeriesType enums or as the plain strings
        # train.py passes straight through from the YAML config.
        self.series = [
            s if isinstance(s, str) else s.value["key"] for s in config["data"]["series"]
        ]
        
        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(series_set)
        
        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        self._dwi_branch_added = False
        
        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue
            self.branches[key] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))
            self.feature_dim += 2048
        
        self.img_proj = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.ReLU())
        
        # tabular_dims is filled in by train.py from the tabular CSV; if it's
        # absent the encoder falls back to the checkpoint's input width.
        self.clinical_encoder = FrozenClinicalEncoder(
            in_dims=config["data"].get("tabular_dims"),
            clinical_ckpt=config["model_weights"].get("clinical_ckpt"),
        )
        
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(256 + self.clinical_encoder.out_dim, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2),
        )
        
        if "axt2" in self.branches:
            self.resnet_single_branch = self.branches["axt2"]
        elif self.branch_specs:
            self.resnet_single_branch = self.branches[self.branch_specs[0][0]]
        
        if self.stack_adc_b1500:
            if "adc_b1500" in self.branches:
                self.resnet_dual_branch1 = self.branches["adc_b1500"]
        else:
            if "adc" in self.branches:
                self.resnet_dual_branch1 = self.branches["adc"]
            if "b1500" in self.branches:
                self.resnet_dual_branch2 = self.branches["b1500"]
    
    def on_train_start(self):
        print("Freezing ResNet and clinical encoder...")
        # Freeze by module, not by parameter name: the axt2 branch is also
        # registered as self.resnet_single_branch, and named_parameters()
        # reports shared parameters under that alias only, so a name-based
        # "branches" check silently left the whole T2 ResNet trainable.
        for param in self.parameters():
            param.requires_grad = True
        for module in (self.branches, self.clinical_encoder):
            for param in module.parameters():
                param.requires_grad = False
        self.branches.eval()

    def train(self, mode=True):
        # Lightning calls .train() again after every validation loop; keep the
        # frozen ResNet branches in eval so their BatchNorm stats don't drift.
        super().train(mode)
        self.branches.eval()
        return self
    
    def forward(self, data_dict, tabular_features):
        img_feats = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            img_feats.append(self.branches[branch_name](inputs))
        
        img_vec = torch.cat(img_feats, dim=1)
        img_emb = self.img_proj(img_vec)
        clin_emb = self.clinical_encoder(tabular_features)
        if not hasattr(self, "_debug_printed"):
            print(f"DEBUG clin_emb sample: {clin_emb[0][:5]}")
            print(f"DEBUG clin_emb std across batch: {clin_emb.std(dim=0).mean().item():.6f}")
            print(f"DEBUG img_emb std across batch: {img_emb.std(dim=0).mean().item():.6f}")
            self._debug_printed = True
        
        fused = torch.cat([img_emb, clin_emb], dim=1)
        out = self.fc(fused)
        return out
    
    def training_step(self, batch, batch_idx):
        data_dict = batch["volume_data_dict"]
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(data_dict, tabular)
        loss = self.criterion(logits, target)
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch["volume_data_dict"]
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(data_dict, tabular)
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
        data_dict = batch["volume_data_dict"]
        tabular = batch["tabular_features"]
        target = batch["label"]
        logits = self(data_dict, tabular)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            print("Test Loss", loss)
        return {"test_loss": loss}
