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


class ClinicalEncoder(nn.Module):
    """The feature-extractor half of mlpclinical.ClinicalMLPModel
    (Linear -> ReLU -> Linear -> ReLU), built from the same build_clinical_mlp
    so the two can never drift apart.

    frozen=True (the default, used by LateFusionFlatFrozen) disables gradients
    and locks the encoder in eval mode. frozen=False leaves it trainable so it
    can be fine-tuned, e.g. with layer-wise LR decay in LateFusionFlatLLRD.

    in_dims is the number of clinical features. If it isn't given it's read
    from the checkpoint; if both are given they must agree.
    """

    def __init__(self, in_dims=None, clinical_ckpt=None, frozen=True):
        super().__init__()
        self.frozen = frozen
        state = _load_clinical_encoder_state(clinical_ckpt) if clinical_ckpt else None
        ckpt_dims = state["0.weight"].shape[1] if state is not None else None

        if in_dims is None:
            if ckpt_dims is None:
                raise ValueError(
                    "ClinicalEncoder needs the number of clinical features: "
                    "set data.tabular_dims / data.tabular_csv, or model_weights.clinical_ckpt."
                )
            in_dims = ckpt_dims
        elif ckpt_dims is not None and ckpt_dims != in_dims:
            raise ValueError(
                f"Clinical checkpoint {clinical_ckpt} was trained on {ckpt_dims} features "
                f"but the tabular data has {in_dims}. Use the same tabular_csv as the "
                f"clinical_mlp run that produced the checkpoint."
            )

        # The encoder slice ends before the MLP's Dropout, so this value is unused.
        self.mlp = build_clinical_mlp(in_dims, dropout=0.0)[:CLINICAL_ENCODER_END]
        self.in_dims = in_dims
        self.out_dim = self.mlp[-2].out_features

        tag = "FrozenClinicalEncoder" if frozen else "ClinicalEncoder"
        if state is not None:
            self.mlp.load_state_dict(state, strict=True)
            print(f"[{tag}] Loaded {in_dims} -> {self.out_dim} encoder from {clinical_ckpt}")
        elif frozen:
            print(
                f"[{tag}] WARNING: no clinical_ckpt given, so the frozen "
                "clinical encoder is randomly initialized and will never be trained."
            )
        else:
            print(f"[{tag}] WARNING: no clinical_ckpt given; training the encoder from scratch.")

        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)

    def forward(self, x):
        return self.mlp(x)


# Backwards-compatible name for the frozen variant.
FrozenClinicalEncoder = ClinicalEncoder


class LateFusionFlatFrozen(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")
    # Subclasses that fine-tune the encoders set this to False.
    FREEZE_ENCODERS = True
    
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
        self.clinical_encoder = ClinicalEncoder(
            in_dims=config["data"].get("tabular_dims"),
            clinical_ckpt=config["model_weights"].get("clinical_ckpt"),
            frozen=self.FREEZE_ENCODERS,
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


class LateFusionFlatLLRD(LateFusionFlatFrozen):
    """LateFusionFlatFrozen with both encoders unfrozen and fine-tuned using
    layer-wise learning-rate decay (LLRD).

    Every parameter's LR is hyperparameters.learning_rate * layer_decay**depth,
    where depth counts steps back from the (randomly initialized) fusion head:

        depth 0  img_proj, fc                      (fusion head, full LR)
        depth 1  ResNet layer4  | clinical fc2
        depth 2  ResNet layer3  | clinical fc1
        depth 3  ResNet layer2
        depth 4  ResNet layer1
        depth 5  ResNet stem (conv1 + bn1)

    The two encoders decay independently from the head, so the clinical MLP's
    last layer sits at the same depth as ResNet layer4. ResNet depth is per
    stage rather than per bottleneck block, which is the usual granularity for
    CNNs and keeps the number of groups small.

    Config (all optional), under a top-level `finetune:` section:
        layer_decay      per-depth LR multiplier                 (default 0.4)
        warmup_epochs    linear LR warmup for every group        (default 1)
        freeze_bn_stats  keep ResNet BatchNorm running stats at
                         their pretrained values (affine params
                         still train)                            (default True)
    """

    FREEZE_ENCODERS = False
    RESNET_STAGES = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, config):
        super().__init__(config)
        ft = config.get("finetune") or {}
        self.layer_decay = float(ft.get("layer_decay", 0.4))
        self.warmup_epochs = float(ft.get("warmup_epochs", 1))
        self.freeze_bn_stats = bool(ft.get("freeze_bn_stats", True))
        self._warmup_steps = 0

        for param in self.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------ #
    # Parameter groups
    # ------------------------------------------------------------------ #
    def _llrd_param_groups(self):
        base_lr = self.hyperparams["learning_rate"]
        weight_decay = self.hyperparams["weight_decay"]
        decay = self.layer_decay
        n_stages = len(self.RESNET_STAGES)

        groups = {}
        seen = set()

        def add(tag, depth, param):
            # Parameters are collected per module, de-duplicated by identity:
            # the ResNet branches are also registered under alias attributes
            # (resnet_single_branch, resnet_dual_branch1, ...), and
            # named_parameters() on the whole model would report them under
            # the alias names instead of the branch names.
            if id(param) in seen or not param.requires_grad:
                return
            seen.add(id(param))
            no_wd = param.ndim <= 1  # biases and BatchNorm affine params
            key = (tag, no_wd)
            if key not in groups:
                lr = base_lr * decay**depth
                groups[key] = {
                    "name": f"{tag}/no_wd" if no_wd else tag,
                    "params": [],
                    "lr": lr,
                    "target_lr": lr,  # read by the warmup in on_train_batch_start
                    "weight_decay": 0.0 if no_wd else weight_decay,
                }
            groups[key]["params"].append(param)

        # Fusion head (randomly initialized): full LR.
        for module in (self.img_proj, self.fc):
            for param in module.parameters():
                add("head", 0, param)

        # ResNet3D branches. The same stage of every branch shares one group.
        for branch in self.branches.values():
            for name, param in branch.named_parameters():
                top = name.split(".")[0]
                if top in self.RESNET_STAGES:
                    depth = n_stages - self.RESNET_STAGES.index(top)  # layer4 -> 1
                    add(f"resnet.{top}", depth, param)
                else:  # conv1 / bn1
                    add("resnet.stem", n_stages + 1, param)

        # Clinical MLP encoder: the last Linear is closest to the head.
        linears = [m for m in self.clinical_encoder.mlp if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            depth = len(linears) - i
            for param in layer.parameters():
                add(f"clinical.fc{i + 1}", depth, param)

        missing = [
            n for n, p in self.named_parameters() if p.requires_grad and id(p) not in seen
        ]
        if missing:
            raise RuntimeError(f"LLRD: parameters not assigned to any group: {missing}")

        # Sort shallow -> deep so the printout reads head first.
        return sorted(groups.values(), key=lambda g: -g["lr"])

    def configure_optimizers(self):
        param_groups = self._llrd_param_groups()

        print(f"[LLRD] base_lr={self.hyperparams['learning_rate']:.2e} "
              f"layer_decay={self.layer_decay} weight_decay={self.hyperparams['weight_decay']}")
        for g in param_groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"[LLRD]   {g['name']:<26} lr={g['lr']:.2e}  wd={g['weight_decay']:<6g} "
                  f"params={n_params:,}")

        optimizer = torch.optim.AdamW(param_groups, lr=self.hyperparams["learning_rate"])
        # ReduceLROnPlateau multiplies every group's LR by the same factor, so
        # the layer-wise ratios are preserved after each reduction.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"],
            threshold=1e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    # ------------------------------------------------------------------ #
    # Warmup and train/eval mode
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        # Nothing is frozen here (unlike the parent). Set up the warmup length
        # in optimizer steps, which accounts for gradient accumulation.
        steps_per_epoch = self.trainer.estimated_stepping_batches / max(self.trainer.max_epochs, 1)
        self._warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        print(f"[LLRD] Fine-tuning all encoders; linear warmup over {self._warmup_steps} steps; "
              f"BatchNorm stats {'frozen' if self.freeze_bn_stats else 'updating'}.")

    def on_train_batch_start(self, batch, batch_idx):
        # Linear warmup on top of the layer-wise LRs. It stops at the end of
        # warmup, after which ReduceLROnPlateau owns the LRs (keep
        # hyperparameters.lr_patience longer than warmup_epochs so it can't act
        # while warmup is still running).
        if self.global_step < self._warmup_steps:
            scale = (self.global_step + 1) / self._warmup_steps
            for g in self.trainer.optimizers[0].param_groups:
                g["lr"] = g["target_lr"] * scale

    def train(self, mode=True):
        # Skip LateFusionFlatFrozen.train, which forces the whole ResNet into
        # eval. Here only the BatchNorm layers are optionally kept in eval, so
        # their running stats stay at the pretrained values while their affine
        # parameters (and everything else) still train.
        Base3DResNet.train(self, mode)
        if mode and getattr(self, "freeze_bn_stats", True):
            for module in self.branches.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self
