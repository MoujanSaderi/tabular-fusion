"""
DAFT (Dynamic Affine Feature Map Transform) fusion of 3D MRI + clinical features,
optionally combined with late fusion of the clinical embedding.

Wolf et al., "DAFT: A Universal Module to Interweave Tabular Data and 3D Images
in CNNs", NeuroImage 2022. Reference code: https://github.com/ai-med/DAFT

DAFT block
----------
One DAFT block is inserted into the LAST bottleneck of layer4 in each ResNet3D
branch (T2, stacked ADC+B1500). Following DAFT's default (location 0), it
modulates the bottleneck's *input*, before the split into the conv path and
the identity shortcut:

    x                     (B, 2048, D, H, W)  output of layer4[-2]
    squeeze = GAP(x)      (B, 2048)
    h = [squeeze ; tab]   (B, 2048 + T)        T = raw features, or the
                                               64-dim frozen clinical embedding
    h -> Linear -> ReLU -> Linear  (bias-free, bottleneck = (2048 + T) / 7)
      -> scale alpha (B, 2048), shift beta (B, 2048)
    x' = alpha * x + beta  ->  layer4[-1] (conv path + shortcut) -> avgpool

Heads (daft.late_fusion, daft.init_from)
----------------------------------------
late_fusion: false
    fc: 4096 -> 256 -> 2 on image features only (the original DAFT setup).
    Same layout as the baseline TriSeriesModel, so its fc loads from the
    baseline checkpoint.

late_fusion: true, init_from: baseline
    fc: (4096 + 64) -> 256 -> 2 on [image features ; frozen clinical embedding].
    The first layer's image columns are copied from the baseline fc and its 64
    clinical columns are zero-initialized, so the late path starts with no
    effect and grows in during training.

late_fusion: true, init_from: latefusion
    The head of aug11latefusionflat.LateFusionFlatFrozen:
    img_proj 4096 -> 256, then fc (256 + 64) -> 128 -> 2. EVERYTHING except the
    DAFT MLPs (ResNet branches, img_proj, fc, clinical encoder) is loaded from
    the late-only checkpoint given as model_weights.model_ckpt, so the backbone
    always matches the features the head was trained on. Missing or
    mis-shaped weights are an error, not a silent skip.

In every configuration DAFT starts as an exact identity (identity_init), so at
step 0 the model reproduces the checkpoint it was initialized from: the
baseline, or the late-only model.

identity_init (default true): the DAFT MLP's output layer is zero-initialized
and the scale is parametrized around 1 (1 + a, 1 + tanh(a), 2 * sigmoid(a)).
With false, the reference DAFT parametrization and default init are used.

Training schedule
-----------------
Trainable: the DAFT MLPs and the head (img_proj + fc); optionally the host
bottleneck's convs (train_host_block). BatchNorm running stats stay frozen.

freeze_head_epochs N  keep the head frozen for the first N epochs, so all the
                      learning pressure goes to DAFT, then unfreeze it.
head_lr_scale         LR multiplier for the head relative to DAFT (e.g. 0.1
                      when the head is already trained, init_from: latefusion).
late_clinical_dropout probability, per sample and training step, of zeroing
                      the clinical embedding on the LATE path only, so the
                      model can't route all clinical information through it.

Config (all optional), under a top-level `daft:` section:
    tabular_input          "raw" | "encoder"                     (default "raw")
    bottleneck_factor                                            (default 7)
    scale / shift                                                (default true)
    activation             "linear" | "tanh" | "sigmoid"         (default "linear")
    identity_init                                                (default true)
    train_host_block                                             (default false)
    late_fusion                                                  (default false)
    init_from              "baseline" | "latefusion"             (default "baseline")
    late_clinical_dropout                                        (default 0.0)
    freeze_head_epochs                                           (default 0)
    head_lr_scale                                                (default 1.0)

Run with: python train.py --config configs/daft.yaml   (or daft_late*.yaml)
"""
import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch

from aug11latefusionflat import ClinicalEncoder

DAFT_DEFAULTS = {
    "tabular_input": "raw",
    "bottleneck_factor": 7.0,
    "scale": True,
    "shift": True,
    "activation": "linear",
    "identity_init": True,
    "train_host_block": False,
    "late_fusion": False,
    "init_from": "baseline",
    "late_clinical_dropout": 0.0,
    "freeze_head_epochs": 0,
    "head_lr_scale": 1.0,
}


class DAFTModule(nn.Module):
    """Computes a per-channel affine transform of a 3D feature map from its
    global-average-pooled channels concatenated with the tabular vector."""

    def __init__(
        self,
        channels,
        tabular_dim,
        bottleneck_factor=7.0,
        scale=True,
        shift=True,
        activation="linear",
        identity_init=True,
    ):
        super().__init__()
        if not (scale or shift):
            raise ValueError("DAFT needs at least one of scale / shift enabled.")
        if activation not in ("linear", "tanh", "sigmoid"):
            raise ValueError(f"Unknown DAFT activation '{activation}'.")

        self.channels = channels
        self.use_scale = scale
        self.use_shift = shift
        self.activation = activation
        self.identity_init = identity_init

        bottleneck_dim = max(int((channels + tabular_dim) / bottleneck_factor), 1)
        out_dim = channels * (int(scale) + int(shift))
        # Bias-free, as in the reference implementation.
        self.aux = nn.Sequential(
            nn.Linear(channels + tabular_dim, bottleneck_dim, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, out_dim, bias=False),
        )
        if identity_init:
            nn.init.zeros_(self.aux[-1].weight)

        # Detached summary of the last forward pass, for logging.
        self.last_stats = {}

    def _scale(self, raw):
        if self.identity_init:  # identity when raw == 0
            if self.activation == "linear":
                return 1.0 + raw
            if self.activation == "tanh":
                return 1.0 + torch.tanh(raw)
            return 2.0 * torch.sigmoid(raw)
        if self.activation == "linear":
            return raw
        if self.activation == "tanh":
            return torch.tanh(raw)
        return torch.sigmoid(raw)

    def forward(self, x, tabular):
        squeeze = x.mean(dim=(2, 3, 4))
        params = self.aux(torch.cat([squeeze, tabular], dim=1))

        if self.use_scale and self.use_shift:
            raw_scale, shift = params.split(self.channels, dim=1)
            scale = self._scale(raw_scale)
        elif self.use_scale:
            scale, shift = self._scale(params), None
        else:
            scale, shift = None, params

        with torch.no_grad():
            if scale is not None:
                self.last_stats["scale_dev"] = (scale - 1).abs().mean()
            if shift is not None:
                self.last_stats["shift_abs"] = shift.abs().mean()

        view = (x.size(0), self.channels, 1, 1, 1)
        if scale is not None:
            x = x * scale.view(view)
        if shift is not None:
            x = x + shift.view(view)
        return x


class DAFTBottleneck(Bottleneck):
    """A Bottleneck whose input is modulated by DAFT (reference location 0).

    Keeps Bottleneck's attribute names (conv1, bn1, ..., downsample), so
    pretrained checkpoints load into it by name unchanged; only the `daft.*`
    parameters are new.
    """

    def __init__(self, inplanes, planes, tabular_dim, daft_kwargs,
                 stride=1, dilation=1, downsample=None):
        super().__init__(inplanes, planes, stride=stride, dilation=dilation,
                         downsample=downsample)
        self.daft = DAFTModule(inplanes, tabular_dim, **daft_kwargs)

    def forward(self, x, tabular):
        return super().forward(self.daft(x, tabular))


class DAFTResNetBranch(ResNetBranch):
    """ResNetBranch with its final layer4 bottleneck swapped for a DAFTBottleneck."""

    def __init__(self, block, layers, in_chans, tabular_dim, daft_kwargs):
        super().__init__(block, layers, in_chans)
        host = self.layer4[-1]
        if not isinstance(host, Bottleneck):
            raise TypeError("DAFTResNetBranch expects Bottleneck blocks.")
        daft_block = DAFTBottleneck(
            host.conv1.in_channels,
            host.conv1.out_channels,
            tabular_dim,
            daft_kwargs,
            stride=host.stride,
            dilation=host.dilation,
            downsample=host.downsample,
        )
        missing, unexpected = daft_block.load_state_dict(host.state_dict(), strict=False)
        assert not unexpected and all(k.startswith("daft.") for k in missing)
        self.layer4[-1] = daft_block

    @property
    def daft_block(self):
        return self.layer4[-1]

    def forward(self, x, tabular):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4[:-1](out)
        out = self.layer4[-1](out, tabular)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)


class DAFTFusionModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        cfg = {**DAFT_DEFAULTS, **(config.get("daft") or {})}
        self.daft_cfg = cfg
        self.tabular_input = cfg["tabular_input"]
        self.late_fusion = bool(cfg["late_fusion"])
        self.init_from = cfg["init_from"]
        self.train_host_block = bool(cfg["train_host_block"])
        self.late_clinical_dropout = float(cfg["late_clinical_dropout"])
        self.freeze_head_epochs = int(cfg["freeze_head_epochs"])
        self.head_lr_scale = float(cfg["head_lr_scale"])
        self._pretrained_loaded = False
        self._head_frozen = None

        if self.tabular_input not in ("raw", "encoder"):
            raise ValueError(f"daft.tabular_input must be 'raw' or 'encoder', got {self.tabular_input!r}")
        if self.init_from not in ("baseline", "latefusion"):
            raise ValueError(f"daft.init_from must be 'baseline' or 'latefusion', got {self.init_from!r}")
        if self.init_from == "latefusion" and not self.late_fusion:
            raise ValueError("daft.init_from: latefusion requires daft.late_fusion: true.")
        if not 0.0 <= self.late_clinical_dropout < 1.0:
            raise ValueError("daft.late_clinical_dropout must be in [0, 1).")
        if self.late_clinical_dropout and not self.late_fusion:
            raise ValueError("daft.late_clinical_dropout only applies with daft.late_fusion: true.")

        # Frozen clinical encoder: needed for the late path and/or as DAFT's input.
        tabular_dims = config["data"].get("tabular_dims")
        if self.late_fusion or self.tabular_input == "encoder":
            clinical_ckpt = config["model_weights"].get("clinical_ckpt")
            if self.init_from == "latefusion" and clinical_ckpt is None:
                print("[DAFT] clinical_ckpt is null; the clinical encoder will be "
                      "loaded from the late-only model_ckpt instead.")
            self.clinical_encoder = ClinicalEncoder(
                in_dims=tabular_dims, clinical_ckpt=clinical_ckpt, frozen=True
            )
            clin_dim = self.clinical_encoder.out_dim
        else:
            self.clinical_encoder = None
            clin_dim = 0

        if self.tabular_input == "encoder":
            daft_tab_dim = clin_dim
        else:
            if tabular_dims is None:
                raise ValueError(
                    "DAFT with tabular_input 'raw' needs data.tabular_dims "
                    "(train.py fills it in from data.tabular_csv)."
                )
            daft_tab_dim = int(tabular_dims)

        daft_kwargs = {
            "bottleneck_factor": float(cfg["bottleneck_factor"]),
            "scale": bool(cfg["scale"]),
            "shift": bool(cfg["shift"]),
            "activation": cfg["activation"],
            "identity_init": bool(cfg["identity_init"]),
        }

        # Accept series either as SeriesType enums or as plain strings.
        self.series = [
            s if isinstance(s, str) else s.value["key"] for s in config["data"]["series"]
        ]
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"] and set(
            self.DWI_KEYS
        ).issubset(self.series)

        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if "adc_b1500" not in self.branches:
                    self.branches["adc_b1500"] = DAFTResNetBranch(
                        Bottleneck, [3, 4, 6, 3], 2, daft_tab_dim, daft_kwargs
                    )
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.feature_dim += 2048
                continue
            self.branches[key] = DAFTResNetBranch(
                Bottleneck, [3, 4, 6, 3], 1, daft_tab_dim, daft_kwargs
            )
            self.branch_specs.append((key, (key,)))
            self.feature_dim += 2048

        # Heads. Layer names/indices match the checkpoint each mode loads from.
        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        if self.late_fusion and self.init_from == "latefusion":
            # == LateFusionFlatFrozen
            self.img_proj = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.ReLU())
            self.fc = nn.Sequential(
                nn.Linear(256 + clin_dim, 128),
                nn.ReLU(),
                self.dropout,
                nn.Linear(128, 2),
            )
        else:
            # == baseline TriSeriesModel, plus clinical columns if late_fusion
            self.img_proj = None
            self.fc = nn.Sequential(
                nn.Linear(self.feature_dim + (clin_dim if self.late_fusion else 0), 256),
                nn.ReLU(inplace=False),
                self.dropout,
                nn.Linear(256, 2),
            )
            if self.late_fusion:
                with torch.no_grad():
                    self.fc[0].weight[:, self.feature_dim:] = 0.0

        # Replace Base3DResNet's unused ResNet with an alias to a real branch
        # (also keeps state_dict keys compatible with the checkpoints).
        if "axt2" in self.branches:
            self.resnet_single_branch = self.branches["axt2"]
        elif self.branch_specs:
            self.resnet_single_branch = self.branches[self.branch_specs[0][0]]
        if self.stack_adc_b1500:
            self.resnet_dual_branch1 = self.branches["adc_b1500"]
        else:
            if "adc" in self.branches:
                self.resnet_dual_branch1 = self.branches["adc"]
            if "b1500" in self.branches:
                self.resnet_dual_branch2 = self.branches["b1500"]

        self._apply_freeze()
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = (
            "image-only fc" if not self.late_fusion
            else f"late fusion (init_from={self.init_from}, "
                 f"late_clinical_dropout={self.late_clinical_dropout})"
        )
        print(f"[DAFT] tabular_input={self.tabular_input} (dim {daft_tab_dim}), {daft_kwargs}, "
              f"head: {head}, train_host_block={self.train_host_block}, "
              f"freeze_head_epochs={self.freeze_head_epochs}, head_lr_scale={self.head_lr_scale}, "
              f"trainable params={n_train:,}")

    # ------------------------------------------------------------------ #
    # Initialization from a checkpoint (called by train.build_model)
    # ------------------------------------------------------------------ #
    def load_pretrained_checkpoint(self, checkpoint):
        """Load every non-DAFT weight from `checkpoint`, failing loudly if any is
        missing or mis-shaped. Which weights are expected depends on the mode:

          baseline   : ResNet branches + fc (the clinical encoder comes from
                       clinical_ckpt; with late_fusion the first fc layer's
                       image columns come from the baseline fc.0 and the
                       clinical columns stay zero)
          latefusion : ResNet branches + img_proj + fc + clinical encoder
        """
        src = checkpoint.get("state_dict", checkpoint)
        own = self.state_dict(keep_vars=True)

        # Group aliased keys (resnet_single_branch.* == branches.axt2.*, ...) by
        # tensor so a checkpoint only needs to contain one name per tensor.
        groups = {}
        for key, tensor in own.items():
            groups.setdefault(id(tensor), (tensor, []))[1].append(key)

        special = set()
        if self.late_fusion and self.init_from == "baseline":
            special.add("fc.0.weight")

        problems, n_loaded = [], 0
        with torch.no_grad():
            for tensor, keys in groups.values():
                if any(".daft." in k for k in keys) or special.intersection(keys):
                    continue
                if self.init_from == "baseline" and any(
                    k.startswith("clinical_encoder.") for k in keys
                ):
                    continue
                src_key = next((k for k in keys if k in src), None)
                if src_key is None:
                    problems.append(f"missing: {keys[0]}")
                elif src[src_key].shape != tensor.shape:
                    problems.append(
                        f"shape mismatch: {src_key} {tuple(src[src_key].shape)} "
                        f"in checkpoint vs {tuple(tensor.shape)} in model"
                    )
                else:
                    if src_key.startswith("clinical_encoder.") and not torch.allclose(
                        tensor, src[src_key].to(tensor)
                    ):
                        print(f"[DAFT] WARNING: {src_key} in model_ckpt differs from "
                              "clinical_ckpt; using model_ckpt's (what the head was trained with).")
                    tensor.copy_(src[src_key])
                    n_loaded += 1

            if "fc.0.weight" in special:
                w = self.fc[0].weight
                src_w = src.get("fc.0.weight")
                if src_w is None or src_w.shape != (w.shape[0], self.feature_dim):
                    problems.append(
                        "fc.0.weight: expected baseline shape "
                        f"{(w.shape[0], self.feature_dim)}, got "
                        f"{None if src_w is None else tuple(src_w.shape)}"
                    )
                else:
                    w[:, : self.feature_dim] = src_w
                    w[:, self.feature_dim:] = 0.0
                    n_loaded += 1

        if problems:
            hint = (
                "init_from: latefusion expects a LateFusionFlatFrozen checkpoint"
                if self.init_from == "latefusion"
                else "init_from: baseline expects a baseline TriSeriesModel checkpoint"
            )
            raise RuntimeError(
                f"[DAFT] model_ckpt doesn't match this configuration ({hint}):\n  "
                + "\n  ".join(problems[:20])
                + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else "")
            )
        self._pretrained_loaded = True
        print(f"[DAFT] Loaded {n_loaded} tensors from model_ckpt (init_from={self.init_from}); "
              "DAFT blocks start at identity.")
        return self

    # ------------------------------------------------------------------ #
    # Freezing and optimization
    # ------------------------------------------------------------------ #
    def _head_modules(self):
        return [m for m in (self.img_proj, self.fc) if m is not None]

    def _daft_modules(self):
        return [
            branch.daft_block if self.train_host_block else branch.daft_block.daft
            for branch in self.branches.values()
        ]

    def _set_head_frozen(self, frozen):
        for module in self._head_modules():
            for param in module.parameters():
                param.requires_grad = not frozen
        if frozen != self._head_frozen:
            print(f"[DAFT] epoch {self.current_epoch}: "
                  f"head {'frozen' if frozen else 'trainable'}")
        self._head_frozen = frozen

    def _apply_freeze(self):
        # By module, not parameter name: the branches are also registered
        # under alias attributes, which named_parameters() reports instead.
        for param in self.parameters():
            param.requires_grad = False
        for module in self._daft_modules() + self._head_modules():
            for param in module.parameters():
                param.requires_grad = True

    def on_train_start(self):
        if self.init_from == "latefusion" and not self._pretrained_loaded:
            raise RuntimeError(
                "[DAFT] init_from: latefusion, but no checkpoint was loaded. Set "
                "model_weights.load_weights: true and model_weights.model_ckpt to "
                "the late-only (latefusion_flat) checkpoint."
            )
        if not self._pretrained_loaded:
            print("[DAFT] WARNING: no model_ckpt loaded; the frozen ResNet "
                  "branches are randomly initialized.")
        self._apply_freeze()

    def on_train_epoch_start(self):
        self._set_head_frozen(self.current_epoch < self.freeze_head_epochs)

    def train(self, mode=True):
        # Keep the ResNet branches (incl. BatchNorm running stats) in eval;
        # the DAFT MLPs have no dropout/BN, so this doesn't affect them.
        super().train(mode)
        self.branches.eval()
        return self

    def configure_optimizers(self):
        # Groups are built from modules, not requires_grad, so head parameters
        # frozen for the first epochs are already in the optimizer when they
        # unfreeze. While frozen they get no gradient and AdamW skips them.
        lr = self.hyperparams["learning_rate"]
        groups = [
            {"name": "daft", "lr": lr,
             "params": [p for m in self._daft_modules() for p in m.parameters()]},
            {"name": "head", "lr": lr * self.head_lr_scale,
             "params": [p for m in self._head_modules() for p in m.parameters()]},
        ]
        optimizer = torch.optim.AdamW(groups, lr, weight_decay=self.hyperparams["weight_decay"])
        # ReduceLROnPlateau scales every group by the same factor, keeping the ratio.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"],
            threshold=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # ------------------------------------------------------------------ #
    # Forward / steps
    # ------------------------------------------------------------------ #
    def forward(self, data_dict, tabular_features):
        tabular_features = tabular_features.float()
        clin_emb = (
            self.clinical_encoder(tabular_features) if self.clinical_encoder is not None else None
        )
        cond = clin_emb if self.tabular_input == "encoder" else tabular_features

        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs, cond))
        img = torch.cat(features, dim=1)

        if not self.late_fusion:
            return self.fc(img)

        late_clin = clin_emb
        if self.training and self.late_clinical_dropout > 0:
            keep = torch.rand(late_clin.size(0), 1, device=late_clin.device)
            late_clin = late_clin * (keep >= self.late_clinical_dropout).to(late_clin.dtype)
        if self.img_proj is not None:
            img = self.img_proj(img)
        return self.fc(torch.cat([img, late_clin], dim=1))

    def _log_daft_stats(self):
        # How far each DAFT block is from identity; both start at 0 with identity_init.
        for name, branch in self.branches.items():
            for stat, value in branch.daft_block.daft.last_stats.items():
                self.log(f"daft/{name}_{stat}", value, on_step=False, on_epoch=True)

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
            self._log_daft_stats()
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
