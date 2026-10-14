"""
CBAM + late fusion: aug11cbam.TriSeriesModel with the frozen clinical embedding
ALSO fed directly to the classifier head.

In plain CBAM the clinical embedding only produces gates that rescale the
frozen imaging features; the head never sees it. Here it gets a direct,
additive path to the logits as well:

    img  = [branch_T2 ; branch_ADC+B1500]     (B, 4096)  CBAM-gated features
    h    = img_proj(img)                      (B, 256)   Linear + ReLU
    clin = frozen clinical encoder            (B, 64)
    out  = fc([h ; clin])                     (B, 2)     Linear(320, 128) -> ReLU
                                                         -> dropout -> Linear(128, 2)

The head (img_proj + fc) is built exactly like aug11latefusionflat.
LateFusionFlatFrozen's, so the only architectural difference from late fusion
is the CBAM modules inside the frozen backbone. As in late fusion, the
clinical embedding meets the image features in a hidden layer, so the head
itself can also model image/clinical interactions.

Initialization (cbam_late.init_from)
------------------------------------
cbam       Backbone, CBAM modules and clinical encoder are loaded from a
           trained CBAM checkpoint (aug11cbam). CBAM's first head layer
           (fc.0, 4096 -> 256, trained on the gated features) becomes
           img_proj.0. The fusion MLP `fc` has no counterpart in CBAM and starts
           from PyTorch's default init, as in late fusion; CBAM's last layer
           (fc.3) is discarded. So step 0 does NOT reproduce the CBAM model.
cbam_late  Resume from a checkpoint of THIS model (e.g. phase 1 -> phase 2).
           Everything loads by name.
baseline   Backbone + img_proj (from fc.0) from the plain baseline checkpoint;
           fc at default init; CBAM modules
           keep their identity init; clinical encoder from clinical_ckpt.
           For the joint-training comparison arm.

All three load strictly: missing, mis-shaped or unused CBAM weights are an
error (e.g. a `cbam:` section that doesn't match the checkpoint's stages).

Training schedule
-----------------
Each phase is its own run with its own config; the attention is either frozen
or trainable for the WHOLE run, never switched partway through.
train_attention          false: CBAM modules frozen, only the head trains
                                (phase 1, configs/cbam_late_phase1.yaml).
                          true: CBAM modules train alongside the head
                                (phase 2 or the joint arm,
                                configs/cbam_late_phase2.yaml / _joint.yaml).
attention_lr_scale       CBAM-module LR = learning_rate * attention_lr_scale.
late_clinical_dropout    per-sample probability of zeroing the clinical
                         embedding on the LATE path only, during training.
                         The gates always see it.

Monitoring (cbam_late.log_gate_stats)
-------------------------------------
On every validation epoch, for each branch and ResNet stage, logs
    gates/<branch>_<stage>_<chan|spat>_dev     mean |gate - neutral|
                                               (neutral = 1 with identity_init)
    gates/<branch>_<stage>_<chan|spat>_spread  std of the gate ACROSS PATIENTS,
                                               averaged over channels / voxels
plus late/clin_weight_norm and late/img_weight_norm (the clinical and image
input columns of fc's first layer).
A channel-gate spread near zero means the attention no longer depends on the
patient, i.e. it has collapsed to a fixed reweighting and the late path is
doing all the clinical work. While the CBAM modules are frozen (phase 1) the
gate statistics are the CBAM checkpoint's own and stay constant: that's the
reference to compare phase 2 against.

Config, under a top-level `cbam_late:` section (all optional):
    init_from                "cbam" | "cbam_late" | "baseline"   (default "cbam")
    train_attention                                              (default false)
    attention_lr_scale                                           (default 0.1)
    late_clinical_dropout                                        (default 0.0)
    log_gate_stats                                               (default true)
The `cbam:` section is read exactly as in aug11cbam and must match the CBAM
checkpoint's settings.

Run with: python train.py --config configs/cbam_late_phase1.yaml
"""
import torch
import torch.nn as nn

from aug11cbam import ClinicalChannelAttention
from aug11cbam import ClinicalSpatialAttention
from aug11cbam import TriSeriesModel
from aug11cbam import _gate

CBAM_LATE_DEFAULTS = {
    "init_from": "cbam",
    "train_attention": False,
    "attention_lr_scale": 0.1,
    "late_clinical_dropout": 0.0,
    "log_gate_stats": True,
}
INIT_MODES = ("cbam", "cbam_late", "baseline")
# Head widths, identical to aug11latefusionflat.LateFusionFlatFrozen.
IMG_EMB_DIM = 256     # img_proj output; also the width of CBAM's / baseline's fc.0
FUSION_HIDDEN = 128   # fc hidden layer


def _is_cbam_key(key):
    return ".cbam." in key or ".post_cbam." in key or key.startswith("post_cbam.")


# --------------------------------------------------------------------------- #
# Gate recomputation for monitoring. These mirror ClinicalChannelAttention /
# ClinicalSpatialAttention.forward in aug11cbam.py, which return the reweighted
# features rather than the gates; keep them in sync if those change.
# --------------------------------------------------------------------------- #
def _channel_gate(module, x, clin):
    avg = x.mean(dim=(2, 3, 4))
    mx = x.amax(dim=(2, 3, 4))
    logits = module.mlp(torch.cat([avg, clin], dim=1)) + module.mlp(torch.cat([mx, clin], dim=1))
    return _gate(logits, module.identity_init)  # (B, C)


def _spatial_gate(module, x, clin):
    pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
    if module.film is not None:
        gamma, beta = module.film(clin).view(-1, 2, 2).unbind(dim=1)
        pooled = pooled * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
    return _gate(module.conv(pooled), module.identity_init)  # (B, 1, D, H, W)


class CBAMLateFusion(TriSeriesModel):
    def __init__(self, config):
        # Builds the branches with CBAM modules, the frozen clinical encoder,
        # and CBAM's image-only fc head, which is replaced below.
        super().__init__(config)

        user_cfg = config.get("cbam_late") or {}
        if "freeze_attention_epochs" in user_cfg:
            raise ValueError(
                "cbam_late.freeze_attention_epochs was removed: phases 1 and 2 are separate "
                "runs. Use cbam_late.train_attention: false (phase 1) or true (phase 2 / joint)."
            )
        unknown = set(user_cfg) - set(CBAM_LATE_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown cbam_late keys: {sorted(unknown)}")
        cfg = {**CBAM_LATE_DEFAULTS, **user_cfg}
        self.init_from = cfg["init_from"]
        if not isinstance(cfg["train_attention"], bool):
            raise ValueError("cbam_late.train_attention must be true or false.")
        self.train_attention = cfg["train_attention"]
        self.attention_lr_scale = float(cfg["attention_lr_scale"])
        self.late_clinical_dropout = float(cfg["late_clinical_dropout"])
        self.log_gate_stats = bool(cfg["log_gate_stats"])
        if self.init_from not in INIT_MODES:
            raise ValueError(f"cbam_late.init_from must be one of {INIT_MODES}, got {self.init_from!r}")
        if not 0.0 <= self.late_clinical_dropout < 1.0:
            raise ValueError("cbam_late.late_clinical_dropout must be in [0, 1).")

        clin_dim = self.clinical_encoder.out_dim
        # Same head as LateFusionFlatFrozen (same layers, widths, dropout
        # placement and default init), replacing CBAM's image-only head.
        self.img_proj = nn.Sequential(nn.Linear(self.feature_dim, IMG_EMB_DIM), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(IMG_EMB_DIM + clin_dim, FUSION_HIDDEN),
            nn.ReLU(),
            self.dropout,
            nn.Linear(FUSION_HIDDEN, 2),
        )

        self._clinical_ckpt_given = bool(config["model_weights"].get("clinical_ckpt"))
        self._pretrained_loaded = False
        self._late_ready = True
        self._apply_freeze()

        self._gate_sums = {}
        self._collect_gates = False
        self._register_gate_hooks()

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_cbam = sum(p.numel() for m in self._cbam_modules() for p in m.parameters())
        print(f"[CBAM+late] init_from={self.init_from}, "
              f"train_attention={self.train_attention}, "
              f"attention_lr_scale={self.attention_lr_scale}, "
              f"late_clinical_dropout={self.late_clinical_dropout}; "
              f"{len(self._cbam_modules())} CBAM modules ({n_cbam:,} params), "
              f"head {self.feature_dim} -> {IMG_EMB_DIM} -> [{IMG_EMB_DIM} + {clin_dim}] "
              f"-> {FUSION_HIDDEN} -> 2 (late-fusion head); "
              f"trainable params now={n_train:,}")

    # ------------------------------------------------------------------ #
    # Initialization from a checkpoint (called by train.build_model)
    # ------------------------------------------------------------------ #
    def load_pretrained_checkpoint(self, checkpoint):
        src = checkpoint.get("state_dict", checkpoint)
        own = self.state_dict(keep_vars=True)

        # Group aliased keys (resnet_single_branch.* == branches.axt2.*, ...) by
        # tensor, so a checkpoint only needs one of the names for each tensor.
        groups = {}
        for key, tensor in own.items():
            groups.setdefault(id(tensor), (tensor, []))[1].append(key)

        from_cbam_head = self.init_from in ("cbam", "baseline")
        # CBAM/baseline head: its fc.0 (4096 -> 256) becomes img_proj.0. Its
        # fc.3 (256 -> 2) has no counterpart; this model's fc stays at default
        # init, so none of its keys are looked up in the checkpoint.
        remap = (
            {"img_proj.0.weight": "fc.0.weight", "img_proj.0.bias": "fc.0.bias"}
            if from_cbam_head else {}
        )

        problems, consumed, n_loaded, encoder_differs = [], set(), 0, False
        with torch.no_grad():
            for tensor, keys in groups.values():
                if from_cbam_head and any(k.startswith("fc.") for k in keys):
                    continue  # new fusion MLP, default init
                if self.init_from == "baseline" and (
                    any(_is_cbam_key(k) for k in keys)
                    or any(k.startswith("clinical_encoder.") for k in keys)
                ):
                    continue  # identity-init CBAM; encoder from clinical_ckpt
                candidates = [remap.get(k, k) for k in keys]
                present = [c for c in candidates if c in src]
                consumed.update(present)
                if not present:
                    problems.append(f"missing: {candidates[0]}")
                    continue
                src_key = present[0]
                if src[src_key].shape != tensor.shape:
                    problems.append(
                        f"shape mismatch: {src_key} {tuple(src[src_key].shape)} in checkpoint "
                        f"vs {tuple(tensor.shape)} in model"
                    )
                    continue
                if src_key.startswith("clinical_encoder.") and not torch.allclose(
                    tensor, src[src_key].to(tensor)
                ):
                    encoder_differs = True
                tensor.copy_(src[src_key])
                n_loaded += 1

            if from_cbam_head and tuple(src.get("fc.3.weight", torch.empty(0)).shape) != (2, IMG_EMB_DIM):
                # Sanity check that model_ckpt really has a 4096 -> 256 -> 2 head.
                problems.append(
                    f"expected a CBAM/baseline head (fc.0 -> fc.3 of shape (2, {IMG_EMB_DIM})) "
                    "in model_ckpt"
                )

        if encoder_differs and self._clinical_ckpt_given:
            print("[CBAM+late] WARNING: the clinical encoder in model_ckpt differs from "
                  "clinical_ckpt; using model_ckpt's (what the gates were trained with).")

        leftover_cbam = sorted(k for k in src if _is_cbam_key(k) and k not in consumed)
        if leftover_cbam:
            if self.init_from == "baseline":
                problems.append(
                    "model_ckpt contains trained CBAM weights; use init_from: cbam to load "
                    f"them (e.g. {leftover_cbam[0]})"
                )
            else:
                problems.append(
                    f"{len(leftover_cbam)} CBAM tensors in model_ckpt have no place in this "
                    f"model; does the cbam: section match the checkpoint's? (e.g. {leftover_cbam[0]})"
                )

        if problems:
            hint = {
                "cbam": "expects a trained aug11cbam (model: cbam) checkpoint",
                "cbam_late": "expects a checkpoint of this model (model: cbam_late)",
                "baseline": "expects the plain baseline TriSeriesModel checkpoint",
            }[self.init_from]
            raise RuntimeError(
                f"[CBAM+late] model_ckpt doesn't match init_from: {self.init_from} ({hint}):\n  "
                + "\n  ".join(problems[:20])
                + (f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else "")
            )
        self._pretrained_loaded = True
        print(f"[CBAM+late] Loaded {n_loaded} tensors from model_ckpt (init_from={self.init_from})"
              + ("; fusion MLP fc at default init" if from_cbam_head else ""))
        return self

    # ------------------------------------------------------------------ #
    # Freezing and optimization
    # ------------------------------------------------------------------ #
    def _head_modules(self):
        return [self.img_proj, self.fc]

    def _apply_freeze(self, verbose=False):
        # TriSeriesModel.__init__ calls this before the late head exists.
        if not getattr(self, "_late_ready", False):
            return super()._apply_freeze(verbose=False)
        for param in self.parameters():
            param.requires_grad = False
        modules = self._head_modules()
        if self.train_attention:
            modules = modules + self._cbam_modules()
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def on_train_start(self):
        if not self._pretrained_loaded:
            msg = ("no model_ckpt was loaded. Set model_weights.load_weights: true and "
                   "model_weights.model_ckpt.")
            if self.init_from in ("cbam", "cbam_late"):
                raise RuntimeError(f"[CBAM+late] init_from: {self.init_from}, but {msg}")
            print(f"[CBAM+late] WARNING: {msg} The frozen backbone is randomly initialized.")
        self._apply_freeze()

    def configure_optimizers(self):
        # CBAM modules are only in the optimizer when they train this run.
        lr = self.hyperparams["learning_rate"]
        groups = [
            {"name": "head", "lr": lr,
             "params": [p for m in self._head_modules() for p in m.parameters()]},
        ]
        if self.train_attention:
            groups.append(
                {"name": "cbam", "lr": lr * self.attention_lr_scale,
                 "params": [p for m in self._cbam_modules() for p in m.parameters()]}
            )
        optimizer = torch.optim.AdamW(groups, lr, weight_decay=self.hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"],
            threshold=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, data_dict, tabular_features):
        clin = self.clinical_encoder(tabular_features.float())
        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs, clin))
        h = self.img_proj(torch.cat(features, dim=1))

        late_clin = clin
        if self.training and self.late_clinical_dropout > 0:
            keep = torch.rand(clin.size(0), 1, device=clin.device) >= self.late_clinical_dropout
            late_clin = clin * keep.to(clin.dtype)
        return self.fc(torch.cat([h, late_clin], dim=1))

    # ------------------------------------------------------------------ #
    # Gate monitoring
    # ------------------------------------------------------------------ #
    def _register_gate_hooks(self):
        for branch_name, branch in self.branches.items():
            for name, module in branch.named_modules():
                if isinstance(module, ClinicalChannelAttention):
                    kind, fn = "chan", _channel_gate
                elif isinstance(module, ClinicalSpatialAttention):
                    kind, fn = "spat", _spatial_gate
                else:
                    continue
                stage = name.split(".")[0]  # "layer3", or "post_cbam"
                module.register_forward_hook(self._make_gate_hook(f"{branch_name}_{stage}_{kind}", fn))

    def _make_gate_hook(self, tag, gate_fn):
        def hook(module, inputs, output):
            if not self._collect_gates:
                return
            x, clin = inputs
            if x.size(0) < 2:  # spread across patients needs >= 2
                return
            with torch.no_grad():
                w = gate_fn(module, x, clin)
                neutral = 1.0 if module.identity_init else 0.5
                dev = (w - neutral).abs().mean()
                spread = w.std(dim=0, unbiased=False).mean()
            sums = self._gate_sums.setdefault(tag, [0.0, 0.0, 0])
            sums[0] = sums[0] + dev
            sums[1] = sums[1] + spread
            sums[2] += 1
        return hook

    def on_validation_epoch_start(self):
        self._gate_sums = {}
        self._collect_gates = self.log_gate_stats and self.log_configs["log_run"]

    def on_validation_epoch_end(self):
        self._collect_gates = False
        if self.log_configs["log_run"]:
            for tag, (dev, spread, n) in self._gate_sums.items():
                self.log(f"gates/{tag}_dev", dev / n, on_epoch=True)
                self.log(f"gates/{tag}_spread", spread / n, on_epoch=True)
            # Input columns of the fusion MLP's first layer, per modality.
            w = self.fc[0].weight
            self.log("late/clin_weight_norm", w[:, IMG_EMB_DIM:].norm(), on_epoch=True)
            self.log("late/img_weight_norm", w[:, :IMG_EMB_DIM].norm(), on_epoch=True)
        self._gate_sums = {}
        super().on_validation_epoch_end()
