"""
Late fusion + cross-attention correction.

The base prediction is your trained late-fusion model (aug11latefusionflat.
LateFusionFlatFrozen): concat[img_proj(pooled image), clinical encoder] -> fc.
A cross-attention side branch (clinical tokens querying image tokens, see
crossattnfusion.py) adds a correction to those logits:

    branch:  conv1 .. layer2 ──(tap)──► image tokens ─┐
                       │                               ├─ cross-attn ─► CLS ─► delta_head ─┐
                       ▼                clinical tokens┘                                   │
                 layer3, layer4 ─► pool ─► img_proj ─┐                                     │
                                  clinical_encoder ──┴─ concat ─► fc ─► base logits ──(+)──┴─► logits

Everything in the late-fusion model is loaded from its checkpoint
(model_weights.model_ckpt) and frozen; only the side branch trains. delta_head's
last layer is zero-initialized, so at step 0 the model reproduces the late-fusion
model exactly, and the correction measures what clinically guided attention adds
BEYOND late fusion.

Config: the same `xattn:` section as crossattnfusion.py, plus
    train_head      also fine-tune img_proj + fc                       (default false)
    head_lr_scale   LR multiplier for the head if train_head            (default 0.1)
    side_lr         learning rate for the side branch       (default: hyperparameters.learning_rate)
    center_delta    remove the uniform part of the correction, so it can only
                    change how patients are ranked, not shift everyone
                    together (see LogOddsCorrection)                    (default true)
    delta_init_std  init std of the correction's output layer; 0 = exact
                    zero-init. A small non-zero value shortens the "dead" phase
                    where upstream layers get almost no gradient       (default 0.0)
    lr_monitor      metric ReduceLROnPlateau watches: "val_auc" (mode max) or
                    "val_loss" (mode min)                               (default "val_auc")
residual_query defaults to FALSE here: the late-fusion head already sees the
clinical features, so the correction should carry image content selected by
the clinical queries, not a second copy of the clinical signal.

Run with: python train.py --config configs/crossattn_late.yaml
"""
import torch
import torch.nn as nn

from aug11latefusionflat import LateFusionFlatFrozen
from crossattnfusion import (
    TAP_CHANNELS, ClinicalTokenizer, FusionBlock, ImageTokenizer, branch_forward,
)


class LogOddsCorrection(nn.Module):
    """Maps the CLS token to a correction of the positive-class log-odds.

    Softmax only depends on the difference between the two logits, so the
    correction is a single scalar d per patient, returned as (-d/2, +d/2).
    With center=True, the part of d shared by every patient is removed: in
    training by subtracting the batch mean (so the uniform part gets no
    gradient at all), in eval by subtracting a running mean of the batch means
    (so predictions don't depend on what else is in the batch). A uniform
    shift can't change AUC, and in the overfit test it was what the branch
    learned first, delaying the patient-specific part.
    """

    def __init__(self, dim, dropout, center=True, init_std=0.0, momentum=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.out = nn.Linear(dim, 1, bias=not center)
        if init_std > 0:
            nn.init.normal_(self.out.weight, std=init_std)
        else:
            nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)
        self.center = center
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(1))

    def forward(self, cls):
        d = self.out(self.mlp(cls)).squeeze(-1)  # (B,)
        if self.center:
            if self.training and d.numel() > 1:
                batch_mean = d.mean()
                with torch.no_grad():
                    self.running_mean.lerp_(batch_mean.detach(), self.momentum)
                d = d - batch_mean
            else:
                d = d - self.running_mean
        return torch.stack([-d / 2, d / 2], dim=1)  # (B, 2)


class CrossAttnLateFusion(LateFusionFlatFrozen):
    SIDE_MODULES = ("clin_tok", "img_tok", "blocks", "delta_head")

    def __init__(self, config):
        super().__init__(config)  # branches, img_proj, clinical_encoder, fc
        cfg = config.get("xattn", {}) or {}
        dim = cfg.get("dim", 128)
        heads = cfg.get("heads", 4)
        dropout = cfg.get("dropout", 0.1)
        self.tap = cfg.get("tap", "layer4")
        self.feature_mask_prob = cfg.get("feature_mask_prob", 0.0)
        self.train_head = cfg.get("train_head", False)
        self.head_lr_scale = float(cfg.get("head_lr_scale", 0.1))
        self.side_lr = cfg.get("side_lr")
        self.lr_monitor = cfg.get("lr_monitor", "val_auc")
        if self.lr_monitor not in ("val_auc", "val_loss"):
            raise ValueError(f"xattn.lr_monitor must be val_auc or val_loss, got {self.lr_monitor}")
        self._pretrained_loaded = False

        n_features = cfg.get("n_features") or config["data"].get("tabular_dims") or 37
        self.clin_tok = ClinicalTokenizer(n_features, dim)
        self.img_tok = nn.ModuleDict({
            name: ImageTokenizer(TAP_CHANNELS[self.tap], dim, cfg.get("kv_pool", 1))
            for name, _ in self.branch_specs
        })
        rq = cfg.get("residual_query", False)
        self.blocks = nn.ModuleList([
            FusionBlock(dim, heads, dropout, residual_query=(rq or i > 0))
            for i in range(cfg.get("layers", 1))
        ])
        self.delta_head = LogOddsCorrection(
            dim, dropout,
            center=cfg.get("center_delta", True),
            # float(): PyYAML reads exponent notation without a decimal point
            # (e.g. 1e-3) as a string, not a number
            init_std=float(cfg.get("delta_init_std", 0.0)),
        )
        self.last_attn = None
        self.last_grids = None

    # ------------------------------------------------------------------ #
    def load_pretrained_checkpoint(self, checkpoint):
        """Load every late-fusion weight; fail if anything outside the side
        branch is missing or mis-shaped (train.py's default loader would skip
        it silently and leave it randomly initialized)."""
        src = checkpoint.get("state_dict", checkpoint)
        own = self.state_dict()
        mismatched = [k for k in src if k in own and src[k].shape != own[k].shape]
        if mismatched:
            raise RuntimeError(f"[crossattn_late] shape mismatch: {mismatched[:5]}")
        result = self.load_state_dict(src, strict=False)
        missing = [k for k in result.missing_keys
                   if not k.startswith(self.SIDE_MODULES)]
        if missing:
            raise RuntimeError(
                "[crossattn_late] model_ckpt is missing late-fusion weights "
                f"(is it a latefusion_flat checkpoint?): {missing[:5]}"
            )
        if result.unexpected_keys:
            print(f"[crossattn_late] ignoring {len(result.unexpected_keys)} "
                  f"unexpected checkpoint keys, e.g. {result.unexpected_keys[:3]}")
        self._pretrained_loaded = True
        return self

    def _side_modules(self):
        return [getattr(self, n) for n in self.SIDE_MODULES]

    def _head_modules(self):
        return [self.img_proj, self.fc]

    def on_train_start(self):
        if not self._pretrained_loaded:
            raise RuntimeError(
                "[crossattn_late] no late-fusion checkpoint loaded. Set "
                "model_weights.load_weights: true and model_weights.model_ckpt."
            )
        for p in self.parameters():
            p.requires_grad = False
        trainable = self._side_modules() + (self._head_modules() if self.train_head else [])
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
        self.branches.eval()
    
    def on_before_optimizer_step(self, optimizer):
        if not self.log_configs["log_run"]:
            return
        def grad_norm(modules):
            grads = [p.grad.norm() for m in modules for p in m.parameters() if p.grad is not None]
            return torch.norm(torch.stack(grads)) if grads else torch.tensor(0.0)
        self.log("grad_norm_delta_out", grad_norm([self.delta_head.out]))
        self.log("grad_norm_upstream", grad_norm([self.clin_tok, self.img_tok, self.blocks]))
        self.log("delta_abs_mean", self.last_delta.abs().mean())
        self.log("delta_out_weight_norm", self.delta_head.out.weight.norm())
        d = self.last_delta[:, 1] - self.last_delta[:, 0]   # correction to the positive-class log-odds
        self.log("delta_logodds_std", d.std())              # patient-specific part (can change ranking)
        self.log("delta_logodds_mean", d.mean())            # uniform part (can't change AUC)
        self.log("delta_running_mean", self.delta_head.running_mean.squeeze())  # 0 unless center_delta is off

    def configure_optimizers(self):
        lr = self.hyperparams["learning_rate"]
        side_lr = float(self.side_lr) if self.side_lr is not None else lr
        groups = [{"params": [p for m in self._side_modules() for p in m.parameters()],
                   "lr": side_lr, "name": "side"}]
        if self.train_head:
            groups.append({"params": [p for m in self._head_modules() for p in m.parameters()],
                           "lr": lr * self.head_lr_scale, "name": "head"})
        optimizer = torch.optim.AdamW(groups, side_lr, weight_decay=self.hyperparams["weight_decay"])
        # val_auc: only cut the LR when ranking stops improving. val_loss also
        # improves from recalibration alone, which made the scheduler cut the
        # LR before the patient-specific correction had a chance to grow.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max" if self.lr_monitor == "val_auc" else "min",
            patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"], threshold=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.lr_monitor}

    # ------------------------------------------------------------------ #
    def forward(self, data_dict, tabular_features):
        tab = tabular_features.float()
        pooled, kv, grids = [], [], []
        for name, keys in self.branch_specs:
            inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            with torch.no_grad():
                fmap, vec = branch_forward(self.branches[name], inputs, self.tap)
            pooled.append(vec)
            tokens, grid = self.img_tok[name](fmap)
            kv.append(tokens)
            grids.append((name, tuple(grid)))

        # base: the late-fusion model, unchanged
        img_emb = self.img_proj(torch.cat(pooled, dim=1))
        clin_emb = self.clinical_encoder(tab)
        base_logits = self.fc(torch.cat([img_emb, clin_emb], dim=1))

        # side branch: clinical queries attend over image tokens
        mask = None
        if self.training and self.feature_mask_prob > 0:
            mask = torch.rand_like(tab) < self.feature_mask_prob
        q = self.clin_tok(tab, mask)
        kv = torch.cat(kv, dim=1)
        attns = []
        for block in self.blocks:
            q, attn = block(q, kv)
            attns.append(attn.detach())
        self.last_attn, self.last_grids = attns, grids

        delta = self.delta_head(q[:, 0])
        self.last_delta = delta.detach()
        return base_logits + delta
    # training/validation/test steps are inherited from LateFusionFlatFrozen
