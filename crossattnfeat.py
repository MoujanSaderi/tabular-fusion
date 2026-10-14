"""
Cross-attention with the CLS output as HEAD FEATURES (no late clinical path).

Same backbone, tokenizers and attention blocks as crossattnfusion.py, but the
CLS token feeds the image head's hidden layer instead of adding a correction
(delta_head) to its logits:

    crossattnfusion:  logits = fc(pooled) + delta_head(CLS)
    this model:       hidden = ReLU( fc.0(pooled) + cls_in(cls_proj(CLS)) )   fc.0: 4096 -> 256
                      logits = fc.3(dropout(hidden))

so the head can combine the pooled image features and the attention features
non-linearly. Clinical information reaches the prediction only through the
CLS token: directly with residual_query: true, or only through WHERE the
clinical queries attend with residual_query: false.

Main use: phase 1 for crossattn_late_feat (crossattnlatefeat.py), which loads
this model's clin_tok, img_tok, blocks and cls_proj via xattn.side_ckpt.
Compared with pretraining via crossattnfusion's delta_head (a Linear(dim -> 2)
readout, where only the logit difference matters, so the CLS features are only
trained along one direction), cls_in feeds 256 hidden units, so the CLS
representation is shaped along many directions, in nearly the role it plays in
phase 2. cls_in itself does not transfer (different head, different hidden
units).

Initialization
--------------
model_weights.model_ckpt   the plain baseline (no clinical fusion) checkpoint:
                           backbone and image head fc load by name, strictly.
                           cls_in starts with small random weights (std
                           xattn.cls_init_std), so step 0 is very close to the
                           image-only baseline and the attention gets gradient
                           from the first step.

Config, under the `xattn:` section (all optional; unknown keys are an error)
    dim, heads, layers, dropout, tap, kv_pool, feature_mask_prob, n_features
                    as in crossattnfusion.py
    residual_query  default FALSE (unlike crossattnfusion), since this model
                    is mainly pretraining for crossattn_late_feat, where the
                    CLS should carry clinically selected image content
                                                                      (default false)
    cls_init_std    std of cls_in's small random init; must be > 0   (default 1e-3)
    train_head      also fine-tune the image head fc (both layers)   (default false)
    head_lr_scale   head LR = learning_rate * head_lr_scale, if train_head
                                                                      (default 0.1)
    side_lr         LR for the attention branch and cls_in (default: hyperparameters.learning_rate)
    lr_monitor      "val_loss" (mode min, as crossattnfusion) or "val_auc" (mode max)
                                                                      (default "val_loss")

Monitoring: val_auc_no_cls, cls_auc_gain, cls/feat_spread, cls/in_weight_norm
(see clsfeatures.py). Here val_auc_no_cls is the image head alone (with its
current weights), so cls_auc_gain is what the attention adds to the images.

Run with: python train.py --config configs/crossattn_feat_pretrain.yaml
"""
import torch

from clsfeatures import (
    ClsFeatureMonitorMixin, build_cls_in, build_cls_proj, check_head_layout, head_with_cls,
)
from crossattnfusion import CrossAttnFusionModel, TAP_CHANNELS, branch_forward

XATTN_DEFAULTS = {
    "dim": 128,
    "heads": 4,
    "layers": 1,
    "dropout": 0.1,
    "tap": "layer4",
    "kv_pool": 1,
    "residual_query": False,
    "feature_mask_prob": 0.0,
    "n_features": None,
    "cls_init_std": 1e-3,
    "train_head": False,
    "head_lr_scale": 0.1,
    "side_lr": None,
    "lr_monitor": "val_loss",
}


class CrossAttnFeatFusion(ClsFeatureMonitorMixin, CrossAttnFusionModel):
    SIDE_MODULES = ("clin_tok", "img_tok", "blocks", "cls_proj", "cls_in")

    def __init__(self, config):
        user_cfg = config.get("xattn") or {}
        unknown = set(user_cfg) - set(XATTN_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown xattn keys for crossattn_feat: {sorted(unknown)} "
                "(use train_head instead of train_fc; delta_head options don't apply)"
            )
        cfg = {**XATTN_DEFAULTS, **user_cfg}
        if cfg["tap"] not in TAP_CHANNELS:
            raise ValueError(f"xattn.tap must be one of {list(TAP_CHANNELS)}, got {cfg['tap']!r}")
        if cfg["lr_monitor"] not in ("val_auc", "val_loss"):
            raise ValueError(f"xattn.lr_monitor must be val_auc or val_loss, got {cfg['lr_monitor']}")
        # Rebind `config` (Lightning's save_hyperparameters reads it from this
        # frame) and pass the filled-in section to the parent, so its defaults
        # (e.g. residual_query: true) don't apply and the checkpoint's saved
        # hyperparameters record every setting crossattn_late_feat checks.
        config = {**config, "xattn": cfg}
        super().__init__(config)
        self.xattn_cfg = cfg

        del self.delta_head  # replaced by cls_proj + cls_in
        check_head_layout(self.fc, "CrossAttnFeat")
        dim = int(cfg["dim"])
        self.cls_proj = build_cls_proj(dim, float(cfg["dropout"]))
        self.cls_in = build_cls_in(dim, self.fc[0].out_features, cfg["cls_init_std"])

        self.train_head = bool(cfg["train_head"])
        self.head_lr_scale = float(cfg["head_lr_scale"])
        self.side_lr = cfg["side_lr"]
        self.lr_monitor = cfg["lr_monitor"]
        self._pretrained_loaded = False
        self._last_no_cls = None
        self._last_cls_spread = None
        self._reset_cls_monitor()

        n_side = sum(p.numel() for n in self.SIDE_MODULES for p in getattr(self, n).parameters())
        print(f"[CrossAttnFeat] tap={self.tap}, dim={dim}, layers={cfg['layers']}, "
              f"residual_query={cfg['residual_query']}, kv_pool={cfg['kv_pool']}, "
              f"train_head={self.train_head}; head fc.0 reads [{self.fc[0].in_features} + "
              f"{dim} (CLS)] -> {self.fc[0].out_features}; side params={n_side:,}")

    # ------------------------------------------------------------------ #
    # Initialization (called by train.build_model)
    # ------------------------------------------------------------------ #
    def load_pretrained_checkpoint(self, checkpoint):
        """Load the backbone and image head strictly from the baseline checkpoint."""
        src = checkpoint.get("state_dict", checkpoint)
        groups = {}
        for key, tensor in self.state_dict(keep_vars=True).items():
            groups.setdefault(id(tensor), (tensor, []))[1].append(key)  # aliases share tensors
        problems, n_loaded = [], 0
        with torch.no_grad():
            for tensor, keys in groups.values():
                if any(k.startswith(self.SIDE_MODULES) for k in keys):
                    continue
                src_key = next((k for k in keys if k in src), None)
                if src_key is None:
                    problems.append(f"missing: {keys[0]}")
                elif src[src_key].shape != tensor.shape:
                    problems.append(f"shape mismatch: {src_key} {tuple(src[src_key].shape)} "
                                    f"in checkpoint vs {tuple(tensor.shape)} in model")
                else:
                    tensor.copy_(src[src_key])
                    n_loaded += 1
        if problems:
            raise RuntimeError(
                "[CrossAttnFeat] model_ckpt must be the plain baseline checkpoint "
                "(branches + fc):\n  " + "\n  ".join(problems[:20])
            )
        print(f"[CrossAttnFeat] Loaded {n_loaded} baseline tensors from model_ckpt")
        self._pretrained_loaded = True
        return self

    # ------------------------------------------------------------------ #
    # Freezing and optimization
    # ------------------------------------------------------------------ #
    def _side_modules(self):
        return [getattr(self, n) for n in self.SIDE_MODULES]

    def _fusion_modules(self):
        return self._side_modules() + ([self.fc] if self.train_head else [])

    def on_train_start(self):
        if not self._pretrained_loaded:
            raise RuntimeError(
                "[CrossAttnFeat] no baseline checkpoint loaded. Set "
                "model_weights.load_weights: true and model_weights.model_ckpt."
            )
        super().on_train_start()  # freezes everything except _fusion_modules()

    def configure_optimizers(self):
        lr = self.hyperparams["learning_rate"]
        side_lr = float(self.side_lr) if self.side_lr is not None else lr
        groups = [{"params": [p for m in self._side_modules() for p in m.parameters()],
                   "lr": side_lr, "name": "side"}]
        if self.train_head:
            groups.append({"params": list(self.fc.parameters()),
                           "lr": lr * self.head_lr_scale, "name": "head"})
        optimizer = torch.optim.AdamW(groups, side_lr, weight_decay=self.hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max" if self.lr_monitor == "val_auc" else "min",
            patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"], threshold=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.lr_monitor}

    # ------------------------------------------------------------------ #
    # Forward
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
        kv = torch.cat(kv, dim=1)

        mask = None
        if self.training and self.feature_mask_prob > 0:
            mask = torch.rand_like(tab) < self.feature_mask_prob
        q = self.clin_tok(tab, mask)
        attns = []
        for block in self.blocks:
            q, attn = block(q, kv)
            attns.append(attn.detach())
        self.last_attn, self.last_grids = attns, grids
        cls_feat = self.cls_proj(q[:, 0])

        base_pre = self.fc[0](torch.cat(pooled, dim=1))
        self.cls_eval_stats(self.fc, base_pre, cls_feat)  # image head alone, for val_auc_no_cls
        return head_with_cls(self.fc, base_pre, self.cls_in(cls_feat))

    # training/validation/test steps are inherited from CrossAttnFusionModel
    # (validation wrapped by ClsFeatureMonitorMixin)
