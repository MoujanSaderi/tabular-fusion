"""
Late fusion + cross-attention FEATURES.

Same pieces as crossattnlate.py, but the cross-attention output is an input to
the late-fusion head instead of a separate correction added to its logits:

    branch:  conv1 .. layer4 ──(tap)──► image tokens ─┐
                       │                               ├─ cross-attn ─► CLS ─► cls_proj ─► cls_feat ─┐
                       │                clinical tokens┘                                             │
                       ▼                                                                             ▼
                 pool ─► img_proj ─► img_emb ─┐                                                  cls_in
                 clinical_encoder ─► clin_emb ┴─► fc.0 ──────────────────────────────────(+)──► ReLU ─► dropout ─► fc.3 ─► logits

i.e. the late-fusion head's first layer now reads [img_emb ; clin_emb ; cls_feat]:

    hidden = ReLU( fc.0([img_emb ; clin_emb]) + cls_in(cls_feat) )
    logits = fc.3(dropout(hidden))

`cls_in` is the CLS block of fc.0's weight matrix, kept as its own module
(Linear(dim -> 128), no bias) so it can have its own LR and the late-fusion
weights keep their checkpoint names. It is equivalent to widening fc.0 from
320 to 320 + dim inputs.

Initialization
--------------
model_weights.model_ckpt   a trained LateFusionFlatFrozen checkpoint (required):
                           backbone, img_proj, clinical encoder and fc load by
                           name, strictly. cls_in starts with small random
                           weights (std xattn.cls_init_std), so at step 0 the
                           outputs are very close to, but not exactly, the
                           late-fusion model's; the attention blocks get
                           gradient from the first step.
xattn.side_ckpt            optional: a phase-1 checkpoint of either
                             crossattn_feat (crossattnfeat.py; recommended, see
                               configs/crossattn_feat_pretrain.yaml): its
                               clin_tok, img_tok, blocks and cls_proj are copied
                               over; its cls_in is not (it maps into the
                               baseline head's hidden units, not this head's).
                             crossattn (crossattnfusion.py; see
                               configs/crossattn_pretrain.yaml): clin_tok,
                               img_tok and blocks are copied, and its
                               delta_head's hidden layers (LayerNorm, Linear)
                               become cls_proj; its delta_head readout is NOT
                               used (it was trained to correct an image-only
                               model).
                           The
                           attention settings (dim, heads, layers, tap, kv_pool,
                           residual_query) must match the phase-1 run; this is
                           checked against the hyperparameters saved in the
                           checkpoint. null = side branch trains from scratch.

Config, under the `xattn:` section (all optional; unknown keys are an error)
    dim, heads, layers, dropout, tap, kv_pool, feature_mask_prob, n_features
                    as in crossattnfusion.py
    residual_query  default FALSE: the head already sees the clinical
                    embedding, so the CLS should carry image content selected
                    by the clinical queries, not a second copy of the
                    clinical signal                                   (default false)
    side_ckpt       see above                                        (default null)
    cls_init_std    std of cls_in's small random init; must be > 0
                                                                      (default 1e-3)
    train_head      also fine-tune img_proj + fc (the late-fusion head)
                                                                      (default false)
    head_lr_scale   head LR = learning_rate * head_lr_scale, if train_head
                                                                      (default 0.1)
    side_lr         LR for the side branch and cls_in   (default: hyperparameters.learning_rate)
    lr_monitor      "val_auc" (mode max) or "val_loss" (mode min)    (default "val_auc")

Monitoring (every validation epoch)
-----------------------------------
    val_auc_no_cls   val AUC of the same model with cls_feat zeroed, i.e. the
                     current head without the attention features
    cls_auc_gain     val_auc - val_auc_no_cls: how much the head uses the
                     attention features. ~0 means it ignores them.
    cls/in_weight_norm, cls/feat_spread (std of cls_feat across patients,
    averaged over dims; ~0 means the CLS is the same for everyone)
Both AUCs here use softmax probabilities (sklearn roc_auc_score).

Run with: python train.py --config configs/crossattnlate_feat.yaml
"""
import os

import torch
import torch.nn as nn

from aug11latefusionflat import LateFusionFlatFrozen
from clsfeatures import (
    ClsFeatureMonitorMixin, build_cls_in, build_cls_proj, check_head_layout, head_with_cls,
)
from crossattnfusion import (
    TAP_CHANNELS, ClinicalTokenizer, FusionBlock, ImageTokenizer, branch_forward,
)

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
    "side_ckpt": None,
    "cls_init_std": 1e-3,
    "train_head": False,
    "head_lr_scale": 0.1,
    "side_lr": None,
    "lr_monitor": "val_auc",
}
# Attention settings that must match a side_ckpt, with each phase-1 model's
# defaults (used when its saved config left a key out).
_MATCH_COMMON = {"dim": 128, "heads": 4, "layers": 1, "tap": "layer4", "kv_pool": 1}
SIDE_MATCH_DEFAULTS = {
    "crossattn": {**_MATCH_COMMON, "residual_query": True},       # crossattnfusion.py
    "crossattn_feat": {**_MATCH_COMMON, "residual_query": False},  # crossattnfeat.py
}
# Where this model's cls_proj comes from in each phase-1 checkpoint type.
CLS_PROJ_SOURCE = {"crossattn": "delta_head.", "crossattn_feat": "cls_proj."}


class CrossAttnLateFeatFusion(ClsFeatureMonitorMixin, LateFusionFlatFrozen):
    SIDE_MODULES = ("clin_tok", "img_tok", "blocks", "cls_proj", "cls_in")
    # Side modules a side_ckpt provides (cls_in is always new).
    TRANSFERRED = ("clin_tok", "img_tok", "blocks", "cls_proj")

    def __init__(self, config):
        super().__init__(config)  # branches, img_proj, clinical_encoder, fc
        user_cfg = config.get("xattn") or {}
        unknown = set(user_cfg) - set(XATTN_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unknown xattn keys for crossattn_late_feat: {sorted(unknown)} "
                "(delta_head options such as center_delta / delta_init_std / "
                "train_fc don't apply to this model)"
            )
        cfg = {**XATTN_DEFAULTS, **user_cfg}
        self.xattn_cfg = cfg
        dim = int(cfg["dim"])
        self.tap = cfg["tap"]
        if self.tap not in TAP_CHANNELS:
            raise ValueError(f"xattn.tap must be one of {list(TAP_CHANNELS)}, got {self.tap!r}")
        self.feature_mask_prob = float(cfg["feature_mask_prob"])
        self.train_head = bool(cfg["train_head"])
        self.head_lr_scale = float(cfg["head_lr_scale"])
        self.side_lr = cfg["side_lr"]
        self.side_ckpt = cfg["side_ckpt"]
        self.lr_monitor = cfg["lr_monitor"]
        if self.lr_monitor not in ("val_auc", "val_loss"):
            raise ValueError(f"xattn.lr_monitor must be val_auc or val_loss, got {self.lr_monitor}")

        # The head this model extends: Linear -> ReLU -> Dropout -> Linear.
        fc = self.fc
        check_head_layout(fc, "CrossAttnLateFeat")
        hidden = fc[0].out_features

        n_features = cfg["n_features"] or config["data"].get("tabular_dims") or 37
        self.clin_tok = ClinicalTokenizer(n_features, dim)
        self.img_tok = nn.ModuleDict({
            name: ImageTokenizer(TAP_CHANNELS[self.tap], dim, int(cfg["kv_pool"]))
            for name, _ in self.branch_specs
        })
        rq = bool(cfg["residual_query"])
        self.blocks = nn.ModuleList([
            FusionBlock(dim, int(cfg["heads"]), float(cfg["dropout"]), residual_query=(rq or i > 0))
            for i in range(int(cfg["layers"]))
        ])
        self.cls_proj = build_cls_proj(dim, float(cfg["dropout"]))
        self.cls_in = build_cls_in(dim, hidden, cfg["cls_init_std"])

        self._pretrained_loaded = False
        self._reset_cls_monitor()
        self._last_no_cls = None
        self._last_cls_spread = None
        self.last_attn = None
        self.last_grids = None

        n_side = sum(p.numel() for n in self.SIDE_MODULES for p in getattr(self, n).parameters())
        print(f"[CrossAttnLateFeat] tap={self.tap}, dim={dim}, layers={cfg['layers']}, "
              f"residual_query={rq}, kv_pool={cfg['kv_pool']}, side_ckpt={self.side_ckpt}, "
              f"train_head={self.train_head}; head fc.0 reads [{fc[0].in_features} + {dim} (CLS)] "
              f"-> {hidden}; side params={n_side:,}")

    # ------------------------------------------------------------------ #
    # Initialization (called by train.build_model)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unique_keys(module_state):
        """Group aliased keys (resnet_single_branch.* == branches.axt2.*) by tensor."""
        groups = {}
        for key, tensor in module_state.items():
            groups.setdefault(id(tensor), (tensor, []))[1].append(key)
        return groups.values()

    def load_pretrained_checkpoint(self, checkpoint):
        """Load every late-fusion weight strictly from model_ckpt, then (if
        xattn.side_ckpt is set) the pretrained side branch."""
        src = checkpoint.get("state_dict", checkpoint)
        problems, n_loaded = [], 0
        with torch.no_grad():
            for tensor, keys in self._unique_keys(self.state_dict(keep_vars=True)):
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
                "[CrossAttnLateFeat] model_ckpt must be a LateFusionFlatFrozen (latefusion_flat) "
                "checkpoint:\n  " + "\n  ".join(problems[:20])
            )
        print(f"[CrossAttnLateFeat] Loaded {n_loaded} late-fusion tensors from model_ckpt")

        if self.side_ckpt:
            self._load_side_checkpoint(self.side_ckpt)
        self._pretrained_loaded = True
        return self

    def _load_side_checkpoint(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[CrossAttnLateFeat] xattn.side_ckpt not found: {path!r}. Point it at the best "
                "phase-1 checkpoint (crossattn_feat or crossattn), or set it to null to train the "
                "side branch from scratch."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        src = ckpt.get("state_dict", ckpt)

        # Attention settings that leave no trace in the weights' shapes
        # (residual_query, kv_pool) can only be checked via saved hparams.
        saved = ((ckpt.get("hyper_parameters") or {}).get("config") or {})
        saved_x = saved.get("xattn")
        model_name = (saved.get("model") or {}).get("model")
        if model_name is None:  # no saved config: tell the type apart by its keys
            model_name = "crossattn_feat" if any(k.startswith("cls_proj.") for k in src) else "crossattn"
        if model_name not in SIDE_MATCH_DEFAULTS:
            raise RuntimeError(f"[CrossAttnLateFeat] side_ckpt is a '{model_name}' checkpoint; "
                               f"expected one of {sorted(SIDE_MATCH_DEFAULTS)}.")
        if saved_x is None:
            print(f"[CrossAttnLateFeat] WARNING: {path} has no saved xattn config; can't check "
                  "that residual_query / kv_pool / tap match the phase-1 run.")
        else:
            mismatched = []
            for key, default in SIDE_MATCH_DEFAULTS[model_name].items():
                theirs, ours = saved_x.get(key, default), self.xattn_cfg[key]
                if theirs != ours:
                    mismatched.append(f"{key}: side_ckpt {theirs!r} vs this config {ours!r}")
            if mismatched:
                raise RuntimeError("[CrossAttnLateFeat] xattn settings differ from the side_ckpt "
                                   "run:\n  " + "\n  ".join(mismatched))

        remap = {"cls_proj.": CLS_PROJ_SOURCE[model_name]}
        problems, n_loaded = [], 0
        with torch.no_grad():
            for name in self.TRANSFERRED:
                for key, tensor in getattr(self, name).state_dict(prefix=f"{name}.",
                                                                  keep_vars=True).items():
                    src_key = key
                    for old, new in remap.items():
                        if key.startswith(old):
                            src_key = new + key[len(old):]
                    if src_key not in src:
                        problems.append(f"missing: {src_key}")
                    elif src[src_key].shape != tensor.shape:
                        problems.append(f"shape mismatch: {src_key} {tuple(src[src_key].shape)} "
                                        f"in side_ckpt vs {tuple(tensor.shape)} in model")
                    else:
                        tensor.copy_(src[src_key])
                        n_loaded += 1
        if problems:
            raise RuntimeError(
                "[CrossAttnLateFeat] side_ckpt must be a crossattn checkpoint with the same "
                "xattn settings and tabular features:\n  " + "\n  ".join(problems[:20])
            )
        print(f"[CrossAttnLateFeat] Loaded {n_loaded} side-branch tensors from {path} "
              f"({model_name} checkpoint; cls_in starts at small random values, "
              f"std={self.xattn_cfg['cls_init_std']})")

    # ------------------------------------------------------------------ #
    # Freezing and optimization
    # ------------------------------------------------------------------ #
    def _side_modules(self):
        return [getattr(self, n) for n in self.SIDE_MODULES]

    def _head_modules(self):
        return [self.img_proj, self.fc]

    def on_train_start(self):
        if not self._pretrained_loaded:
            raise RuntimeError(
                "[CrossAttnLateFeat] no late-fusion checkpoint loaded. Set "
                "model_weights.load_weights: true and model_weights.model_ckpt."
            )
        for p in self.parameters():
            p.requires_grad = False
        trainable = self._side_modules() + (self._head_modules() if self.train_head else [])
        for m in trainable:
            for p in m.parameters():
                p.requires_grad = True
        self.branches.eval()

    def configure_optimizers(self):
        lr = self.hyperparams["learning_rate"]
        side_lr = float(self.side_lr) if self.side_lr is not None else lr
        groups = [{"params": [p for m in self._side_modules() for p in m.parameters()],
                   "lr": side_lr, "name": "side"}]
        if self.train_head:
            groups.append({"params": [p for m in self._head_modules() for p in m.parameters()],
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

        img_emb = self.img_proj(torch.cat(pooled, dim=1))
        clin_emb = self.clinical_encoder(tab)

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
        cls_feat = self.cls_proj(q[:, 0])

        base_pre = self.fc[0](torch.cat([img_emb, clin_emb], dim=1))
        self.cls_eval_stats(self.fc, base_pre, cls_feat)  # no-CLS logits, for val_auc_no_cls
        return head_with_cls(self.fc, base_pre, self.cls_in(cls_feat))

    # training/test steps are inherited from LateFusionFlatFrozen
