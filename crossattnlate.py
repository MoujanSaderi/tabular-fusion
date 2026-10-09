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
        self.head_lr_scale = cfg.get("head_lr_scale", 0.1)
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
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, 2),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
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
        self.log("grad_norm_delta_out", grad_norm([self.delta_head[-1]]))
        self.log("grad_norm_upstream", grad_norm([self.clin_tok, self.img_tok, self.blocks]))
        self.log("delta_abs_mean", self.last_delta.abs().mean())
        self.log("delta_out_weight_norm", self.delta_head[-1].weight.norm())
        d = self.last_delta[:, 1] - self.last_delta[:, 0]   # correction to the positive-class log-odds
        self.log("delta_logodds_std", d.std())              # patient-specific part (can change ranking)
        self.log("delta_logodds_mean", d.mean())            # uniform part (can't change AUC)

    def configure_optimizers(self):
        lr = self.hyperparams["learning_rate"]
        groups = [{"params": [p for m in self._side_modules() for p in m.parameters()],
                   "lr": lr}]
        if self.train_head:
            groups.append({"params": [p for m in self._head_modules() for p in m.parameters()],
                           "lr": lr * self.head_lr_scale})
        optimizer = torch.optim.AdamW(groups, lr, weight_decay=self.hyperparams["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=int(self.hyperparams["lr_patience"]),
            factor=self.hyperparams["factor"], threshold=1e-4,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

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
