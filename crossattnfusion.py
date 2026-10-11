"""
Cross-attention fusion of 3D MRI (frozen ResNet3D) + clinical features.

Clinical features are tokenized one-token-per-feature (FT-Transformer style) plus a
[CLS] token. These tokens act as QUERIES that attend over the spatial layer4
feature maps of both branches (T2 and stacked ADC+B1500), which act as KEYS/VALUES.
So the attention weights are a dot product between clinical context and *image
content at each location*: unlike CBAM / early-scalar / DAFT, where the clinical
signal produces a gate that does not depend on what the image shows where.

    layer4 map (B, 2048, D, H, W) --1x1x1 conv--> (B, d, D, H, W)
        + depthwise 3x3x3 conv (positional encoding) + branch embedding
        -> optional avg-pool -> flatten -> image tokens (B, N, d)
    clinical x (B, 37) -> [CLS ; 37 feature tokens] (B, 38, d)
    blocks: cross-attn(clinical -> image), self-attn(clinical), FFN
    readout: CLS -> delta_head -> (B, 2)

    logits = fc(avgpooled image features)  +  delta_head(CLS)
             ^ baseline head, loaded from     ^ last layer zero-initialized,
               the baseline checkpoint          so step 0 == baseline exactly

Trainable: image projections, clinical tokenizer, attention blocks, delta_head
(optionally fc). Backbone frozen, BatchNorm stats frozen.

Config (all optional), under a top-level `xattn:` section:
    dim                  token width                                (default 128)
    heads                                                           (default 4)
    layers               number of fusion blocks                    (default 1)
    dropout                                                         (default 0.1)
    tap                  "layer4" | "layer2": which feature map becomes the
                         image tokens (side branch; backbone unchanged) (default "layer4")
    kv_pool              avg-pool factor on (H, W) before tokenizing (default 1)
    residual_query       true: clinical tokens keep their own content
                         false: first block's output is ONLY attended image
                         content, so clinical info can enter solely through
                         *where* it looks (a cleaner test vs late fusion)  (default true)
    feature_mask_prob    per-feature prob. of replacing a clinical token with a
                         learned [MASK] token during training           (default 0.0)
    train_fc             also fine-tune the baseline fc                  (default false)
    n_features           defaults to data.tabular_dims, which train.py fills
                         from the tabular CSV

Run with: python train.py --config configs/crossattn.yaml
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ResNet3D.base_3Dresnet import Base3DResNet, Bottleneck, ResNetBranch


TAP_CHANNELS = {"layer2": 512, "layer4": 2048}


def branch_forward(branch, x, tap="layer4"):
    """Run a ResNetBranch once; return (tapped feature map, pooled layer4 vector).

    The tapped map is only read by the side branch, never modified, so the
    pooled vector is exactly what the plain ResNetBranch.forward returns.
    """
    out = branch.relu(branch.bn1(branch.conv1(x)))
    out = branch.maxpool(out)
    out = branch.layer1(out)
    out = branch.layer2(out)
    tapped = out
    out = branch.layer3(out)
    out = branch.layer4(out)
    if tap == "layer4":
        tapped = out
    return tapped, out.mean(dim=(2, 3, 4))


def series_keys(config):
    # train.py passes config["data"]["series"] through as plain strings
    return [s if isinstance(s, str) else s.value["key"] for s in config["data"]["series"]]


class ClinicalTokenizer(nn.Module):
    """token_f = x_f * W_f + b_f, plus a [CLS] token and per-feature [MASK] tokens."""

    def __init__(self, n_features, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        # bias doubles as a feature-identity embedding, so x_f == 0 still tells
        # the model *which* feature the token is
        self.bias = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(n_features, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, x, mask=None):
        # x: (B, F); mask: optional bool (B, F), True = treat feature as missing
        tok = x.unsqueeze(-1) * self.weight + self.bias  # (B, F, d)
        if mask is not None:
            tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        return torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)


class ImageTokenizer(nn.Module):
    def __init__(self, in_ch, dim, kv_pool=1):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, dim, kernel_size=1)
        # conditional positional encoding: works for any (D, H, W), so T2 and
        # DWI grids of different sizes need no fixed-size position table
        self.pos = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.branch_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.kv_pool = kv_pool

    def forward(self, fmap):
        x = self.proj(fmap)
        x = x + self.pos(x)
        if self.kv_pool > 1:
            x = F.avg_pool3d(x, kernel_size=(1, self.kv_pool, self.kv_pool),
                             ceil_mode=True)
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N, d)
        return x + self.branch_emb, grid


class FusionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, residual_query=True):
        super().__init__()
        self.residual_query = residual_query
        self.ln_q, self.ln_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, q, kv):
        attended, attn = self.cross(
            self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
            need_weights=True, average_attn_weights=False,
        )  # attn: (B, heads, Q, N)
        q = q + attended if self.residual_query else attended
        h = self.ln_s(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        q = q + self.ffn(self.ln_f(q))
        return q, attn


class CrossAttnFusionModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        cfg = config.get("xattn", {}) or {}
        dim = cfg.get("dim", 128)
        heads = cfg.get("heads", 4)
        n_layers = cfg.get("layers", 1)
        dropout = cfg.get("dropout", 0.1)
        self.feature_mask_prob = cfg.get("feature_mask_prob", 0.0)
        self.train_fc = cfg.get("train_fc", False)

        self.tap = cfg.get("tap", "layer4")
        self.series = series_keys(config)
        assert len(self.series) == 3
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"] and set(
            self.DWI_KEYS
        ).issubset(self.series)

        # Same branch names / layout as the baseline, so the non-strict loader in
        # train.py fills branches.* and fc.* from model_weights.model_ckpt.
        self.branches = nn.ModuleDict()
        self.branch_specs = []
        dwi_added = False
        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not dwi_added:
                    self.branches["adc_b1500"] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    dwi_added = True
                continue
            self.branches[key] = ResNetBranch(Bottleneck, [3, 4, 6, 3], 1)
            self.branch_specs.append((key, (key,)))
        self.feature_dim = 2048 * len(self.branch_specs)

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.ReLU(inplace=False),
            self.dropout, nn.Linear(256, 2),
        )

        # Aliases the baseline checkpoint's keys may use (as in aug11cbam).
        if "axt2" in self.branches:
            self.resnet_single_branch = self.branches["axt2"]
        if "adc_b1500" in self.branches:
            self.resnet_dual_branch1 = self.branches["adc_b1500"]

        # ---- cross-attention fusion ----
        n_features = cfg.get("n_features") or config["data"].get("tabular_dims") or 37
        self.clin_tok = ClinicalTokenizer(n_features, dim)
        self.img_tok = nn.ModuleDict({
            name: ImageTokenizer(TAP_CHANNELS[self.tap], dim, cfg.get("kv_pool", 1))
            for name, _ in self.branch_specs
        })
        rq = cfg.get("residual_query", True)
        self.blocks = nn.ModuleList([
            FusionBlock(dim, heads, dropout, residual_query=(rq or i > 0))
            for i in range(n_layers)
        ])
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, 2),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        self.last_attn = None  # for visualization: list of (B, heads, Q, N) per block
        self.last_grids = None

    # ------------------------------------------------------------------ #
    def _fusion_modules(self):
        mods = [self.clin_tok, self.img_tok, self.blocks, self.delta_head]
        return mods + ([self.fc] if self.train_fc else [])

    def on_train_start(self):
        for p in self.parameters():
            p.requires_grad = False
        for m in self._fusion_modules():
            for p in m.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.branches.eval()  # keep BatchNorm running stats frozen
        return self

    def configure_optimizers(self):
        params = [p for m in self._fusion_modules() for p in m.parameters()]
        optimizer = torch.optim.AdamW(
            params, self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )
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

        return self.fc(torch.cat(pooled, dim=1)) + self.delta_head(q[:, 0])

    # ------------------------------------------------------------------ #
    def _shared_step(self, batch):
        logits = self(batch["volume_data_dict"], batch["tabular_features"])
        return logits, batch["label"]

    def training_step(self, batch, batch_idx):
        logits, target = self._shared_step(batch)
        loss = self.criterion(logits, target)
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _eval_step(self, batch, dataloader_idx, log_name):
        logits, target = self._shared_step(batch)
        loss = self.unweighted_loss(logits, target)
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
            if log_name == "val_loss" and self.log_configs["log_run"]:
                self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return {log_name: loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, dataloader_idx, "val_loss")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, dataloader_idx, "test_loss")
