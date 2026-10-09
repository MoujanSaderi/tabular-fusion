"""
Clinically conditioned CBAM (Woo et al., ECCV 2018) on a frozen ResNet3D backbone.

Follows the original CBAM design:
  * channel attention pools each channel with BOTH average and max pooling and
    scores the two descriptors with one shared bottleneck MLP (C -> C/r -> C),
    then sums the two logits and applies a sigmoid;
  * spatial attention pools across channels with BOTH average and max pooling,
    stacks the two maps, and applies one large-kernel conv (7x7x7) + sigmoid;
  * channel attention runs first, then spatial attention;
  * by default a CBAM module sits in EVERY bottleneck block, on the residual
    branch just before the skip-connection add, so the identity path is never
    gated.

Additions for clinical conditioning and the frozen backbone (see `cbam:` in
configs/cbam.yaml):
  * channel attention: the clinical embedding is concatenated to each pooled
    descriptor before the shared MLP, so the channel gate depends jointly on
    image content and the patient's clinical profile, and the MLP stays shared;
  * spatial attention: the clinical embedding FiLM-modulates the stacked
    avg/max maps (per-map scale and shift) before the conv. Clinical context
    also reaches the spatial gate indirectly, since the maps are pooled from
    features the clinically conditioned channel gate already reweighted;
  * identity_init: gates are 2*sigmoid(.) with the last layer of each gate
    zero-initialised, so every CBAM module starts as the identity and the
    frozen pretrained backbone initially behaves exactly as it was trained.
    Set identity_init: false for the paper's plain sigmoid gates.
"""
import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch

from aug11latefusionflat import ClinicalEncoder

STAGES = ("layer1", "layer2", "layer3", "layer4")

DEFAULT_CBAM_CONFIG = {
    # "blocks": a CBAM module inside every bottleneck of the listed stages
    #           (original CBAM placement).
    # "post":   a single CBAM module on the layer4 output, before avgpool
    #           (the previous version of this model), for ablations.
    "placement": "blocks",
    "stages": [1, 2, 3, 4],  # only used with placement "blocks"
    "reduction": 16,
    "kernel_size": 7,
    "spatial_clinical": "film",  # "film" | "none"
    "identity_init": True,
}


def _gate(logits, identity_init):
    # identity_init: gate in (0, 2), exactly 1 when the logits are 0.
    return 2.0 * torch.sigmoid(logits) if identity_init else torch.sigmoid(logits)


class ClinicalChannelAttention(nn.Module):
    """Mc = gate(MLP([avg(F); c]) + MLP([max(F); c])), one MLP shared by both."""

    def __init__(self, channels, clin_dim, reduction=16, identity_init=True):
        super().__init__()
        self.identity_init = identity_init
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels + clin_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        if identity_init:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, clin):
        # x: (B, C, D, H, W), clin: (B, clin_dim)
        avg = x.mean(dim=(2, 3, 4))
        mx = x.amax(dim=(2, 3, 4))
        logits = self.mlp(torch.cat([avg, clin], dim=1)) + self.mlp(torch.cat([mx, clin], dim=1))
        weights = _gate(logits, self.identity_init)
        return x * weights[:, :, None, None, None]


class ClinicalSpatialAttention(nn.Module):
    """Ms = gate(conv([avg_c(F); max_c(F)])), with the two pooled maps optionally
    FiLM-modulated by the clinical embedding before the conv."""

    def __init__(self, clin_dim, kernel_size=7, spatial_clinical="film", identity_init=True):
        super().__init__()
        if spatial_clinical not in ("film", "none"):
            raise ValueError(f"cbam.spatial_clinical must be 'film' or 'none', got {spatial_clinical!r}")
        self.identity_init = identity_init
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # (gamma, beta) for each of the 2 pooled maps. Zero-initialised so FiLM
        # starts as the identity regardless of identity_init.
        self.film = nn.Linear(clin_dim, 4) if spatial_clinical == "film" else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if identity_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x, clin):
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        if self.film is not None:
            gamma, beta = self.film(clin).view(-1, 2, 2).unbind(dim=1)  # each (B, 2)
            pooled = pooled * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
        weights = _gate(self.conv(pooled), self.identity_init)  # (B, 1, D, H, W)
        return x * weights


class ClinicalCBAM(nn.Module):
    """Channel attention, then spatial attention."""

    def __init__(self, channels, clin_dim, cfg):
        super().__init__()
        self.channel_attn = ClinicalChannelAttention(
            channels, clin_dim, reduction=cfg["reduction"], identity_init=cfg["identity_init"]
        )
        self.spatial_attn = ClinicalSpatialAttention(
            clin_dim,
            kernel_size=cfg["kernel_size"],
            spatial_clinical=cfg["spatial_clinical"],
            identity_init=cfg["identity_init"],
        )

    def forward(self, x, clin):
        return self.spatial_attn(self.channel_attn(x, clin), clin)


def _bottleneck_forward(block, x, clin):
    """Bottleneck.forward with block.cbam (if present) applied to the residual
    branch before the skip-connection add, as in the CBAM paper."""
    residual = x
    out = block.relu(block.bn1(block.conv1(x)))
    out = block.relu(block.bn2(block.conv2(out)))
    out = block.bn3(block.conv3(out))
    if getattr(block, "cbam", None) is not None:
        out = block.cbam(out, clin)
    if block.downsample is not None:
        residual = block.downsample(x)
    out = out + residual
    return block.relu(out)


class ResNetBranchCBAM(ResNetBranch):
    """ResNetBranch with clinically conditioned CBAM modules.

    The pretrained layers keep their original parameter names; CBAM modules are
    registered as new children (layerN.i.cbam.* or post_cbam.*), so a baseline
    checkpoint still loads into the backbone and simply has no CBAM weights.
    """

    def __init__(self, block, layers, in_chans, clin_dim, cbam_cfg):
        super().__init__(block, layers, in_chans)
        self.placement = cbam_cfg["placement"]
        self.post_cbam = None

        if self.placement == "blocks":
            for stage_idx in cbam_cfg["stages"]:
                for blk in getattr(self, STAGES[stage_idx - 1]):
                    blk.cbam = ClinicalCBAM(blk.bn3.num_features, clin_dim, cbam_cfg)
        elif self.placement == "post":
            self.post_cbam = ClinicalCBAM(self.layer4[-1].bn3.num_features, clin_dim, cbam_cfg)
        else:
            raise ValueError(f"cbam.placement must be 'blocks' or 'post', got {self.placement!r}")

    def cbam_modules(self):
        return [m for m in self.modules() if isinstance(m, ClinicalCBAM)]

    def forward(self, x, clin):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for stage in STAGES:
            for blk in getattr(self, stage):
                out = _bottleneck_forward(blk, out, clin)
        if self.post_cbam is not None:
            out = self.post_cbam(out, clin)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)


def _cbam_config(config):
    cfg = dict(DEFAULT_CBAM_CONFIG)
    cfg.update(config.get("cbam") or {})
    stages = cfg["stages"]
    if cfg["placement"] == "blocks" and (not stages or not set(stages) <= {1, 2, 3, 4}):
        raise ValueError(f"cbam.stages must be a non-empty subset of [1, 2, 3, 4], got {stages!r}")
    if cfg["kernel_size"] % 2 != 1:
        raise ValueError(f"cbam.kernel_size must be odd, got {cfg['kernel_size']}")
    return cfg


class TriSeriesModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
        self.cbam_cfg = _cbam_config(config)
        self.stack_adc_b1500 = config["training"]["stack_adc_b1500"]
        # Accept series either as SeriesType enums or as the plain strings
        # train.py passes straight through from the YAML config.
        self.series = [
            s if isinstance(s, str) else s.value["key"] for s in config["data"]["series"]
        ]
        assert len(self.series) == 3

        # Frozen clinical MLP encoder loaded from model_weights.clinical_ckpt,
        # shared with aug11latefusionflat / mlpclinical so the architecture and
        # checkpoint format can't drift. tabular_dims is filled in by train.py
        # from the tabular CSV; if absent it falls back to the checkpoint's width.
        self.clinical_encoder = ClinicalEncoder(
            in_dims=config["data"].get("tabular_dims"),
            clinical_ckpt=config["model_weights"].get("clinical_ckpt"),
            frozen=True,
        )
        clin_dim = self.clinical_encoder.out_dim

        series_set = set(self.series)
        self.stack_adc_b1500 = self.stack_adc_b1500 and set(self.DWI_KEYS).issubset(
            series_set
        )

        self.branches = nn.ModuleDict()
        self.branch_specs = []
        self.feature_dim = 0
        self._dwi_branch_added = False

        def make_branch(in_chans):
            return ResNetBranchCBAM(Bottleneck, [3, 4, 6, 3], in_chans, clin_dim, self.cbam_cfg)

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    self.branches["adc_b1500"] = make_branch(2)
                    self.branch_specs.append(("adc_b1500", self.DWI_KEYS))
                    self.feature_dim += 2048
                    self._dwi_branch_added = True
                continue
            self.branches[key] = make_branch(1)
            self.branch_specs.append((key, (key,)))
            self.feature_dim += 2048

        self.dropout = nn.Dropout(p=config["hyperparameters"]["dropout"])
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=False),
            self.dropout,
            nn.Linear(256, 2),
        )

        # Backwards-compatible attributes for existing utilities (e.g., Grad-CAM)
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

        self._apply_freeze(verbose=True)

    # ------------------------------------------------------------------ #
    # Freezing: only the CBAM modules and the fc head train.
    # ------------------------------------------------------------------ #
    def _cbam_modules(self):
        return [m for branch in self.branches.values() for m in branch.cbam_modules()]

    def _apply_freeze(self, verbose=False):
        # By module, not by parameter name: the branches are also registered
        # under alias attributes (resnet_single_branch, ...), and
        # named_parameters() reports shared parameters under the alias only.
        for param in self.parameters():
            param.requires_grad = False
        for module in [self.fc, *self._cbam_modules()]:
            for param in module.parameters():
                param.requires_grad = True
        if verbose:
            n_cbam = sum(p.numel() for m in self._cbam_modules() for p in m.parameters())
            n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
            where = (
                f"every block of stages {self.cbam_cfg['stages']}"
                if self.cbam_cfg["placement"] == "blocks"
                else "layer4 output"
            )
            print(f"[CBAM] {len(self._cbam_modules())} modules ({where}), "
                  f"identity_init={self.cbam_cfg['identity_init']}, "
                  f"spatial_clinical={self.cbam_cfg['spatial_clinical']}; "
                  f"CBAM params={n_cbam:,}, trainable params={n_train:,}")

    def on_train_start(self):
        print("Freezing ResNet backbone and clinical encoder; training CBAM + fc...")
        self._apply_freeze()

    def train(self, mode=True):
        # Lightning calls .train() again after every validation loop; keep the
        # frozen backbone (and its BatchNorm running stats) in eval mode. CBAM
        # has no BatchNorm or dropout, so its mode only matters for clarity.
        super().train(mode)
        if not hasattr(self, "branches"):  # called before __init__ finished
            return self
        self.branches.eval()
        for module in self._cbam_modules():
            module.train(mode)
        return self

    def forward(self, data_dict, tabular_features):
        clinical_emb = self.clinical_encoder(tabular_features)
        features = []
        for branch_name, keys in self.branch_specs:
            if len(keys) > 1:
                inputs = torch.cat([data_dict[k] for k in keys], dim=1)
            else:
                inputs = data_dict[keys[0]]
            features.append(self.branches[branch_name](inputs, clinical_emb))

        x = torch.cat(features, dim=1)
        out = self.fc(x)
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
