import torch
import torch.nn as nn

from src.models.ResNet3D.base_3Dresnet import Base3DResNet
from src.models.ResNet3D.base_3Dresnet import Bottleneck
from src.models.ResNet3D.base_3Dresnet import ResNetBranch

from aug11latefusionflat import ClinicalEncoder
"""
Clinical features define how important each channel is for each specific patient. Our output 
from this function is essentially 2048 numbers indicating how credibly important each channel is, and we just
multiply the MRI features with those weights.

The output is the original MRI feature map, but with each channel scaled by its importance weight.
"""

class ClinicalChannelAttention(nn.Module):
    def __init__(self, channel_dim=2048, tabular_dim=64, reduction=16):
        super().__init__()
        # MLP: squeezed features + clinical → channel weights
        self.mlp = nn.Sequential(
            nn.Linear(channel_dim + tabular_dim, channel_dim // reduction),
            nn.ReLU(),
            nn.Linear(channel_dim // reduction, channel_dim),
        )

    def forward(self, x, tabular_features):
        # x: (batch, 2048, H, W, D)
        # squeeze spatial dims → (batch, 2048)
        avg = x.mean(dim=[2, 3, 4])
        # concatenate with clinical features → (batch, 2048 + 64)
        combined = torch.cat([avg, tabular_features], dim=1)
        # MLP → (batch, 2048) weights
        weights = torch.sigmoid(self.mlp(combined))
        # reshape for broadcasting → (batch, 2048, 1, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * weights


"""
This averages the 2048 features into one spatial map. This ends up giving us a
3D heatmap of where the ResNet found the most interesting features in the MRI, with
one value per location indicating how much activity is there
at each location overall.

The clinical features give us a global shift here as a bias, and we shift the entire
spatial map based on clinical context. We then smooth the map, and nearby locations also end up influencing each other

Essentially both channel and spatial attention
functions are receiving these clinical features independently, one for the purposes of
knowing what to look for, and one for knowing where to look for.
"""
class ClinicalSpatialAttention(nn.Module):
    def __init__(self, tabular_dim=64):
        super().__init__()
        # MLP: clinical → 1 value, then conv creates spatial map
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # conv: takes channel-averaged features + clinical bias → spatial map
        self.conv = nn.Conv3d(1, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x, tabular_features):
        # x: (batch, 2048, H, W, D)
        # average across channels → (batch, 1, H, W, D)
        avg = x.mean(dim=1, keepdim=True)
        # clinical bias → (batch, 1)
        bias = self.mlp(tabular_features)
        # reshape bias → (batch, 1, 1, 1, 1) and broadcast
        bias = bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # add clinical bias to spatial map
        spatial = avg + bias
        # conv refines the spatial map
        spatial = self.conv(spatial)
        # sigmoid → weights between 0 and 1
        weights = torch.sigmoid(spatial)  # (batch, 1, H, W, D)
        return x * weights


class ResNetBranchEarly(ResNetBranch):
    """ResNet branch with clinical CBAM after layer4. The clinical embedding is
    computed once by the model-level ClinicalEncoder and passed in, so every
    branch attends with the same (loaded) encoder output."""

    def __init__(self, block, layers, in_chans, tabular_dim=64):
        super().__init__(block, layers, in_chans)
        self.channel_attn = ClinicalChannelAttention(
            channel_dim=2048,
            tabular_dim=tabular_dim
        )
        self.spatial_attn = ClinicalSpatialAttention(tabular_dim=tabular_dim)

    def forward(self, x, clinical_emb):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # CBAM: channel attention first, then spatial attention
        out = self.channel_attn(out, clinical_emb)
        out = self.spatial_attn(out, clinical_emb)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class TriSeriesModel(Base3DResNet):
    DWI_KEYS = ("adc", "b1500")

    def __init__(self, config):
        super().__init__(config)
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

        for key in self.series:
            if self.stack_adc_b1500 and key in self.DWI_KEYS:
                if not self._dwi_branch_added:
                    branch_name = "adc_b1500"
                    self.branches[branch_name] = ResNetBranchEarly(
                        Bottleneck, [3, 4, 6, 3], 2, tabular_dim=clin_dim
                    )
                    self.branch_specs.append((branch_name, self.DWI_KEYS))
                    self.feature_dim += 2048  # fixed: was 2065, now 2048
                    self._dwi_branch_added = True
                continue

            branch_name = key
            self.branches[branch_name] = ResNetBranchEarly(Bottleneck, [3, 4, 6, 3], 1, tabular_dim=clin_dim)
            self.branch_specs.append((branch_name, (key,)))
            self.feature_dim += 2048  # fixed: was 2065, now 2048

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

    def on_train_start(self):
        print("Freezing ResNet branches (permanent)...")
        for name, param in self.named_parameters():
            if "channel_attn" in name or "spatial_attn" in name or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for branch in self.branches.values():
            branch.conv1.eval()
            branch.bn1.eval()
            branch.layer1.eval()
            branch.layer2.eval()
            branch.layer3.eval()
            branch.layer4.eval()

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