"""
Pieces shared by the models whose cross-attention CLS token feeds the head as
features (crossattnfeat.CrossAttnFeatFusion, crossattnlatefeat.CrossAttnLateFeatFusion):

    cls_feat = cls_proj(CLS)                     LayerNorm -> Linear -> GELU -> Dropout
    hidden   = ReLU(fc.0(...) + cls_in(cls_feat)) cls_in: Linear(dim -> hidden), no bias

cls_proj has the same layer names in both models (and the same layout as
crossattnfusion's delta_head minus its readout), so it transfers between them.

ClsFeatureMonitorMixin logs, every validation epoch:
    val_auc_no_cls      val AUC of the same model with cls_feat zeroed
    cls_auc_gain        val_auc - val_auc_no_cls (~0: the head ignores the CLS)
    cls/feat_spread     std of cls_feat across patients, averaged over dims
                        (~0: the CLS is the same for everyone)
    cls/in_weight_norm  norm of cls_in's weights
Both AUCs use softmax probabilities (sklearn roc_auc_score). The model's
forward must set self._last_no_cls and self._last_cls_spread in eval mode
(see cls_eval_stats).
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def build_cls_proj(dim, dropout):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))


def build_cls_in(dim, hidden, init_std):
    std = float(init_std)  # float(): YAML may read 1e-3 as a string
    if not std > 0:
        raise ValueError(f"xattn.cls_init_std must be > 0 (small random init), got {std}")
    layer = nn.Linear(dim, hidden, bias=False)
    nn.init.normal_(layer.weight, std=std)
    return layer


def head_with_cls(fc, base_pre, cls_term):
    """Rest of a Linear -> ReLU -> Dropout -> Linear head, given fc.0's output."""
    return fc[3](fc[2](fc[1](base_pre + cls_term)))


def check_head_layout(fc, who):
    if not (len(fc) == 4 and isinstance(fc[0], nn.Linear) and isinstance(fc[1], nn.ReLU)
            and isinstance(fc[2], nn.Dropout) and isinstance(fc[3], nn.Linear)):
        raise TypeError(f"[{who}] unexpected head layout (expected Linear, ReLU, Dropout, Linear): {fc}")


class ClsFeatureMonitorMixin:
    """Put first in the bases so its validation hooks wrap the model's own."""

    def _reset_cls_monitor(self):
        self._val_no_cls = {"full": [], "no_cls": [], "targets": [], "spread": []}

    def cls_eval_stats(self, fc, base_pre, cls_feat):
        """Call from forward(); records the no-CLS logits in eval mode."""
        if self.training:
            return
        with torch.no_grad():
            self._last_no_cls = fc[3](fc[2](fc[1](base_pre)))
            self._last_cls_spread = (
                cls_feat.std(dim=0, unbiased=False).mean() if cls_feat.size(0) > 1 else None
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = super().validation_step(batch, batch_idx, dataloader_idx)
        if dataloader_idx == 0:
            self._val_no_cls["full"].append(self.val_preds["preds"][-1].detach())
            self._val_no_cls["no_cls"].append(self._last_no_cls)
            self._val_no_cls["targets"].append(batch["label"].detach())
            if self._last_cls_spread is not None:
                self._val_no_cls["spread"].append(self._last_cls_spread)
        return out

    def on_validation_epoch_start(self):
        self._reset_cls_monitor()
        parent = getattr(super(), "on_validation_epoch_start", None)
        if parent is not None:
            parent()

    def on_validation_epoch_end(self):
        acc = self._val_no_cls
        if acc["targets"]:
            targets = torch.cat(acc["targets"]).cpu().numpy()
            if len(set(targets.tolist())) > 1:
                def auc(logits):
                    probs = torch.softmax(torch.cat(logits).float(), dim=1)[:, 1].cpu().numpy()
                    return roc_auc_score(targets, probs)
                auc_full, auc_no_cls = auc(acc["full"]), auc(acc["no_cls"])
                self.log("val_auc_no_cls", auc_no_cls, on_epoch=True)
                self.log("cls_auc_gain", auc_full - auc_no_cls, on_epoch=True)
            if acc["spread"]:
                self.log("cls/feat_spread", torch.stack(acc["spread"]).mean(), on_epoch=True)
            self.log("cls/in_weight_norm", self.cls_in.weight.detach().norm(), on_epoch=True)
        self._reset_cls_monitor()
        super().on_validation_epoch_end()
