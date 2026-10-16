"""
Rule-based PI-RADS baseline: predicts 1 when maxPIRADS >= pirads_threshold
(default 3) and 0 otherwise. No learned parameters.

It plugs into the same Lightning test / metrics / save_preds pipeline as the
trained models, so its numbers are directly comparable to theirs:

    python train.py --config configs/pirads_rule.yaml   # no fitting; runs trainer.test()
    python eval.py  --config configs/pirads_rule.yaml --out preds_pirads_rule.csv  # no --ckpt

maxPIRADS is read from the split CSV (batch["maxPIRADS"]), so the loader never
opens the H5/MRI volumes and no tabular CSV is needed.

The rule's output is returned as 2-class logits (+/-RULE_LOGIT_MARGIN), so that
softmax gives probabilities of ~1 / ~0 for the predicted / other class.
"""
import torch

from src.models.ResNet3D.base_3Dresnet import Base3DResNet

DEFAULT_PIRADS_THRESHOLD = 3
# softmax([-10, 10]) = [2e-9, 1 - 2e-9]: effectively a hard 0/1 prediction.
RULE_LOGIT_MARGIN = 10.0


class PIRADSRuleClassifier(Base3DResNet):
    # Tells eval.py to pass batch["maxPIRADS"] to forward() and not to expect
    # a checkpoint.
    uses_max_pirads = True
    requires_checkpoint = False

    def __init__(self, config):
        super().__init__(config)

        # Base3DResNet always builds a ResNet3D branch + fc head; this model
        # uses neither.
        del self.resnet_single_branch
        del self.fc

        self.pirads_threshold = int(
            config["model"].get("pirads_threshold", DEFAULT_PIRADS_THRESHOLD)
        )
        print(f"[pirads_rule] predicting 1 when maxPIRADS >= {self.pirads_threshold}")

    def forward(self, data_dict, tabular_features, max_pirads=None):
        if max_pirads is None:
            raise ValueError("PIRADSRuleClassifier needs batch['maxPIRADS'].")
        pred = (max_pirads >= self.pirads_threshold).float()  # (B,)
        signed = (2 * pred - 1) * RULE_LOGIT_MARGIN           # +margin for 1, -margin for 0
        return torch.stack([-signed, signed], dim=1)          # (B, 2) logits

    def _record(self, batch):
        target = batch["label"]
        logits = self(batch["volume_data_dict"], batch["tabular_features"], batch["maxPIRADS"])
        loss = self.unweighted_loss(logits, target)
        self.val_preds["preds"].append(logits)
        self.val_preds["targets"].append(target)
        self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
        self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])
        return loss

    def training_step(self, batch, batch_idx):
        raise RuntimeError(
            "pirads_rule has nothing to train; train.py runs trainer.test() for it directly."
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return {"val_loss": self._record(batch)}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            return {"test_loss": self._record(batch)}
