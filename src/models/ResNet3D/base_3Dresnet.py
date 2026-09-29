"""
A PyTorch Lightning ResNet3D implementation structured to be initialized with
Med3D weights.

It is a modified version of the ResNet3D implementation from the MedicalNet repository:
https://github.com/Tencent/MedicalNet

Medical Net (Med3D) is a 3D ResNet-based neural network for medical image analysis.
It is a PyTorch implementation of the ResNet-50 3D architecture from the paper
"Med3D: Transfer Learning for 3D Medical Image Analysis" by Chen et al. (2019).

Patricia Johnson and Amritha Musipatla
Dec 25 2023
"""

from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.metrics.metrics import epoch_end_metrics
from src.metrics.plot import save_preds


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBranch(nn.Module):
    def __init__(self, block, layers, in_chans, shortcut_type="B", no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_chans,
            64,
            kernel_size=[7, 7, 7],
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        return out


class Base3DResNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        self.resnet_single_branch = ResNetBranch(
            Bottleneck, [3, 4, 6, 3], 1
        )  # 3D ResNet branch 1

        self._set_default_config()
        self._set_config(config)

        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 2),
        )

    def forward(self, data_dict):
        # single volume branch
        vol = data_dict[list(data_dict.keys())[0]]

        x = self.resnet_single_branch(vol)

        out = self.fc(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hyperparams["max_patience"],
            factor=self.hyperparams["factor"],
            threshold=1e-4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def _set_default_config(self):
        # set default hyperparams
        self.hyperparams = {}
        self.hyperparams["learning_rate"] = 1e-04
        self.hyperparams["weight_decay"] = 0.001
        self.hyperparams["max_patience"] = 5
        self.hyperparams["factor"] = 0.3
        self.hyperparams["dropout"] = 0.5

        # set logging hyperparams
        self.log_run = False

    def _set_config(self, config):
        self.config = config

        # set the hyperparameters
        for key in self.hyperparams.keys():
            hyperparam_keys = config["hyperparameters"].keys()
            if key in hyperparam_keys:
                self.hyperparams[key] = float(config["hyperparameters"][key])

        # define loss
        if config["training"]["imbalance_strategy"] == "weighted_loss":
            self.criterion = nn.CrossEntropyLoss(
                weight=config["training"]["class_weights"],
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.unweighted_loss = nn.CrossEntropyLoss()

        # save predictions for epoch level metrics
        self.val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.reader_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.extra_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.train_preds = {"preds": [], "targets": []}
        self.epoch_count = 0

        # save best metric across epochs
        self.best_val_pirads_auc = 0

        # log hyperparams
        self.log_configs = config["logging"]

    def training_step(self, batch, batch_idx):
        data_dict = batch["volume_data_dict"]
        target = batch["label"]

        logits = self(data_dict)
        loss = self.criterion(logits, target)
        unweighted_loss = self.unweighted_loss(logits, target)

        # save predictions for computing metrics at the end of the epoch
        self.train_preds["preds"].append(logits)
        self.train_preds["targets"].append(target)

        # log loss
        if self.log_configs["log_run"]:
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            self.log(
                "train_unweighted_loss",
                unweighted_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch["volume_data_dict"]
        target = batch["label"]

        logits = self(data_dict)
        loss = self.unweighted_loss(logits, target)

        # save predictions for computing metrics at the end of the epoch
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])

            if self.log_configs["log_run"]:
                self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        elif dataloader_idx == 1:
            self.reader_val_preds["preds"].append(logits)
            self.reader_val_preds["targets"].append(target)
            self.reader_val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.reader_val_preds["AccessionNumber"].append(batch["AccessionNumber"])

            if self.log_configs["log_run"]:
                self.log(
                    "reader_val_loss", loss, prog_bar=True, on_step=True, on_epoch=True
                )
        elif dataloader_idx == 2:
            self.extra_val_preds["preds"].append(logits)
            self.extra_val_preds["targets"].append(target)
            self.extra_val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.extra_val_preds["AccessionNumber"].append(batch["AccessionNumber"])

            if self.log_configs["log_run"]:
                self.log(
                    "top5_val_loss", loss, prog_bar=True, on_step=True, on_epoch=True
                )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch["volume_data_dict"]
        target = batch["label"]

        logits = self(data_dict)
        loss = self.unweighted_loss(logits, target)

        # save predictions for computing metrics at the end of the epoch
        if dataloader_idx == 0:
            self.val_preds["preds"].append(logits)
            self.val_preds["targets"].append(target)
            self.val_preds["maxPIRADS"].append(batch["maxPIRADS"])
            self.val_preds["AccessionNumber"].append(batch["AccessionNumber"])

            print("Test Loss: ", loss)

        return {"test_loss": loss}

    def on_train_epoch_end(self):
        if self.log_configs["log_run"]:
            # convert list of tensors to single tensor
            train_preds = torch.cat(self.train_preds["preds"], dim=0)
            train_targets = torch.cat(self.train_preds["targets"], dim=0)

            # compute per epoch stats
            auc, opt_threshold, precision, recall, f1 = epoch_end_metrics(
                train_preds, train_targets, self.epoch_count
            )

            # log metrics
            self.log("train_auc", auc, prog_bar=True, on_epoch=True)

        # reset train_preds for next epoch
        self.train_preds = {"preds": [], "targets": []}

    def on_validation_epoch_end(self):
        # Compute and log validation metrics whenever we have predictions, even
        # when log_run is False. The EarlyStopping/ModelCheckpoint callbacks
        # monitor `best_val_pirads_auc`, so this must run for single-task
        # training to checkpoint and early-stop (W&B logging is a separate,
        # optional destination handled by self.log when a logger is attached).
        if self.val_preds["preds"]:
            # convert list of tensors to single tensor
            val_preds = torch.cat(self.val_preds["preds"], dim=0)
            val_targets = torch.cat(self.val_preds["targets"], dim=0)
            val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)
            val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)

            # compute per epoch stats
            auc, opt_threshold, precision, recall, f1 = epoch_end_metrics(
                val_preds,
                val_targets,
                self.epoch_count,
                plot_roc=self.log_configs["plot_roc"],
                plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
                plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
                pirads=self.log_configs["val_pirads"],
                mode="val",
            )

            if auc > self.best_val_pirads_auc:
                self.best_val_pirads_auc = auc

            # configure_optimizers' ReduceLROnPlateau monitors val_loss, but the
            # validation steps only log it when log_run is True, so with
            # log_run False training crashed after the first validation epoch.
            # Log the epoch-level unweighted loss here in that case.
            if not self.log_configs["log_run"]:
                self.log(
                    "val_loss",
                    self.unweighted_loss(val_preds, val_targets.long()),
                    on_epoch=True,
                )

            # log metrics
            self.log("val_auc", auc, prog_bar=True, on_epoch=True)
            self.log("val_precision", precision, prog_bar=True, on_epoch=True)
            self.log("val_recall", recall, prog_bar=True, on_epoch=True)
            self.log(
                "best_val_pirads_auc",
                self.best_val_pirads_auc,
                prog_bar=True,
                on_epoch=True,
            )

            if self.config["debugging"]["debug"]:
                # save predictions for debugging
                save_preds(
                    val_preds,
                    val_targets,
                    val_max_pirads,
                    val_acc_num,
                    self.config["debugging"]["preds_dir"],
                    self.epoch_count,
                )

            if self.config["data"].get("readers", "all") != "all":
                val_preds = torch.cat(self.reader_val_preds["preds"], dim=0)
                val_targets = torch.cat(self.reader_val_preds["targets"], dim=0)
                val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)
                val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)

                # compute per epoch stats
                auc, opt_threshold, precision, recall, f1 = epoch_end_metrics(
                    val_preds,
                    val_targets,
                    self.epoch_count,
                    plot_roc=self.log_configs["plot_roc"],
                    plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
                    plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
                    pirads=self.log_configs["val_pirads"],
                    mode="val",
                )

                if auc > self.best_val_pirads_auc:
                    self.best_val_pirads_auc = auc

                # log metrics
                self.log("reader_val_auc", auc, on_epoch=True)
                self.log("reader_val_precision", precision, on_epoch=True)
                self.log("reader_val_recall", recall, on_epoch=True)
                self.log("reader_best_val_auc", self.best_val_pirads_auc, on_epoch=True)

                if self.config["debugging"]["debug"]:
                    save_preds(
                        val_preds,
                        val_targets,
                        val_max_pirads,
                        val_acc_num,
                        self.config["debugging"]["preds_dir"],
                        self.epoch_count,
                        save_name="reader",
                    )

            if self.config["paths"]["extra_valid_csv"] is not None:
                val_preds = torch.cat(self.extra_val_preds["preds"], dim=0)
                val_targets = torch.cat(self.extra_val_preds["targets"], dim=0)
                val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)
                val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)

                # compute per epoch stats
                auc, opt_threshold, precision, recall, f1 = epoch_end_metrics(
                    val_preds,
                    val_targets,
                    self.epoch_count,
                    plot_roc=self.log_configs["plot_roc"],
                    plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
                    plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
                    pirads=self.log_configs["val_pirads"],
                    mode="val",
                )
                if auc > self.best_val_pirads_auc:
                    self.best_val_pirads_auc = auc

                # log metrics
                self.log("top5_val_auc", auc, on_epoch=True)
                self.log("top5_val_precision", precision, on_epoch=True)
                self.log("top5_val_recall", recall, on_epoch=True)
                self.log("top5_best_val_auc", self.best_val_pirads_auc, on_epoch=True)

                if self.config["debugging"]["debug"]:
                    save_preds(
                        val_preds,
                        val_targets,
                        val_max_pirads,
                        val_acc_num,
                        self.config["debugging"]["preds_dir"],
                        self.epoch_count,
                        save_name="top5",
                    )

        # reset val_preds for next epoch
        self.val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.reader_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.extra_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }

        # increment epoch count
        self.epoch_count += 1

    def on_test_epoch_end(self):
        # convert list of tensors to single tensor
        val_preds = torch.cat(self.val_preds["preds"], dim=0)
        val_targets = torch.cat(self.val_preds["targets"], dim=0)
        val_max_pirads = torch.cat(self.val_preds["maxPIRADS"], dim=0)
        val_acc_num = torch.cat(self.val_preds["AccessionNumber"], dim=0)

        # compute per epoch stats
        auc, opt_threshold, precision, recall, f1 = epoch_end_metrics(
            val_preds,
            val_targets,
            self.epoch_count,
            plot_roc=self.log_configs["plot_roc"],
            plot_confusion_matrix=self.log_configs["plot_confusion_matrix"],
            plot_pirads_breakdown=self.log_configs["plot_pirads_breakdown"],
            pirads=self.log_configs["val_pirads"],
            mode="test",
        )

        # log metrics
        self.log("test_auc", auc, on_epoch=True)
        self.log("test_precision", precision, on_epoch=True)
        self.log("test_recall", recall, on_epoch=True)

        # save predictions for debugging
        save_preds(
            val_preds,
            val_targets,
            val_max_pirads,
            val_acc_num,
            self.config["debugging"]["preds_dir"],
            str(auc),
            save_name=getattr(self, "save_name", None),
        )

        # reset val_preds for next epoch
        self.val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.reader_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }
        self.extra_val_preds = {
            "preds": [],
            "targets": [],
            "maxPIRADS": [],
            "AccessionNumber": [],
        }

        # increment epoch count
        self.epoch_count += 1
