import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torchmetrics import AUROC
from torchmetrics import ROC

from src.metrics.plot import plot_pirads_cm
from src.metrics.plot import plot_roc_curve


def weighted_loss(prediction, target, loader, device="cpu"):
    """Calculate weighted binary cross entropy loss

    args:
        prediction: torch tensor of shape (batch_size, 1)
        target: torch.float32 tensor of shape (batch_size, 1).
        loader: src.data.loader.ExamH5Dataset

    returns:
        loss: torch tensor
    """
    weights_npy = np.array(
        [loader.dataset.class_weights[int(t.item())].item() for t in target.detach()]
    )
    weights_tensor = torch.tensor(weights_npy, dtype=torch.float32, device=device)
    weights_tensor = weights_tensor.view_as(target)
    loss = F.binary_cross_entropy_with_logits(
        prediction, target, weight=weights_tensor
    )
    return loss


def binary_stats(preds, targets, threshold=0.5):
    """Calculate metrics for a binary classification task.

    args:
        preds: np.array of shape (batch_size,)
        targets: np.arry of shape (batch_size,)
        threshold: float, threshold for classification

    returns:
        precision: float
        recall: float
        f1: float
    """
    # Compute metrics
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)
    f1 = f1_score(targets, preds)

    return precision, recall, f1


def compute_balanced_acc(fpr, tpr, thresholds):
    """Compute the balanced accuracy for a binary classification task.

    args:
        fpr: list of false positive rates
        tpr: list of true positive rates
        thresholds: threshold values

    return:
        (threshold, balanced_acc)
    """
    balanced_acc = []
    for i in range(len(fpr)):
        balanced_acc.append((1 + tpr[i] - fpr[i]) / 2)
    max_idx = np.argmax(balanced_acc)
    return thresholds[max_idx], balanced_acc[max_idx]


def get_threshold_from_tn(fpr, tpr, thresholds, tn=0.95):
    """Return the threshold that gives the specified True Negative Rate.

    args:
        fpr: list of false positive rates
        tpr: list of true positive rates
        thresholds: threshold values
        tn: float, desired True Negative Rate

    returns:
        (threshold: float, balanced_acc: float)
    """
    tnr = [1 - f for f in fpr]
    idx = np.argmin(np.abs(np.array(tnr) - tn))
    balanced_acc = (1 + tpr[idx] - fpr[idx]) / 2
    return thresholds[idx], balanced_acc


def get_threshold_from_npv(fpr, tpr, thresholds, target_npv=0.95):
    """Return the threshold that gives the specified Negative Predictive Value.

    args:
        fpr: list of false positive rates
        tpr: list of true positive rates
        thresholds: threshold values
        target_npv: float, desired Negative Predictive Value

    returns:
        (threshold: float, balanced_acc: float)
    """
    tnr = [1 - f for f in fpr]
    fnr = [1 - t for t in tpr]
    npv = [t / (t + f) for t, f in zip(tnr, fnr)]
    idx = np.argmin(np.abs(np.array(npv) - target_npv))
    balanced_acc = (1 + tpr[idx] - fpr[idx]) / 2

    return thresholds[idx], balanced_acc


def epoch_end_metrics(
    preds,
    targets,
    epoch_count,
    pirads=None,
    plot_roc=False,
    plot_confusion_matrix=False,
    plot_pirads_breakdown=False,
    mode="train",
    log_dir="./logs",
):
    """Compute metrics at the end of an epoch.

    args:
        preds: list of torch tensors of shape (batch_size, 2)
        targets: list targets of shape (batch_size)
        epoch_count: int, current epoch
        plot_roc: bool, whether to plot ROC curve
        plot_confusion_matrix: bool, whether to plot confusion matrix
        plot_pirads_breakdown: bool, chart the pirads breakdown of TP, FP, TN, FN
        pirads: list of PIRADS scores of shape (batch_size) or None (if not plotting)

    returns:
        auc: (torch tensor) area under ROC curve
        operating_point: (torch tensor) optimal threshold for balanced accuracy
        balanced_acc: (torch tensor) balanced accuracy at optimal threshold
        precision: torch tensor
        recall: torch tensor
        f1: torch tensor
    """
    softmax = nn.Softmax(dim=1)
    auroc = AUROC(task="multiclass", num_classes=2)
    roc = ROC(task="multiclass", num_classes=2)

    # use softmax to get class probabilities
    pred_probs = softmax(preds)

    # compute AUROC
    auc = auroc(pred_probs, targets.type(torch.int))
    fpr, tpr, thresholds = roc(pred_probs.cpu(), targets.type(torch.int).cpu())
    fpr = fpr[1]
    tpr = tpr[1]
    thresholds = thresholds[1]
    # get optimal threshold (wrt npv)
    opt_threshold, balanced_acc = get_threshold_from_npv(fpr, tpr, thresholds, 0.95)

    # get one hot encoding of predictions based on threshold
    #   pred tensor now has shape (batch_size,)
    preds_one_hot = [1 if p[1] > opt_threshold else 0 for p in pred_probs]

    # compute precision/recall
    precision, recall, f1 = binary_stats(preds_one_hot, targets.cpu().numpy())

    # print metrics
    print(mode, "epoch", epoch_count, end=" ")
    print("\tAUC:", auc)

    if epoch_count % 5 == 0:
        if plot_roc:
            # plot ROC curve with torchmetrics
            plot_roc_curve(
                fpr,
                tpr,
                thresholds,
                opt_threshold,
                balanced_acc,
                auc,
                f"{log_dir}/roc/roc_curve_{epoch_count}.png",
            )

        if plot_confusion_matrix:
            # compute and plot confusion matrix at selected threshold
            cm = confusion_matrix(targets.cpu().numpy(), preds_one_hot)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig(f"{log_dir}/cm/confusion_matrix_{epoch_count}.png")

        if plot_pirads_breakdown:
            # plot the PIRADS breakdown of TP, FP, TN, FN
            plot_pirads_cm(
                np.array(preds_one_hot),
                targets.cpu().numpy(),
                np.array(pirads),
                epoch_count,
                save_path=f"{log_dir}/pirads_breakdown/pirads_{epoch_count}.png",
            )

    return auc, opt_threshold, precision, recall, f1
