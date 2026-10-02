"""
This module provides methods to select and generate the correct PyTorch Lightning
ResNet3D implementation based on config settings.
Early fusion version — uses lakshita_earlyfusion.
"""
import torch

from src.models.ResNet3D.lakshita_earlyfusion import Base3DResNet
from src.models.ResNet3D.branched_3Dresnet import DualSeriesModel
from src.models.ResNet3D.branched_3Dresnet import QuadSeriesModel
from src.models.ResNet3D.branched_3Dresnet import TriSeriesModel


def load_med3d_weights(model, checkpoint):
    for param_tensor in model.state_dict():
        new_param_tensor = "module." + param_tensor.replace(
            "resnet_branch1.", ""
        ).replace("resnet_branch2.", "").replace("resnet_branch3.", "")
        if new_param_tensor in checkpoint["state_dict"]:
            if (
                model.state_dict()[param_tensor].size()
                == checkpoint["state_dict"][new_param_tensor].size()
            ):
                with torch.no_grad():
                    model.state_dict()[param_tensor].copy_(
                        checkpoint["state_dict"][new_param_tensor]
                    )
    return model


def load_saved_resnet3d_weights(model, checkpoint, exclude_layers=None, disable_gradient=False):
    if exclude_layers is None:
        exclude_layers = []

    for param_tensor in model.state_dict():
        if param_tensor in checkpoint["state_dict"]:
            if not any(param_tensor.startswith(layer) for layer in exclude_layers):
                if (
                    model.state_dict()[param_tensor].size()
                    == checkpoint["state_dict"][param_tensor].size()
                ):
                    if disable_gradient:
                        with torch.no_grad():
                            model.state_dict()[param_tensor].copy_(
                                checkpoint["state_dict"][param_tensor]
                            )
                    else:
                        model.state_dict()[param_tensor].copy_(
                            checkpoint["state_dict"][param_tensor]
                        )

    return model


def generate_resnet3d(config):
    if len(config["data"]["series"]) == 4:
        model = QuadSeriesModel(config)
    elif len(config["data"]["series"]) == 3:
        model = TriSeriesModel(config)
    elif len(config["data"]["series"]) == 2:
        model = DualSeriesModel(config)
    elif len(config["data"]["series"]) == 1:
        model = Base3DResNet(config)
    else:
        raise ValueError(
            f"Unsupported number of series ({len(config['data']['series'])})."
        )

    init_weights = config["model_weights"]["load_weights"]
    if init_weights:
        print("loading pretrained weights...")

        model_ckpt = config["model_weights"].get("model_ckpt")
        t2_ckpt = config["model_weights"].get("t2_model_ckpt")
        dwi_ckpt = config["model_weights"].get("dwi_model_ckpt")
        disable_gradient = config["model_weights"]["disable_gradient"]

        if model_ckpt:
            checkpoint = torch.load(
                model_ckpt,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            model = load_saved_resnet3d_weights(
                model,
                checkpoint,
                disable_gradient=disable_gradient
            )
        elif model.__class__.__name__ == "TriSeriesModel":
            t2_checkpoint = torch.load(
                t2_ckpt,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            model.resnet_single_branch = load_saved_resnet3d_weights(
                model.resnet_single_branch,
                t2_checkpoint,
                exclude_layers=['fc'],
                disable_gradient=disable_gradient
            )

            if model.stack_adc_b1500:
                dwi_checkpoint = torch.load(
                    dwi_ckpt,
                    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                model.resnet_dual_branch1 = load_saved_resnet3d_weights(
                    model.resnet_dual_branch1,
                    dwi_checkpoint,
                    exclude_layers=['fc'],
                    disable_gradient=disable_gradient
                )
            else:
                raise NotImplementedError
        else:
            print("Pretrained weights are not supported for this model type.")

    else:
        print("training from scratch...")

    return model
