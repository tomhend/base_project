import torch
import torch.nn as nn
from models.architectures import resnet3d


class MedicalNet10(torch.nn.Module):
    def __init__(
        self,
        pretrain: bool,
        input_shape: list[int],
        n_classes: int,
        weights_path: str = None,
    ) -> None:
        super().__init__()
        depth, height, width = input_shape
        model = resnet3d.resnet10(
            sample_input_D=depth,
            sample_input_H=height,
            sample_input_W=width,
            num_seg_classes=n_classes,
        )
        if pretrain:
            net_dict = model.state_dict()
            pretrain = torch.load(weights_path, map_location=torch.device("cpu"))
            pretrain_dict = {
                k[7:]: v
                for k, v in pretrain["state_dict"].items()
                if k[7:] in net_dict.keys()
            }
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

        self.medicalnet = nn.Sequential(
            *list(model.children())[:-1], nn.ReLU(), nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # for param in self.medicalnet.parameters():
        #    param.requires_grad = False
        # for param in list(self.medicalnet.children())[0].parameters():
        #    param.requires_grad = True
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x) -> torch.Tensor:
        mn_out = self.medicalnet(x)
        out = self.fc1(self.flatten(mn_out))
        return out


class MedicalNet50(torch.nn.Module):
    def __init__(
        self,
        pretrain: bool,
        input_shape: list[int],
        n_classes: int,
        weights_path: str = None,
    ) -> None:
        super().__init__()
        depth, height, width = input_shape
        model = resnet3d.resnet50(
            sample_input_D=depth,
            sample_input_H=height,
            sample_input_W=width,
            num_seg_classes=n_classes,
        )
        if pretrain:
            net_dict = model.state_dict()
            pretrain = torch.load(weights_path, map_location=torch.device("cpu"))
            pretrain_dict = {
                k[7:]: v
                for k, v in pretrain["state_dict"].items()
                if k[7:] in net_dict.keys()
            }
            net_dict.update(pretrain_dict)
            model.load_state_dict(net_dict)

        self.medicalnet = nn.Sequential(
            *list(model.children())[:-1], nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        # for param in self.medicalnet.parameters():
        #    param.requires_grad = False

        self.fc1 = nn.Linear(2048, 1)

    def forward(self, x) -> torch.Tensor:
        mn_out = self.medicalnet(x)
        out = self.fc1(torch.flatten(mn_out, start_dim=1))

        return out
