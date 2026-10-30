import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    VGG16_Weights,
    resnet18,
    vgg16,
)

from ml_experiments.exp001.config import Config
from ml_experiments.exp001.data import get_superclass_ids


def make_linear_head(in_features, out_features, seed=Config.seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(in_features, out_features)
    return head


def build_coarse_head(pretrained_head, class_names, initialization):
    new_head = make_linear_head(pretrained_head.in_features, len(class_names))

    if initialization == "random":
      return new_head

    if initialization == "zero":
      with torch.no_grad():
        new_head.weight.zero_()
        new_head.bias.zero_()
      return new_head

    if initialization == "coarse_average":
      with torch.no_grad():
        for coarse_id, coarse_name in enumerate(class_names):
            fine_ids = get_superclass_ids(include_others="others" in class_names)[coarse_name]
            fine_weights = pretrained_head.weight[fine_ids]
            fine_biases = pretrained_head.bias[fine_ids]

            new_head.weight[coarse_id].copy_(fine_weights.mean(dim=0))
            new_head.bias[coarse_id].copy_(fine_biases.mean())
      return new_head

    raise ValueError(
        f"Unknown initialization: {initialization}"
    )

def add_others_row(trained_head, architecture):
    pretrained = get_head(load_pretrained(architecture), architecture)
    others_ids = get_superclass_ids(include_others=True)["others"]

    new_head = nn.Linear(trained_head.in_features, trained_head.out_features + 1)
    with torch.no_grad():
        new_head.weight[:-1].copy_(trained_head.weight)
        new_head.bias[:-1].copy_(trained_head.bias)
        new_head.weight[-1].copy_(pretrained.weight[others_ids].mean(dim=0))
        new_head.bias[-1].copy_(pretrained.bias[others_ids].mean())
    return new_head.to(trained_head.weight.device)


def load_pretrained(architecture):
    if architecture == "resnet18":
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if architecture == "vgg16":
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    raise ValueError(architecture)


def build_model(architecture, classes, pretrained="pretrained", head_init="coarse_average"):
    
    if architecture == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = build_coarse_head(model.fc, classes, head_init)
        return model

    if architecture == "vgg16":
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.classifier[-1] = build_coarse_head(model.classifier[-1], classes, head_init)
        return model


class ProbabilitySumModel(nn.Module):
    def __init__(self, base_model, superclass_matrix):
        super().__init__()
        self.base = base_model
        self.register_buffer("superclass_matrix", superclass_matrix)

    def forward(self, x):
        probs = self.base(x).softmax(dim=1)              # (batch, 1000)
        super_probs = probs @ self.superclass_matrix     # (batch, 10) — sum per superclass
        return torch.log(super_probs.clamp_min(1e-12))


def get_head(model, architecture):
    if isinstance(model, ProbabilitySumModel):
        model = model.base
    
    if architecture == "resnet18":
        return model.fc

    if architecture == "vgg16":
        return model.classifier[-1]

    raise ValueError(architecture)


def freeze_backbone(
    model,
    architecture,
):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in get_head(
        model,
        architecture,
    ).parameters():
        parameter.requires_grad = True


def unfreeze_everything(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def set_training_mode(
    model,
    architecture,
    train_mode,
):
    if train_mode == "full":
        model.train()
        return

    if train_mode == "head_only":

        model.eval()

        get_head(model, architecture).train()

        return

    raise ValueError(train_mode)
