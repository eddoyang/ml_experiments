from torchvision.models import ResNet18_Weights, VGG16_Weights, resnet18, vgg16


def get_head(model, architecture=None):
    if hasattr(model, "head"):
        return model.head

    if architecture == "resnet18" or hasattr(model, "fc"):
        return model.fc

    if architecture == "vgg16" or hasattr(model, "classifier"):
        return model.classifier[-1]

    raise ValueError(f"Don't know where the head is for {architecture or type(model).__name__}")


def freeze_backbone(model, architecture=None):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in get_head(model, architecture).parameters():
        parameter.requires_grad = True


def unfreeze_everything(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def set_training_mode(model, architecture, train_mode):
    if train_mode == "full":
        model.train()
        return
    if train_mode == "head_only":
        model.eval()  # keep BatchNorm statistics and dropout fixed in the frozen backbone
        get_head(model, architecture).train()
        return
    raise ValueError(train_mode)


def load_pretrained(architecture):
    if architecture == "resnet18":
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if architecture == "vgg16":
        return vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    raise ValueError(architecture)