import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

from ml_experiments.exp001.config import IMAGENET_MEAN, IMAGENET_STD, Config

PASCAL_TO_IMAGENET_IDS = {
    "aeroplane": [404, 895],                                   # airliner, warplane
    "bird":      list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147)),
    "boat":      [403, 472, 484, 510, 554, 576, 625, 628, 724, 780, 814, 833, 871, 914],
    "bottle":    [440, 720, 737, 898, 907],
    "bus":       [654, 779, 874],                              # minibus, school bus, trolleybus
    "car":       [407, 436, 468, 511, 609, 627, 656, 661, 751, 817],
    "cat":       list(range(281, 286)),                        # domestic cats only
    "chair":     [423, 559, 765, 857],
    "dog":       list(range(151, 269)),
    "train":     [466, 547, 565, 705, 820, 829],
}


def get_superclass_ids(include_others=False):
    groups = dict(PASCAL_TO_IMAGENET_IDS)
    if include_others:
        used = {i for ids in groups.values() for i in ids}
        groups["others"] = [i for i in range(1000) if i not in used]
    return groups


def build_superclass_matrix(classes, include_others=False):
    assert list(PASCAL_TO_IMAGENET_IDS) == classes, "superclass order must match ImageFolder"
    groups = get_superclass_ids(include_others)
    matrix = torch.zeros(1000, len(groups))
    for superclass_id, imagenet_ids in enumerate(groups.values()):
        matrix[imagenet_ids, superclass_id] = 1.0
    return matrix


def get_transforms():
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomResizedCrop(224),
        T.ColorJitter(1),
        T.ToTensor(),     # Necessary or loader fails: requires tensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224), # Resize doesn't perfectly resize images, so we resize to a slightly larger version and then center crop
        T.ToTensor(),     # Necessary or loader fails: requires tensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transform, val_transform


def get_dataloaders(cfg: Config):
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(cfg.data_root / "train", train_tf)
    val_ds = datasets.ImageFolder(cfg.data_root / "val", val_tf)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_dl, val_dl, train_ds.classes
