from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

"""
Standard ImageNet transforms for training and validation. 
The training transforms include data augmentation, while 
the validation transforms are more straightforward.
"""
def get_transforms(mean, std):
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.RandomResizedCrop(224),
        T.ColorJitter(1),
        T.ToTensor(),     # Necessary or loader fails: requires tensor
        T.Normalize(mean=mean, std=std),
    ])

    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224), # Resize doesn't perfectly resize images, so we resize to a slightly larger version and then center crop
        T.ToTensor(),     # Necessary or loader fails: requires tensor
        T.Normalize(mean=mean, std=std),
    ])
    
    return train_transform, val_transform


def get_dataloaders(cfg: dict):
    train_tf, val_tf = get_transforms(cfg.mean, cfg.std)
    train_ds = datasets.ImageFolder(cfg.data_root / "train", train_tf)
    val_ds = datasets.ImageFolder(cfg.data_root / "val", val_tf)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)
    return train_dl, val_dl, train_ds.classes