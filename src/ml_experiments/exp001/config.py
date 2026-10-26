from dataclasses import dataclass, field
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class Config:
    data_root: Path = PROJECT_ROOT / "data" / "mini_imagenet_pascal10"
    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    models: list[str] = field(default_factory=lambda: ["resnet18", "resnet50", "vgg16"])
    aggregation: str = "mean"          # your new scenario knob
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
