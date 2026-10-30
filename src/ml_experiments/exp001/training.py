import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_experiments.exp001.models import set_training_mode


def get_device() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")
    return torch.device("mps")


def train_one_epoch(model, train_loader, optimizer, device, desc=""):
    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    progress = tqdm(train_loader, desc=desc, leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == labels).sum().item()
        running_examples += batch_size

        progress.set_postfix(
            loss=f"{running_loss / running_examples:.4f}",
            acc=f"{running_correct / running_examples:.2%}",
        )

    return running_loss / running_examples, running_correct / running_examples


def train_model(model, architecture, train_mode, train_loader, val_loader, optimizer, device, epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    torch.mps.empty_cache()
    peak_memory_bytes = 0
    train_time = 0.0

    for epoch in range(epochs):
        set_training_mode(model, architecture, train_mode)

        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, desc=f"Epoch {epoch + 1}/{epochs}"
        )
        torch.mps.synchronize()
        train_time += time.perf_counter() - epoch_start

        peak_memory_bytes = max(peak_memory_bytes, torch.mps.driver_allocated_memory())

        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_acc:.2%} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={val_metrics['accuracy']:.2%}"
        )

    stats = {
        "training_time_seconds": train_time,
        "peak_mps_memory_gib": peak_memory_bytes / 1024**3,
    }
    return history, stats


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }
