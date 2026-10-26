import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def evaluate_superclass_model(model, loader, device, num_classes):
    model.eval()
    total_loss, total_correct, total, total_others = 0.0, 0, 0, 0
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    class_to_others = torch.zeros(num_classes)  # images of each class predicted as "others"
    has_others = False

    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        has_others = logits.size(1) > num_classes  # model has an extra "others" output

        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (pred == y).sum().item()
        total += y.size(0)
        class_total += torch.bincount(y, minlength=num_classes).cpu()
        class_correct += torch.bincount(y[pred == y], minlength=num_classes).cpu()

        # "others" is any output index past the real superclasses
        y_cpu, is_others = y.cpu(), (pred >= num_classes).cpu()
        total_others += is_others.sum().item()
        class_to_others += torch.bincount(y_cpu[is_others], minlength=num_classes)

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "per_class_accuracy": (class_correct / class_total).tolist(),
        "has_others": has_others,
        "others_rate": total_others / total,
        "per_class_others_rate": (class_to_others / class_total).tolist(),
    }


def print_superclass_metrics(title, metrics, class_names):
    print(title)
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")

    if not metrics["has_others"]:
        for name, acc in zip(class_names, metrics["per_class_accuracy"]):
            print(f"    {name:12s}: {acc:.2%}")
        return

    print(f"  Predicted as others: {metrics['others_rate']:.2%} of all images")
    print(f"    {'class':12s}  {'accuracy':>8s}  {'-> others':>9s}")
    for name, acc, oth in zip(
        class_names, metrics["per_class_accuracy"], metrics["per_class_others_rate"]
    ):
        print(f"    {name:12s}  {acc:8.2%}  {oth:9.2%}")
    print(f"    {'others':12s}  {'n/a':>8s}  {metrics['others_rate']:9.2%}")
