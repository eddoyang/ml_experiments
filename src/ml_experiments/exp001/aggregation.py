import torch

from ml_experiments.common.seed import seed_everything
from ml_experiments.exp001.config import Config
from ml_experiments.exp001.data import build_superclass_matrix, get_dataloaders
from ml_experiments.exp001.evaluation import evaluate_superclass_model, print_superclass_metrics
from ml_experiments.exp001.models import ProbabilitySumModel, build_model, freeze_backbone, load_pretrained
from ml_experiments.exp001.training import get_device, train_model

# ───────────────────────── Experiments ─────────────────────────

def compare_superclass_methods(cfg, architecture):
    device = get_device()
    seed_everything(cfg.seed)
    _, val_loader, classes = get_dataloaders(cfg)

    # Method 1: replace the head with one whose weights are the average of each superclass's subclass weights
    weight_avg = build_model(architecture, classes, "pretrained", "coarse_average")

    # Method 2: keep the original head and sum the subclass probabilities
    prob_sum = ProbabilitySumModel(load_pretrained(architecture), build_superclass_matrix(classes))

    # Method 3: Method 1 w/ other's class
    weight_avg_others = build_model(architecture, classes + ["others"], "pretrained", "coarse_average")

    # Method 4: Method 2 w/ other's class
    prob_sum_others = ProbabilitySumModel(load_pretrained(architecture), build_superclass_matrix(classes, include_others=True))

    results = {}
    models = [("weight averaging", weight_avg), 
              ("probability sum", prob_sum), 
              ("weight averaging w/ others", weight_avg_others), 
              ("probability sum w/ others", prob_sum_others)]
    
    for name, model in models:
        metrics = evaluate_superclass_model(model.to(device), val_loader, device, len(classes))
        print_superclass_metrics(f"\n{architecture} | {name}", metrics, classes)
        results[name] = metrics
    return results


def compare_trained_heads(cfg, architecture):
    device = get_device()
    train_loader, val_loader, classes = get_dataloaders(cfg)

    builders = {
        "weight averaging": lambda: build_model(architecture, classes, "coarse_average"),
        "probability sum": lambda: ProbabilitySumModel(
            load_pretrained(architecture), build_superclass_matrix(classes)
        ),
    }

    results = {}
    for name, build in builders.items():
        print(f"\n{'=' * 60}\n{architecture} | {name} | head-only training\n{'=' * 60}")

        model = build()
        freeze_backbone(model, architecture)
        model = model.to(device)

        before = evaluate_superclass_model(model, val_loader, device, len(classes))
        print(f"Before training: accuracy {before['accuracy']:.2%}")

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
        optimizer = torch.optim.Adam(trainable, lr=cfg.lr)

        seed_everything(cfg.seed)
        history, stats = train_model(
            model, architecture, "head_only", train_loader, val_loader, optimizer, device, cfg.epochs
        )

        after = evaluate_superclass_model(model, val_loader, device, len(classes))
        print_superclass_metrics(f"\n{architecture} | {name} | after training", after, classes)

        results[name] = {"before": before, "after": after, "history": history, "stats": stats}

        del model, optimizer
        torch.mps.empty_cache()

    return results


# ───────────────────────── Run ─────────────────────────

def run(cfg: Config | None = None):
    cfg = cfg or Config()
    results = {}
    for architecture in ["resnet18"]:
        results[architecture, "zero-shot"] = compare_superclass_methods(cfg, architecture)
        results[architecture, "trained"] = compare_trained_heads(cfg, architecture)
    return results


if __name__ == "__main__":
    result = run()
