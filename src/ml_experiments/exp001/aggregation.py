import statistics

import torch

from ml_experiments.common.plotting import plot_history
from ml_experiments.common.seed import seed_everything
from ml_experiments.exp001.config import Config
from ml_experiments.exp001.data import build_superclass_matrix, get_dataloaders
from ml_experiments.exp001.evaluation import (
    evaluate_superclass_model,
    print_superclass_metrics,
)
from ml_experiments.exp001.models import (
    ProbabilitySumModel,
    add_others_row,
    build_model,
    freeze_backbone,
    load_pretrained,
)
from ml_experiments.exp001.training import get_device, train_model

# ───────────────────────── Experiments ─────────────────────────

def compare_superclass_methods(cfg, architecture, seed=None):
    device = get_device()
    seed_everything(cfg.seed if seed is None else seed)
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


def compare_trained_heads(cfg, architecture, seed=None):
    device = get_device()
    train_loader, val_loader, classes = get_dataloaders(cfg)

    builders = {
        "weight averaging": lambda: build_model(architecture, classes, head_init="coarse_average"),
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

        seed_everything(cfg.seed if seed is None else seed)
        history, stats = train_model(
            model, architecture, "head_only", train_loader, val_loader, optimizer, device, cfg.epochs
        )

        after = evaluate_superclass_model(model, val_loader, device, len(classes))
        print_superclass_metrics(f"\n{architecture} | {name} | after training", after, classes)
        
        if isinstance(model, ProbabilitySumModel):
            model.superclass_matrix = build_superclass_matrix(classes, include_others=True).to(device)
        else:  # weight averaging (resnet18)
            model.fc = add_others_row(model.fc, architecture)

        after_others = evaluate_superclass_model(model, val_loader, device, len(classes))
        print_superclass_metrics(f"\n{architecture} | {name} + others | after training", after_others, classes)
        
        results[name] = {
            "before": before,
            "after": after,
            "after_others": after_others,
            "history": history,
            "stats": stats,
        }
        
        del model, optimizer
        torch.mps.empty_cache()

    return results

def mean_std(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def print_seed_summary(trained_runs):
    seeds = list(trained_runs)
    methods = trained_runs[seeds[0]].keys()

    print(f"\n{'=' * 60}\nAveraged over seeds {seeds}\n{'=' * 60}")
    for method in methods:
        for split, label in [("after", "trained"), ("after_others", "trained + others")]:
            accs = [trained_runs[s][method][split]["accuracy"] for s in seeds]
            m, sd = mean_std(accs)
            runs = ", ".join(f"{a:.1%}" for a in accs)
            print(f"  {method:17s} {label:17s} {m:.2%} ± {sd:.2%}   ({runs})")

        rates = [trained_runs[s][method]["after_others"]["others_rate"] for s in seeds]
        m, sd = mean_std(rates)
        print(f"  {method:17s} {'-> others':17s} {m:.2%} ± {sd:.2%}")


def average_histories(histories):
    return {key: [statistics.mean(h[key][i] for h in histories) for i in range(len(histories[0][key]))]
            for key in histories[0]}


# ───────────────────────── Run ─────────────────────────

def run(cfg: Config | None = None):
    cfg = cfg or Config()
    results = {}
    SEEDS = [42, 123, 456]

    for architecture in ["resnet18"]:
        result_zeroshot = compare_superclass_methods(cfg, architecture) # No need to seed
        trained_runs = {seed: compare_trained_heads(cfg, architecture, seed=seed) for seed in SEEDS}
        print_seed_summary(trained_runs)

        methods = trained_runs[SEEDS[0]].keys()
        histories = {
            name: average_histories([trained_runs[s][name]["history"] for s in SEEDS])
            for name in methods
        }
        plot_history(
            histories,
            title=f"{architecture} | trained heads (mean of {len(SEEDS)} seeds)",
            save_path=cfg.results_dir / f"{architecture}_trained_history.png",
        )

        results[architecture, "zero-shot"] = result_zeroshot
        results[architecture, "trained"] = trained_runs

    return results


if __name__ == "__main__":
    result = run()
