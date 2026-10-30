from pathlib import Path

import matplotlib.pyplot as plt

SPLIT_STYLES = {"train": "-", "val": "--", "test": ":"}


def _split_key(key):
    # "train_loss" -> ("train", "loss"); keys without a known split prefix are their own metric
    split, _, metric = key.partition("_")
    if split in SPLIT_STYLES and metric:
        return split, metric
    return None, key

def plot_history(histories, title=None, metrics=None, save_path=None, show=True):
    if all(isinstance(v, (list, tuple)) for v in histories.values()):
        histories = {None: histories}

    if metrics is None:
        metrics = []
        for history in histories.values():
            for key in history:
                metric = _split_key(key)[1]
                if metric not in metrics:
                    metrics.append(metric)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        for key, values in history.items():
            split, metric = _split_key(key)
            if metric not in metrics:
                continue

            label = " ".join(part for part in (run_name, split) if part)
            epochs = range(1, len(values) + 1)
            axes[metrics.index(metric)].plot(
                epochs, values, SPLIT_STYLES.get(split, "-"), color=color, label=label or None
            )

    for ax, metric in zip(axes, metrics):
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize="small")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
