import statistics


def mean_std(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def average_histories(histories):
    """Average a list of history dicts epoch by epoch."""
    return {key: [statistics.mean(h[key][i] for h in histories) for i in range(len(histories[0][key]))]
            for key in histories[0]}