# ML Experiments

A collection of small, self-contained machine-learning experiments in PyTorch, where I test ideas, compare methods, and build intuition beyond coursework.

Each experiment lives in its own folder under `src/ml_experiments/` with its own README.

## Experiments

| ID | Experiment | Question |
|---|---|---|
| [exp001](src/ml_experiments/exp001/) | Superclass aggregation | Probability Aggregation vs. Weight Aggregation on a pre-trained ResNet-18 model for superclass classification |

## Quick start

    uv sync
    uv run ml-experiments          # run all experiments
    uv run ml-experiments exp001   # run one

Datasets aren't included; see each experiment's README.