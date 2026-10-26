from ml_experiments.exp001 import aggregation as exp001

EXPERIMENTS = {
    "exp001": exp001.run,
}


def main():
    import sys

    selected = sys.argv[1:] or list(EXPERIMENTS)
    for name in selected:
        print(f"=== {name} ===")
        EXPERIMENTS[name]()
