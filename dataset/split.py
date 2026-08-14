import argparse
import shutil
from pathlib import Path

try:
    from dataset.group_split import grouped_train_validation_split
except ModuleNotFoundError:
    from group_split import grouped_train_validation_split


def split_dataset(source, destination, seed=42, move=False):
    """Create a deterministic 70/15/15 train/validation/test split."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise ValueError(f"Dataset directory does not exist: {source}")

    import networkx as nx

    files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix == ".gexf")
    labels = []
    transaction_sets = []
    for path in files:
        graph = nx.read_gexf(path)
        raw_label = graph.graph.get("name")
        labels.append("anomaly" if raw_label != "white" else "white")
        transaction_sets.append({
            str(node)
            for node, attrs in graph.nodes(data=True)
            if str(node).startswith("transaction:") or str(attrs.get("node_type")) == "1"
        })

    indices = list(range(len(files)))
    train_indices, holdout_indices = grouped_train_validation_split(
        indices, labels, transaction_sets, validation_fraction=0.30, seed=seed
    )
    valid_indices, test_indices = grouped_train_validation_split(
        holdout_indices,
        [labels[index] for index in holdout_indices],
        [transaction_sets[index] for index in holdout_indices],
        validation_fraction=0.50,
        seed=seed + 1,
    )
    splits = {
        "train": [files[index] for index in train_indices],
        "valid": [files[index] for index in valid_indices],
        "test": [files[index] for index in test_indices],
    }

    operation = shutil.move if move else shutil.copy2
    for split_name, split_files in splits.items():
        split_directory = destination / split_name
        split_directory.mkdir(parents=True, exist_ok=True)
        for source_file in split_files:
            operation(source_file, split_directory / source_file.name)
    return {name: len(items) for name, items in splits.items()}


def main():
    parser = argparse.ArgumentParser(description="Split GEXF graphs into train/valid/test directories")
    parser.add_argument("source", help="Directory containing the source .gexf files")
    parser.add_argument("destination", help="Parent directory for the output splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move instead of copying source files")
    args = parser.parse_args()
    counts = split_dataset(args.source, args.destination, seed=args.seed, move=args.move)
    print(" ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
