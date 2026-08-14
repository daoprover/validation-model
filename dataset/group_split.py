"""Leakage-safe splitting for address graphs that may share transactions."""

import random
from collections import Counter, defaultdict


def grouped_train_validation_split(indices, labels, transaction_sets, validation_fraction, seed):
    """Split graphs while keeping every shared transaction in one partition."""
    if not (len(indices) == len(labels) == len(transaction_sets)):
        raise ValueError("indices, labels, and transaction_sets must have equal lengths")
    if not indices:
        raise ValueError("Cannot split an empty dataset")
    if validation_fraction == 0 or len(indices) == 1:
        return list(indices), []
    if any(not transaction_ids for transaction_ids in transaction_sets):
        raise ValueError(
            "Every graph must contain explicit transaction nodes; regenerate legacy graphs first"
        )

    parent = list(range(len(indices)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    transaction_owner = {}
    for position, transaction_ids in enumerate(transaction_sets):
        for transaction_id in transaction_ids:
            if transaction_id in transaction_owner:
                union(position, transaction_owner[transaction_id])
            else:
                transaction_owner[transaction_id] = position

    groups = defaultdict(list)
    for position in range(len(indices)):
        groups[find(position)].append(position)
    grouped_positions = list(groups.values())
    if len(grouped_positions) < 2:
        raise ValueError(
            "All graphs are connected by shared transactions; a leakage-free validation split is impossible"
        )

    target_validation_size = max(1, min(len(indices) - 1, round(len(indices) * validation_fraction)))
    overall_counts = Counter(labels)
    rng = random.Random(seed)
    best = None

    for _ in range(max(128, len(grouped_positions) * 8)):
        candidate_groups = list(grouped_positions)
        rng.shuffle(candidate_groups)
        validation_positions = []
        selected_groups = []
        for group in candidate_groups:
            if len(validation_positions) >= target_validation_size:
                break
            validation_positions.extend(group)
            selected_groups.append(group)
        if len(validation_positions) == len(indices):
            selected_groups.pop()
            validation_positions = [
                position for group in selected_groups for position in group
            ]
        if not validation_positions:
            continue

        validation_set = set(validation_positions)
        training_positions = [position for position in range(len(indices)) if position not in validation_set]
        validation_counts = Counter(labels[position] for position in validation_positions)
        training_counts = Counter(labels[position] for position in training_positions)

        size_error = abs(len(validation_positions) - target_validation_size) / len(indices)
        distribution_error = sum(
            abs(
                validation_counts[label] / len(validation_positions)
                - overall_counts[label] / len(indices)
            )
            for label in overall_counts
        )
        missing_class_penalty = sum(
            2.0
            for label, count in overall_counts.items()
            if count >= 2 and (validation_counts[label] == 0 or training_counts[label] == 0)
        )
        score = size_error + distribution_error + missing_class_penalty
        if best is None or score < best[0]:
            best = (score, training_positions, validation_positions)

    if best is None:
        raise ValueError("Could not construct a non-empty grouped validation split")

    _, training_positions, validation_positions = best
    return (
        [indices[position] for position in training_positions],
        [indices[position] for position in validation_positions],
    )


def transaction_overlap(left_sets, right_sets):
    left = set().union(*left_sets) if left_sets else set()
    right = set().union(*right_sets) if right_sets else set()
    return left & right
