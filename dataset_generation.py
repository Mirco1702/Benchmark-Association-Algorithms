from pathlib import Path

import numpy as np
import pandas as pd


def validate_generation_inputs(items_per_transaction: int, n_items: int) -> None:
    if items_per_transaction > n_items:
        raise ValueError("ITEMS_PER_TRANSACTION darf nicht groesser als N_ITEMS sein.")


def build_item_pool(n_items: int) -> np.ndarray:
    return np.array([f"Item_{i}" for i in range(1, n_items + 1)])


def generate_dataset_fp_hard(
    n_transactions: int,
    rng: np.random.Generator,
    items: np.ndarray,
    n_items: int,
    items_per_transaction: int,
) -> list[list[str]]:
    records: list[list[str]] = []
    for _ in range(n_transactions):
        chosen_indices = rng.choice(n_items, size=items_per_transaction, replace=False)
        chosen_items = items[chosen_indices]
        records.append(chosen_items.tolist())
    return records


def generate_dataset_fp_friendly(
    n_transactions: int,
    rng: np.random.Generator,
    items: np.ndarray,
    n_items: int,
    items_per_transaction: int,
) -> list[list[str]]:
    cluster_count = min(10, n_transactions)
    base_patterns = []
    for _ in range(cluster_count):
        # wiederkehrende Kernmuster
        chosen_indices = rng.choice(n_items, size=items_per_transaction, replace=False)
        base_patterns.append(items[chosen_indices])

    records: list[list[str]] = []
    for _ in range(n_transactions):
        base = base_patterns[rng.integers(cluster_count)]
        trans_items = np.array(base, copy=True)

        if items_per_transaction < n_items and rng.random() < 0.3:
            # leichtes Rauschen: ein Item tauschen
            pos = rng.integers(items_per_transaction)
            while True:
                cand_idx = rng.integers(n_items)
                cand_item = items[cand_idx]
                if cand_item not in trans_items:
                    trans_items[pos] = cand_item
                    break
        records.append(trans_items.tolist())

    return records


def generate_transactions(
    dataset_type: str,
    n_transactions: int,
    rng: np.random.Generator,
    items: np.ndarray,
    n_items: int,
    items_per_transaction: int,
) -> list[list[str]]:
    if dataset_type == "fp_hard":
        return generate_dataset_fp_hard(
            n_transactions=n_transactions,
            rng=rng,
            items=items,
            n_items=n_items,
            items_per_transaction=items_per_transaction,
        )

    if dataset_type == "fp_friendly":
        return generate_dataset_fp_friendly(
            n_transactions=n_transactions,
            rng=rng,
            items=items,
            n_items=n_items,
            items_per_transaction=items_per_transaction,
        )

    raise ValueError("dataset_type muss 'fp_hard' oder 'fp_friendly' sein.")


def save_transactions_csv(
    transactions: list[list[str]],
    output_dir: Path,
    dataset_type: str,
    n_transactions: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # listenbasiert -> breite CSV mit item_1..item_n
    records = []
    for tid, transaction in enumerate(transactions, start=1):
        row = {"tid": tid}
        for idx, item in enumerate(transaction, start=1):
            row[f"item_{idx}"] = item
        records.append(row)

    output_path = output_dir / f"{dataset_type}_{n_transactions}.csv"
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    return output_path
