import matplotlib.pyplot as plt
import pandas as pd


def build_runtime_pivot(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.pivot(index="n_transactions", columns="algo", values="time")


def print_results_table(pivot_df: pd.DataFrame) -> None:
    print("\n===== Ergebnisse =====")
    print(pivot_df)


def plot_runtime_curves(
    pivot_df: pd.DataFrame,
    dataset_type: str,
    min_support: float,
) -> None:
    plt.figure(figsize=(10, 6))

    markers = ["o", "s", "^", "d", "x"]
    for idx, algo_name in enumerate(pivot_df.columns):
        plt.plot(
            pivot_df.index,
            pivot_df[algo_name],
            marker=markers[idx % len(markers)],
            label=algo_name,
        )

    plt.title(f"Benchmark: Frequent Pattern Mining\nTyp: {dataset_type} | MinSupp: {min_support}")
    plt.xlabel("Anzahl Transaktionen")
    plt.ylabel("Laufzeit (Sekunden)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
