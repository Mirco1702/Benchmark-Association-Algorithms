from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    run_mlxtend_apriori: bool = True
    run_mlxtend_fpgrowth: bool = True
    run_pyspark_fpgrowth: bool = True

    dataset_type: str = "fp_friendly"  # "fp_hard" oder "fp_friendly"
    n_sizes: int = 2
    min_transactions: int = 1_000
    max_transactions: int = 2_000
    n_items: int = 5_000
    items_per_transaction: int = 8
    min_support: float = 0.05
    random_state: int = 42

    save_datasets: bool = False
    dataset_dir: Path = Path("benchmark_datasets")

    spark_app_name: str = "FPGrowth_Benchmark"
    spark_master: str = "local[*]"
    spark_driver_memory: str = "4g"


CONFIG = BenchmarkConfig()


if __name__ == "__main__":
    from auto_plotting_benchmark import main

    main(CONFIG)
