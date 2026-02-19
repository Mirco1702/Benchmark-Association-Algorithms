import time

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from benchmark_apriori import run_apriori
from benchmark_fpgrowth import run_fpgrowth
from benchmark_pyspark import (
    init_spark_session,
    prepare_spark_dataframe,
    run_pyspark_fpgrowth,
    stop_spark_session,
)
from configuration import CONFIG, BenchmarkConfig
from dataset_generation import (
    build_item_pool,
    generate_transactions,
    save_transactions_csv,
    validate_generation_inputs,
)
from plot import build_runtime_pivot, plot_runtime_curves, print_results_table


def run_benchmark(config: BenchmarkConfig) -> pd.DataFrame:
    validate_generation_inputs(config.items_per_transaction, config.n_items)

    transaction_sizes = np.linspace(
        config.min_transactions,
        config.max_transactions,
        config.n_sizes,
        dtype=int,
    )
    rng = np.random.default_rng(config.random_state)
    items = build_item_pool(config.n_items)
    results: list[dict[str, object]] = []

    spark = init_spark_session(
        enabled=config.run_pyspark_fpgrowth,
        app_name=config.spark_app_name,
        master=config.spark_master,
        driver_memory=config.spark_driver_memory,
    )

    try:
        for n_transactions in transaction_sizes:
            print("=" * 60)
            print(f"Datensatz: {n_transactions} Transaktionen ({config.dataset_type})")

            transactions = generate_transactions(
                dataset_type=config.dataset_type,
                n_transactions=int(n_transactions),
                rng=rng,
                items=items,
                n_items=config.n_items,
                items_per_transaction=config.items_per_transaction,
            )

            if config.save_datasets:
                path = save_transactions_csv(
                    transactions=transactions,
                    output_dir=config.dataset_dir,
                    dataset_type=config.dataset_type,
                    n_transactions=int(n_transactions),
                )
                print(f"  [Dataset] gespeichert unter: {path}")

            encoded_df = None
            mlxtend_prep_time = 0.0
            if config.run_mlxtend_apriori or config.run_mlxtend_fpgrowth:
                # gemeinsames One-Hot-Encoding fuer beide MLxtend-Algos
                print("  [MLxtend] One-Hot-Encoding...")
                t_start = time.perf_counter()

                te = TransactionEncoder()
                te_array = te.fit(transactions).transform(transactions)
                encoded_df = pd.DataFrame(te_array, columns=te.columns_)

                mlxtend_prep_time = time.perf_counter() - t_start
                print(
                    f"  [MLxtend] Encoding fertig ({mlxtend_prep_time:.3f} s). "
                    f"Shape: {encoded_df.shape}"
                )

            if config.run_mlxtend_apriori and encoded_df is not None:
                print("  -> Starte MLxtend Apriori...")
                itemset_count, runtime = run_apriori(
                    encoded_df=encoded_df,
                    min_support=config.min_support,
                )
                print(f"     Fertig: {itemset_count} Itemsets in {runtime:.3f} s")
                results.append(
                    {
                        "n_transactions": int(n_transactions),
                        "algo": "MLxtend Apriori",
                        "time": runtime,
                        "prep_time": mlxtend_prep_time,
                    }
                )

            if config.run_mlxtend_fpgrowth and encoded_df is not None:
                print("  -> Starte MLxtend FP-Growth...")
                itemset_count, runtime = run_fpgrowth(
                    encoded_df=encoded_df,
                    min_support=config.min_support,
                )
                print(f"     Fertig: {itemset_count} Itemsets in {runtime:.3f} s")
                results.append(
                    {
                        "n_transactions": int(n_transactions),
                        "algo": "MLxtend FP-Growth",
                        "time": runtime,
                        "prep_time": mlxtend_prep_time,
                    }
                )

            if config.run_pyspark_fpgrowth and spark is not None:
                print("  [PySpark] DataFrame erzeugen...")
                spark_df, spark_prep_time = prepare_spark_dataframe(
                    spark=spark,
                    transactions=transactions,
                )

                print("  -> Starte PySpark FP-Growth...")
                itemset_count, runtime = run_pyspark_fpgrowth(
                    spark_df=spark_df,
                    min_support=config.min_support,
                )
                print(f"     Fertig: {itemset_count} Itemsets in {runtime:.3f} s")

                results.append(
                    {
                        "n_transactions": int(n_transactions),
                        "algo": "PySpark FP-Growth",
                        "time": runtime,
                        # Spark-DataFrame-Aufbau separat halten
                        "prep_time": spark_prep_time,
                    }
                )
                spark_df.unpersist()
    finally:
        stop_spark_session(spark)

    return pd.DataFrame(results)


def main(config: BenchmarkConfig = CONFIG) -> None:
    results_df = run_benchmark(config)
    if results_df.empty:
        print("\nKeine Ergebnisse, da alle Algorithmen deaktiviert waren.")
        return

    pivot_df = build_runtime_pivot(results_df)
    print_results_table(pivot_df)
    plot_runtime_curves(
        pivot_df=pivot_df,
        dataset_type=config.dataset_type,
        min_support=config.min_support,
    )


if __name__ == "__main__":
    main()
