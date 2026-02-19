import time

import pandas as pd
from mlxtend.frequent_patterns import apriori


def run_apriori(encoded_df: pd.DataFrame, min_support: float) -> tuple[int, float]:
    t_start = time.perf_counter()
    result = apriori(encoded_df, min_support=min_support, use_colnames=True, low_memory=True)
    runtime = time.perf_counter() - t_start
    return len(result), runtime
