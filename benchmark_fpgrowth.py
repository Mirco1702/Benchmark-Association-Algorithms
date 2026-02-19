import time

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth


def run_fpgrowth(encoded_df: pd.DataFrame, min_support: float) -> tuple[int, float]:
    t_start = time.perf_counter()
    result = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
    runtime = time.perf_counter() - t_start
    return len(result), runtime
