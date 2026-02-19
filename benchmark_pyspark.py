import os
import sys
import time

from pyspark.ml.fpm import FPGrowth as SparkFPGrowth
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


def init_spark_session(
    enabled: bool,
    app_name: str,
    master: str,
    driver_memory: str,
) -> SparkSession | None:
    if not enabled:
        print("Spark Session uebersprungen (run_pyspark_fpgrowth=False).")
        return None

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    print(f"Genutzter Python Pfad: {sys.executable}")
    print("Starte Spark Session...")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("Spark Session aktiv.\n")
    return spark


def prepare_spark_dataframe(
    spark: SparkSession,
    transactions: list[list[str]],
) -> tuple[DataFrame, float]:
    t_start = time.perf_counter()

    data_for_spark = [(transaction,) for transaction in transactions]
    schema = StructType([StructField("items", ArrayType(StringType()), True)])
    spark_df = spark.createDataFrame(data_for_spark, schema=schema)

    # Materialisieren, damit der Setup-Aufwand als prep_time erfasst wird.
    spark_df.cache()
    spark_df.count()

    prep_time = time.perf_counter() - t_start
    return spark_df, prep_time


def run_pyspark_fpgrowth(spark_df: DataFrame, min_support: float) -> tuple[int, float]:
    fp_model = SparkFPGrowth(itemsCol="items", minSupport=min_support)

    t_start = time.perf_counter()
    model = fp_model.fit(spark_df)
    itemset_count = model.freqItemsets.count()
    runtime = time.perf_counter() - t_start

    return itemset_count, runtime


def stop_spark_session(spark: SparkSession | None) -> None:
    if spark is not None:
        spark.stop()
