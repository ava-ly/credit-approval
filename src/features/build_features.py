from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

def create_features(df: DataFrame) -> DataFrame:
    """
    Creates new features based on EDA findings
    - AGE: age of applicants in years.
    - YEARS_EMPLOYED: Years of employment.
    - IS_UNEMPLOYED: binary flag for employment.
    - INCOME_PER_PERSON: Income per family member.
    """
    featured_df = (
        df.withColumn("AGE", -F.col("DAYS_BIRTH") / 365.0)
            .withColumn("IS_UNEMPLOYED", 
                        F.when(F.col("DAYS_EMPLOYED") > 0, 1)
                        .otherwise(0))
            .withColumn("YEARS_EMPLOYED",
                        F.when(F.col("DAYS_EMPLOYED") > 0, 0)
                        .otherwise(-F.col("DAYS_EMPLOYED") / 365.0))
            .withColumn("INCOME_PER_PERSON", 
                        F.col("AMT_INCOME_TOTAL") / F.col("CNT_FAM_MEMBERS"))
    )
    return featured_df

if __name__ == '__main__':
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    PROCESSED_DATA_PATH = "data/processed/primary_dataset"
    FEATURED_DATA_PATH = "data/processed/featured_dataset"

    print(f"Loading data from {PROCESSED_DATA_PATH}")
    primary_df = spark.read.parquet(PROCESSED_DATA_PATH)

    print("Creating new features...")
    featured_df = create_features(primary_df)

    print(f"Saving featured data to {FEATURED_DATA_PATH}")
    featured_df.write.mode("overwrite").parquet(FEATURED_DATA_PATH)

    print("Feature engineering complete.")
    spark.stop()