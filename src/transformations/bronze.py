from collections.abc import Collection, Mapping

from pyspark import pipelines as dp
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    current_timestamp,
    lit,
    timestamp_millis,
    to_timestamp,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

catalog = spark.conf.get("retail.catalog")
schema = spark.conf.get("retail.bronze_schema")
bronze_table = f"{catalog}.{schema}.retail_bronze_table"
raw_data_path = spark.conf.get("retail.raw_data_root_path")

bronze_table_schema = StructType(
    [
        # Canonical business columns...
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_timestamp", TimestampType(), True),
        StructField("product_sku", StringType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("quantity", LongType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("payment_method", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("store_location", StringType(), True),
        # Data-quality and ingestion metadata
        StructField("_source_retailer", StringType(), False),
        StructField("_rescued_data", StringType(), True),
        StructField("_source_file", StringType(), False),
        StructField("_source_file_modified_at", TimestampType(), True),
        StructField("_ingested_at", TimestampType(), False),
    ]
)

dp.create_streaming_table(
    name=bronze_table,
    comment="Raw retail orders ingested incrementally from JSON and delimited landing files",
    schema=bronze_table_schema,
)

metadata_columns = [
    "_rescued_data",
    "_source_file",
    "_source_file_modified_at",
    "_ingested_at",
]

retailer_a_column_mapping = {
    "RECEIPT_NUM": "order_id",
    "CLIENT_REF": "customer_id",
    "TXN_DATETIME": "order_timestamp",
    "PRODUCT_ID": "product_sku",
    "UNIT_COST": "unit_price",
    "ORDER_QTY": "quantity",
    "FINAL_CHARGE": "total_amount",
    "PAY_TYPE": "payment_method",
    "DISPATCH_FLAG": "order_status",
    "STORE_LOCATION": "store_location",
}

retailer_b_column_mapping = {
    "order_id": "order_id",
    "customer_id": "customer_id",
    "order_timestamp": "order_timestamp",
    "product_sku": "product_sku",
    "unit_price": "unit_price",
    "quantity": "quantity",
    "total_amount": "total_amount",
    "payment_method": "payment_method",
    "order_status": "order_status",
    "store_city": "store_location",
}

retailer_c_column_mapping = {
    "transactionId": "order_id",
    "userId": "customer_id",
    "eventTimestampMs": "order_timestamp",
    "itemCode": "product_sku",
    "itemPrice": "unit_price",
    "unitsPurchased": "quantity",
    "grossRevenue": "total_amount",
    "billingType": "payment_method",
    "fulfillmentStatus": "order_status",
    "branch_city": "store_location",
}

retailer_a_schema = StructType(
    [
        StructField("RECEIPT_NUM", StringType(), True),
        StructField("CLIENT_REF", StringType(), True),
        StructField("TXN_DATETIME", StringType(), True),
        StructField("PRODUCT_ID", StringType(), True),
        StructField("UNIT_COST", DoubleType(), True),
        StructField("ORDER_QTY", IntegerType(), True),
        StructField("FINAL_CHARGE", DoubleType(), True),
        StructField("PAY_TYPE", StringType(), True),
        StructField("DISPATCH_FLAG", StringType(), True),
        StructField("STORE_LOCATION", StringType(), True),
    ]
)

retailer_b_schema = StructType(
    [
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_timestamp", StringType(), True),
        StructField("product_sku", StringType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("payment_method", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("store_city", StringType(), True),
    ]
)

retailer_c_schema = StructType(
    [
        StructField("billingType", StringType(), True),
        StructField(
            "branch",
            StructType([StructField("city", StringType(), True)]),
            True,
        ),
        StructField("eventTimestampMs", LongType(), True),
        StructField("fulfillmentStatus", StringType(), True),
        StructField("grossRevenue", DoubleType(), True),
        StructField("itemCode", StringType(), True),
        StructField("itemPrice", DoubleType(), True),
        StructField("transactionId", StringType(), True),
        StructField("unitsPurchased", LongType(), True),
        StructField("userId", StringType(), True),
    ]
)


def validate_column_mapping(
    source_schema: StructType,
    column_mapping: Mapping[str, str],
    derived_source_columns: Collection[str] = (),
) -> None:
    source_columns = {field.name for field in source_schema.fields} | set(
        derived_source_columns
    )
    target_columns = set(bronze_table_schema.fieldNames()) - set(metadata_columns)
    target_columns.discard("_source_retailer")
    missing_source_columns = set(column_mapping) - source_columns
    unknown_target_columns = set(column_mapping.values()) - target_columns
    unmapped_target_columns = target_columns - set(column_mapping.values())

    if missing_source_columns or unknown_target_columns or unmapped_target_columns:
        raise ValueError(
            "Invalid column mapping: "
            f"unknown source columns={sorted(missing_source_columns)}, "
            f"unknown target columns={sorted(unknown_target_columns)}, "
            f"unmapped target columns={sorted(unmapped_target_columns)}"
        )


validate_column_mapping(retailer_a_schema, retailer_a_column_mapping)
validate_column_mapping(retailer_b_schema, retailer_b_column_mapping)
validate_column_mapping(
    retailer_c_schema,
    retailer_c_column_mapping,
    derived_source_columns={"branch_city"},
)


def standardize_columns(
    df: DataFrame,
    column_mapping: Mapping[str, str],
) -> DataFrame:
    return df.select(
        *[
            col(original).alias(standardized)
            for original, standardized in column_mapping.items()
        ],
        *[col(c) for c in metadata_columns],
    )


def read_raw_delim_data(
    raw_data_folder: str,
    delimiter: str,
    source_schema: StructType,
) -> DataFrame:
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("rescuedDataColumn", "_rescued_data")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("delimiter", delimiter)
        .schema(source_schema)
        .load(raw_data_folder)
        .select(
            "*",
            col("_metadata.file_path").alias("_source_file"),
            col("_metadata.file_modification_time").alias("_source_file_modified_at"),
            current_timestamp().alias("_ingested_at"),
        )
    )


def read_raw_json_data(
    raw_data_folder: str,
    source_schema: StructType,
) -> DataFrame:
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .option("rescuedDataColumn", "_rescued_data")
        .option("cloudFiles.format", "json")
        .schema(source_schema)
        .load(raw_data_folder)
        .withColumn("branch_city", col("branch.city"))
        .drop("branch")
        .select(
            "*",
            col("_metadata.file_path").alias("_source_file"),
            col("_metadata.file_modification_time").alias("_source_file_modified_at"),
            current_timestamp().alias("_ingested_at"),
        )
    )


# 1) read raw data, adjust schema, and standardize columns for each retailer
# 2) append the standardized data to the bronze table

@dp.append_flow(target=bronze_table, name="retailer_a_flow")
def retailer_a_flow():
    df = read_raw_delim_data(
        f'{raw_data_path}/retailer_a_text',
        delimiter="|",
        source_schema=retailer_a_schema,
    )
    return (
        standardize_columns(df, retailer_a_column_mapping)
        .withColumn(
            "order_timestamp",
            to_timestamp(col("order_timestamp"), "MM/dd/yyyy hh:mm:ss a"),
        )
        .withColumn("quantity", col("quantity").cast("long"))
        .withColumn("_source_retailer", lit("retailer_a"))
        .select(*bronze_table_schema.fieldNames())
    )

@dp.append_flow(target=bronze_table, name="retailer_b_flow")
def retailer_b_flow():
    df = read_raw_delim_data(
        f'{raw_data_path}/retailer_b_csv',
        delimiter=",",
        source_schema=retailer_b_schema,
    )
    return (
        standardize_columns(df, retailer_b_column_mapping)
        .withColumn(
            "order_timestamp",
            to_timestamp(col("order_timestamp"), "yyyy-MM-dd HH:mm:ss"),
        )
        .withColumn("quantity", col("quantity").cast("long"))
        .withColumn("_source_retailer", lit("retailer_b"))
        .select(*bronze_table_schema.fieldNames())
    )

@dp.append_flow(target=bronze_table, name="retailer_c_flow")
def retailer_c_flow():
    df = read_raw_json_data(
        f'{raw_data_path}/retailer_c_json',
        source_schema=retailer_c_schema,
    )
    return (
        standardize_columns(df, retailer_c_column_mapping)
        .withColumn(
            "order_timestamp",
            timestamp_millis(col("order_timestamp")),
        )
        .withColumn("_source_retailer", lit("retailer_c"))
        .select(*bronze_table_schema.fieldNames())
    )