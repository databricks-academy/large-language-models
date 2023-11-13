# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Pinecone
# MAGIC
# MAGIC In this section, we are going to try out another vector database, called Pinecone. It has a free tier which you need to sign up for to gain access below.
# MAGIC
# MAGIC Pinecone is a cloud-based vector database that offers fast and scalable similarity search for high-dimensional data, with a focus on simplicity and ease of use. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library pre-requisites
# MAGIC
# MAGIC - pinecone-client
# MAGIC   - pip install below
# MAGIC - Spark connector jar file
# MAGIC   - **IMPORTANT!!** Since we will be interacting with Spark by writing a Spark dataframe out to Pinecone, we need a Spark Connector.
# MAGIC   - You need to attach a Spark-Pinecone connector `s3://pinecone-jars/0.2.1/spark-pinecone-uberjar.jar` in the cluster you are using. Refer to this [documentation](https://docs.pinecone.io/docs/databricks#setting-up-a-spark-cluster) if you need more information. 

# COMMAND ----------

# MAGIC %pip install pinecone-client==2.2.2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ### Setting up your Pinecone
# MAGIC
# MAGIC Step 1: Go to their [home page](https://www.pinecone.io/) and click `Sign Up Free` on the top right corner. 
# MAGIC <br>
# MAGIC Step 2: Click on `Sign Up`. It's possible that you may not be able to sign up for a new account, depending on Pinecone's availability. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/pinecone_register.png" width=300>
# MAGIC
# MAGIC Step 3: Once you are in the console, navigate to `API Keys` and copy the `Environment` and `Value` (this is your API key).
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/pinecone_credentials.png" width=500>

# COMMAND ----------

# TODO
import os

os.environ["PINECONE_API_KEY"] = "<FILL IN>"
os.environ["PINECONE_ENV"] = "<FILL IN>"

# COMMAND ----------

import pinecone

pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_env = os.environ["PINECONE_ENV"]

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# COMMAND ----------

import pyspark.sql.functions as F

df = (
    spark.read.option("header", True)
    .option("sep", ";")
    .format("csv")
    .load(
        f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv".replace(
            "/dbfs", "dbfs:"
        )
    )
    .withColumn("id", F.expr("uuid()"))
)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC For Pinecone, we need to generate the embeddings first and save it to a dataframe, before we can write it out to Pinecone for indexing. 
# MAGIC
# MAGIC There are two ways of doing it: 
# MAGIC 1. Using pandas DataFrame, apply the single-node embedding model, and upsert to Pinecone in batches
# MAGIC 2. Using Spark Dataframe and use pandas UDFs to help us apply the embedding model on batches of data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1: Upsert to Pinecone in batches

# COMMAND ----------

pdf = df.limit(1000).toPandas()
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC Note: Pinecone free tier only allows one index. If you have existing indices, you need to delete them before you are able create a new index.
# MAGIC
# MAGIC We specify the similarity measure, embedding vector dimension within the index.
# MAGIC
# MAGIC Read documentation on how to [create index here](https://docs.pinecone.io/reference/create_index/).

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# We will use embeddings from this model to apply to our data
model = SentenceTransformer(
    "all-MiniLM-L6-v2", cache_folder=DA.paths.datasets
)  # Use a pre-cached model

# COMMAND ----------

# MAGIC %md
# MAGIC Delete the index if it already exists

# COMMAND ----------

pinecone_index_name = "news"

if pinecone_index_name in pinecone.list_indexes():
    pinecone.delete_index(pinecone_index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Create the index.
# MAGIC
# MAGIC We specify the index name (required), embedding vector dimension (required), and a custom similarity metric (cosine is the default) when creating our index.

# COMMAND ----------

# only create index if it doesn't exist
if pinecone_index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=pinecone_index_name,
        dimension=model.get_sentence_embedding_dimension(),
        metric="cosine",
    )

# now connect to the index
pinecone_index = pinecone.Index(pinecone_index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC When the index has been created, we can now upsert vectors of data records to the index. `Upsert` means that we are writing the vectors into the index. 
# MAGIC
# MAGIC Refer to this [documentation page](https://docs.pinecone.io/docs/python-client#indexupsert) to look at example code and vectors.

# COMMAND ----------

from tqdm.auto import tqdm

batch_size = 1000

for i in tqdm(range(0, len(pdf["title"]), batch_size)):
    # find end of batch
    i_end = min(i + batch_size, len(pdf["title"]))
    # create IDs batch
    ids = [str(x) for x in range(i, i_end)]
    # create metadata batch
    metadata = [{"title": title} for title in pdf["title"][i:i_end]]
    # create embeddings
    embedding_title_batch = model.encode(pdf["title"][i:i_end]).tolist()
    # create records list for upsert
    records = zip(ids, embedding_title_batch, metadata)
    # upsert to Pinecone
    pinecone_index.upsert(vectors=records)

# check number of records in the index
pinecone_index.describe_index_stats()

# COMMAND ----------

# MAGIC %md
# MAGIC Once the vectors are upserted, we can now query the index directly! Notice that it returns us the similarity score in the result too.

# COMMAND ----------

query = "fish"

# create the query vector
user_query = model.encode(query).tolist()

# submit the query to the Pinecone index
pinecone_answer = pinecone_index.query(user_query, top_k=3, include_metadata=True)

for result in pinecone_answer["matches"]:
    print(f"{round(result['score'], 2)}, {result['metadata']['title']}")
    print("-" * 120)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2: Process with Spark and write to Pinecone with Spark
# MAGIC
# MAGIC Now that we have seen how to `upsert` with Pinecone, you may be curious whether we can use Spark DataFrame Writer (just like Weaviate) to write the entire dataframe out in a single operation. The answer is yes -- we will now take a look at how to do that and a spoiler alert is that you will need to use a Spark connector too! 
# MAGIC
# MAGIC We first need to write a mapping function to map the tokenizer and embedding model onto batches of rows within the Spark DataFrame. We will be using a type of [pandas UDFs](https://www.databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html), called scalar iterator UDFs. 
# MAGIC
# MAGIC > The function takes and outputs an iterator of pandas.Series.
# MAGIC
# MAGIC > The length of the whole output must be the same length of the whole input. Therefore, it can prefetch the data from the input iterator as long as the lengths of entire input and output are the same. The given function should take a single column as input.
# MAGIC
# MAGIC > It is also useful when the UDF execution requires expensive initialization of some state. 
# MAGIC
# MAGIC We load the model once per partition of data, not per [batch](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html#setting-arrow-batch-size), which is faster. 
# MAGIC
# MAGIC For more documentation, refer [here](https://docs.databricks.com/udf/pandas.html).

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf
from sentence_transformers import SentenceTransformer
from typing import Iterator

@pandas_udf("array<float>")
def create_embeddings_with_transformers(
    sentences: Iterator[pd.Series],) -> Iterator[pd.Series]:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for batch in sentences:
        yield pd.Series(model.encode(batch).tolist())

# COMMAND ----------

import pyspark.sql.functions as F

transformer_type = "sentence-transformers/all-MiniLM-L6-v2"
embedding_spark_df = (
    df.limit(1000)
    .withColumn("values", create_embeddings_with_transformers("title")) 
    .withColumn("namespace", F.lit(None)) ## Pinecone free-tier does not support namespace
    .withColumn("sparse_values", F.lit(None)) ## required by Pinecone v2.0.1 release
    .withColumn("metadata", F.to_json(F.struct(F.col("topic").alias("TOPIC"))))
    # We select these columns because they are expected by the Spark-Pinecone connector
    .select("id", "values", "sparse_values", "namespace", "metadata")
)
display(embedding_spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Repeat the same step as in Method 1 above to delete and recreate the index. Again, we need to delete the index because Pinecone free tier only allows one index.
# MAGIC
# MAGIC Note: This could take ~3 minutes. 

# COMMAND ----------

pinecone_index_name = "news"

if pinecone_index_name in pinecone.list_indexes():
    pinecone.delete_index(pinecone_index_name)

# only create index if it doesn't exist
model = SentenceTransformer(transformer_type)
if pinecone_index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=pinecone_index_name,
        dimension=model.get_sentence_embedding_dimension(),
        metric="cosine",
    )

# now connect to the index
pinecone_index = pinecone.Index(pinecone_index_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of writing in batches, you can now use Spark DataFrame Writer to write the data out to Pinecone directly.
# MAGIC
# MAGIC **IMPORTANT!!** You need to attach a Spark-Pinecone connector `s3://pinecone-jars/0.2.1/spark-pinecone-uberjar.jar` in the cluster you are using. Otherwise, this following command would fail. Refer to this [documentation](https://docs.pinecone.io/docs/databricks#setting-up-a-spark-cluster) and release note [here](https://github.com/pinecone-io/spark-pinecone/releases/tag/v0.2.1) if you need more information. 

# COMMAND ----------

(
    embedding_spark_df.write.option("pinecone.apiKey", pinecone_api_key)
    .option("pinecone.environment", pinecone_env)
    .option("pinecone.projectName", pinecone.whoami().projectname)
    .option("pinecone.indexName", pinecone_index_name)
    .format("io.pinecone.spark.pinecone.Pinecone")
    .mode("append")
    .save()
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
