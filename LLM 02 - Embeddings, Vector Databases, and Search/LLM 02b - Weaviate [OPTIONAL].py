# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Weaviate
# MAGIC
# MAGIC In this notebook, we will use Weaviate as our vector database. We will then write the embedding vectors out to Weaviate and query for similar documents. Weaviate provides customization options, such as to incorporate Product Quantization or not (refer [here](https://weaviate.io/developers/weaviate/concepts/vector-index#hnsw-with-product-quantizationpq)). 
# MAGIC
# MAGIC [Zilliz](https://zilliz.com/) has an enterprise offering for Weaviate.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library pre-requisites
# MAGIC
# MAGIC - weaviate-client
# MAGIC   - pip install below
# MAGIC - Spark connector jar file
# MAGIC   - **IMPORTANT!!** Since we will be interacting with Spark by writing a Spark dataframe out to Weaviate, we need a Spark Connector.
# MAGIC   - [Download the Spark connector jar file](https://github.com/weaviate/spark-connector#download-jar-from-github) and [upload to your Databricks cluster](https://github.com/weaviate/spark-connector#using-the-jar-in-databricks).
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install weaviate-client==3.19.1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weaviate
# MAGIC
# MAGIC [Weaviate](https://weaviate.io/) is an open-source persistent and fault-tolerant [vector database](https://weaviate.io/developers/weaviate/concepts/storage). It integrates with a variety of tools, including OpenAI and Hugging Face Transformers. You can refer to their [documentation here](https://weaviate.io/developers/weaviate/quickstart).
# MAGIC
# MAGIC ### Setting up your Weaviate Network
# MAGIC
# MAGIC Before we could proceed, you need your own Weaviate Network. To start your own network, visit the [homepage](https://weaviate.io/). 
# MAGIC
# MAGIC Step 1: Click on `Start Free` 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/weaviate_homepage.png" width=500>
# MAGIC
# MAGIC Step 2: You will be brought to this [Console page](https://console.weaviate.cloud/). If this is your first time using Weaviate, click `Register here` and pass in your credentials.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/weaviate_register.png" width=500>
# MAGIC
# MAGIC Step 3: Click on `Create cluster` and select `Free sandbox`. Provide your cluster name. For simplicity, we will toggle `enable authentication` to be `No`. Then, hit `Create`. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/weaviate_create_cluster.png" width=900>
# MAGIC
# MAGIC Step 4: Click on `Details` and copy the `Cluster URL` and paste in the cell below.

# COMMAND ----------

# MAGIC %md
# MAGIC We will use embeddings from OpenAI,  so we will need a token from OpenAI API
# MAGIC
# MAGIC Steps:
# MAGIC 1. You need to [create an account](https://platform.openai.com/signup) on OpenAI. 
# MAGIC 2. Generate an OpenAI [API key here](https://platform.openai.com/account/api-keys). 
# MAGIC
# MAGIC Note: OpenAI does not have a free option, but it gives you $5 as credit. Once you have exhausted your $5 credit, you will need to add your payment method. You will be [charged per token usage](https://openai.com/pricing). **IMPORTANT**: It's crucial that you keep your OpenAI API key to yourself. If others have access to your OpenAI key, they will be able to charge their usage to your account! 

# COMMAND ----------

# TODO
import os

os.environ["OPENAI_API_KEY"] = "<FILL IN>"
os.environ["WEAVIATE_NETWORK"] = "<FILL IN>"

# COMMAND ----------

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
weaviate_network = os.environ["WEAVIATE_NETWORK"]

# COMMAND ----------

import weaviate

client = weaviate.Client(
    weaviate_network, additional_headers={"X-OpenAI-Api-Key": openai.api_key}
)
client.is_ready()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataset
# MAGIC
# MAGIC
# MAGIC In this section, we are going to use the data on <a href="https://newscatcherapi.com/" target="_blank">news topics collected by the NewsCatcher team</a>, who collects and indexes news articles and release them to the open-source community. The dataset can be downloaded from <a href="https://www.kaggle.com/kotartemiy/topic-labeled-news-dataset" target="_blank">Kaggle</a>.

# COMMAND ----------

df = (
    spark.read.option("header", True)
    .option("sep", ";")
    .format("csv")
    .load(
        f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv".replace(
            "/dbfs", "dbfs:"
        )
    )
)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to store this dataset in the Weaviate database. To do that, we first need to define a schema. A schema is where we define classes, class properties, data types, and vectorizer modules we would like to use. 
# MAGIC
# MAGIC In the schema below, notice that:
# MAGIC
# MAGIC - We capitalize the first letter of `class_name`. This is Weaviate's rule. 
# MAGIC - We specify data types within `properties`
# MAGIC - We use `text2vec-openai` as the vectorizer. 
# MAGIC   - You can also choose to upload your own vectors (refer to [docs here](https://weaviate.io/developers/weaviate/api/rest/objects#with-a-custom-vector)) or create a class without any vectors (but we won't be able to perform similarity search after).
# MAGIC
# MAGIC [Reference documentation here](https://weaviate.io/developers/weaviate/tutorials/schema)

# COMMAND ----------

class_name = "News"
class_obj = {
    "class": class_name,
    "description": "News topics collected by NewsCatcher",
    "properties": [
        {"name": "topic", "dataType": ["string"]},
        {"name": "link", "dataType": ["string"]},
        {"name": "domain", "dataType": ["string"]},
        {"name": "published_date", "dataType": ["string"]},
        {"name": "title", "dataType": ["string"]},
        {"name": "lang", "dataType": ["string"]},
    ],
    "vectorizer": "text2vec-openai",
}

# COMMAND ----------

# If the class exists before, we will delete it first
if client.schema.exists(class_name):
    print("Deleting existing class...")
    client.schema.delete_class(class_name)

print(f"Creating class: '{class_name}'")
client.schema.create_class(class_obj)

# COMMAND ----------

# MAGIC %md
# MAGIC If you are curious what the schema looks like for your class, run the following command.

# COMMAND ----------

import json

print(json.dumps(client.schema.get(class_name), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Now that the class is created, we are going to write our dataframe to the class. 
# MAGIC
# MAGIC **IMPORTANT!!** Since we are writing a Spark DataFrame out, we need a Spark Connector to Weaviate. You need to [download the Spark connector jar file](https://github.com/weaviate/spark-connector#download-jar-from-github) and [upload to your Databricks cluster](https://github.com/weaviate/spark-connector#using-the-jar-in-databricks) before running the next cell. If you do not do this, the next cell *will fail*.

# COMMAND ----------

(
    df.limit(100)
    .write.format("io.weaviate.spark.Weaviate")
    .option("scheme", "http")
    .option("host", weaviate_network.split("https://")[1])
    .option("header:X-OpenAI-Api-Key", openai.api_key)
    .option("className", class_name)
    .mode("append")
    .save()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check if the data is indeed populated. You can run either the following command or go to 
# MAGIC `https://{insert_your_cluster_url_here}/v1/objects` 
# MAGIC
# MAGIC You should be able to see the data records, rather than null objects.

# COMMAND ----------

client.query.get("News", ["topic"]).do()

# COMMAND ----------

# MAGIC %md
# MAGIC Yay! Looks like the data is populated. We can proceed further and do a query search. We are going to search for any news titles related to `locusts`. Additionally, we are going to add a filter statement, where the topic of the news has to be `SCIENCE`. Notice that we don't have to carry out the step of converting `locusts` into embeddings ourselves because we have included a vectorizer within the class earlier on.
# MAGIC
# MAGIC We will use `with_near_text` to specify the text we would like to query similar titles for. By default, Weaviate uses cosine distance to determine similar objects. Refer to [distance documentation here](https://weaviate.io/developers/weaviate/config-refs/distances#available-distance-metrics).

# COMMAND ----------

where_filter = {
    "path": ["topic"],
    "operator": "Equal",
    "valueString": "SCIENCE",
}

# We are going to search for any titles related to locusts
near_text = {"concepts": "locust"}
(
    client.query.get(class_name, ["topic", "domain", "title"])
    .with_where(where_filter)
    .with_near_text(near_text)
    .with_limit(2)
    .do()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, if you wish to supply your own embeddings at query time, you can do that too. Since embeddings are vectors, we will use `with_near_vector` instead.
# MAGIC
# MAGIC In the code block below, we additionally introduce a `distance` parameter. The lower the distance score is, the closer the vectors are to each other. Read more about the distance thresholds [here](https://weaviate.io/developers/weaviate/config-refs/distances#available-distance-metrics).

# COMMAND ----------

import openai

model = "text-embedding-ada-002"
openai_object = openai.Embedding.create(input=["locusts"], model=model)

openai_embedding = openai_object["data"][0]["embedding"]

(
    client.query.get("News", ["topic", "domain", "title"])
    .with_where(where_filter)
    .with_near_vector(
        {
            "vector": openai_embedding,
            "distance": 0.7,  # this sets a threshold for distance metric
        }
    )
    .with_limit(2)
    .do()
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
