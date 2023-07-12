# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # 02L - Embeddings, Vector Databases, and Search
# MAGIC
# MAGIC
# MAGIC In this lab, we will apply the text vectorization, search, and question answering workflow that you learned in the demo. The dataset we will use this time will be on talk titles and sessions from [Data + AI Summit 2023](https://www.databricks.com/dataaisummit/). 
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Learn how to use Chroma to store your embedding vectors and conduct similarity search
# MAGIC 1. Use OpenAI GPT-3.5 to generate response to your prompt

# COMMAND ----------

# MAGIC %pip install chromadb==0.3.21 tiktoken==0.3.3

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data

# COMMAND ----------

import pandas as pd

dais_pdf = pd.read_parquet(f"{DA.paths.datasets}/dais/dais23_talks.parquet")
display(dais_pdf)

# COMMAND ----------

dais_pdf["full_text"] = dais_pdf.apply(
    lambda row: f"""Title: {row["Title"]}
                Abstract:  {row["Abstract"]}""".strip(),
    axis=1,
)
print(dais_pdf.iloc[0]["full_text"])

# COMMAND ----------

texts = dais_pdf["full_text"].to_list()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1
# MAGIC Set up Chroma and create collection

# COMMAND ----------

import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DA.paths.user_db,  # this is an optional argument. If you don't supply this, the data will be ephemeral
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Assign the value of `my_talks` to the `collection_name` variable.

# COMMAND ----------

# TODO
collection_name = "<FILL_IN>"

# If you have created the collection before, you need to delete the collection first
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)
else:
    print(f"Creating collection: '{collection_name}'")
    talks_collection = chroma_client.create_collection(name=collection_name)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_1(collection_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2
# MAGIC
# MAGIC [Add](https://docs.trychroma.com/reference/Collection#add) data to the collection. 

# COMMAND ----------

# TODO
talks_collection.add(
    documents=<FILL_IN>,
    ids=<FILL_IN>
)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_2(talks_collection)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 3
# MAGIC
# MAGIC [Query](https://docs.trychroma.com/reference/Collection#query) for relevant documents. If you are looking for talks related to language models, your query texts could be `language models`. 

# COMMAND ----------

# TODO
import json

results = talks_collection.query(
    query_texts=<FILL_IN>,
    n_results=<FILL_IN>
)

print(json.dumps(results, indent=4))

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_3(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 4
# MAGIC
# MAGIC Load a language model and create a [pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines).

# COMMAND ----------

# TODO
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Pick a model from HuggingFace that can generate text
model_id = "<FILL_IN>"
tokenizer = AutoTokenizer.from_pretrained(model_id)
lm_model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "<FILL_IN>", model=lm_model, tokenizer=tokenizer, max_new_tokens=512, device_map="auto", handle_long_generation="hole"
)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_4(pipe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 5
# MAGIC
# MAGIC Prompt engineering for question answering

# COMMAND ----------

# TODO
# Come up with a question that you need the LLM assistant to help you with
# A sample question is "Help me find sessions related to XYZ" 
# Note: Your "XYZ" should be related to the query you passed in Question 3. 
question = "<FILL_IN>"

# Provide all returned similar documents from the cell above below
context = <FILL_IN>

# Feel free to be creative how you construct the prompt. You can use the demo notebook as a jumpstart reference.
# You can also provide more requirements in the text how you want the answers to look like.
# Example requirement: "Recommend top-5 relevant sessions for me to attend."
prompt_template = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_5(question, context, prompt_template)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 6 
# MAGIC
# MAGIC Submit query for language model to generate response.
# MAGIC
# MAGIC Hint: If you run into the error `index out of range in self`, make sure to check out this [documentation page](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline.__call__.handle_long_generation).

# COMMAND ----------

# TODO
lm_response = pipe(<FILL_IN>)
print(lm_response[0]["generated_text"])

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion2_6(lm_response)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that the output isn't exactly helpful. Head on to using OpenAI to try out GPT-3.5 instead! 

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL (Non-Graded): Use OpenAI models for Q/A
# MAGIC
# MAGIC For this section to work, you need to generate an Open AI key. 
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

# COMMAND ----------

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# COMMAND ----------

# MAGIC %md
# MAGIC If you would like to estimate how much it would cost to use OpenAI, you can use `tiktoken` library from OpenAI to get the number of tokens from your prompt.
# MAGIC
# MAGIC
# MAGIC We will be using `gpt-3.5-turbo` since it's the most economical option at ($0.002/1k tokens), as of May 2023. GPT-4 charges $0.04/1k tokens. The following code block below is referenced from OpenAI's documentation on ["Managing tokens"](https://platform.openai.com/docs/guides/chat/managing-tokens).

# COMMAND ----------

import tiktoken

price_token = 0.002
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
cost_to_run = len(encoder.encode(prompt_template)) / 1000 * price_token
print(f"It would take roughly ${round(cost_to_run, 5)} to run this prompt")

# COMMAND ----------

# MAGIC %md
# MAGIC We won't have to create a new vector database again. We can just send our `context` from above to OpenAI. We will use their chat completion API to interact with `GPT-3.5-turbo`. You can refer to their [documentation here](https://platform.openai.com/docs/guides/chat).
# MAGIC
# MAGIC Something interesting is that OpenAI models use the system message to help their assistant to be more accurate. From OpenAI's [docs](https://platform.openai.com/docs/guides/chat/introduction):
# MAGIC
# MAGIC > Future models will be trained to pay stronger attention to system messages. The system message helps set the behavior of the assistant.
# MAGIC
# MAGIC

# COMMAND ----------

# TODO
gpt35_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": <FILL_IN>},
    ],
    temperature=0, # 0 makes outputs deterministic; The closer the value is to 1, the more random the outputs are for each time you re-run.
)

# COMMAND ----------

print(gpt35_response.choices[0]["message"]["content"])

# COMMAND ----------

from IPython.display import Markdown

Markdown(gpt35_response.choices[0]["message"]["content"])

# COMMAND ----------

# MAGIC %md
# MAGIC We can also check how many tokens OpenAI has used

# COMMAND ----------

gpt35_response["usage"]["total_tokens"]

# COMMAND ----------

# MAGIC %md
# MAGIC The results are noticeably much better compared to when using Hugging Face's GPT-2! It didn't get stuck in the text generation, but the sessions recommended are not all relevant to pandas either. You can further do more prompt engineering to get better results.

# COMMAND ----------

# MAGIC %md ## Submit your Results (edX Verified Only)
# MAGIC
# MAGIC To get credit for this lab, click the submit button in the top right to report the results. If you run into any issues, click `Run` -> `Clear state and run all`, and make sure all tests have passed before re-submitting. If you accidentally deleted any tests, take a look at the notebook's version history to recover them or reload the notebooks.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
