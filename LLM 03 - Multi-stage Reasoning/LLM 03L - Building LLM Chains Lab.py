# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Lab: Adding Our Own Data to a Multi-Stage Reasoning System
# MAGIC
# MAGIC ### Working with external knowledge bases 
# MAGIC In this notebook we're going to augment the knowledge base of our LLM with additional data. We will split the notebook into two halves:
# MAGIC - First, we will walk through how to load in a relatively small, local text file using a `DocumentLoader`, split it into chunks, and store it in a vector database using `ChromaDB`.
# MAGIC - Second, you will get a chance to show what you've learned by building a larger system with the complete works of Shakespeare. 
# MAGIC ----
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC
# MAGIC By the end of this notebook, you will be able to:
# MAGIC 1. Add external local data to your LLM's knowledge base via a vector database.
# MAGIC 2. Construct a Question-Answer(QA) LLMChain to "talk to your data."
# MAGIC 3. Load external data sources from remote locations and store in a vector database.
# MAGIC 4. Leverage different retrieval methods to search over your data. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %pip install chromadb==0.4.10 tiktoken==0.3.3 sqlalchemy==2.0.15 langchain==0.0.249

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md Fill in your credentials.

# COMMAND ----------

# TODO
# For many of the services that we'll using in the notebook, we'll need a HuggingFace API key so this cell will ask for it:
# HuggingFace Hub: https://huggingface.co/inference-api

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "<FILL IN>"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Building a Personalized Document Oracle
# MAGIC
# MAGIC In this notebook, we're going to build a special type of LLMChain that will enable us to ask questions of our data. We will be able to "speak to our data".

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 1 - Loading Documents into our Vector Store
# MAGIC For this system we'll leverage the [ChromaDB vector database](https://www.trychroma.com/) and load in some text we have on file. This file is of a hypothetical laptop being reviewed in both long form and with brief customer reviews. We'll use LangChain's `TextLoader` to load this data.

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# We have some fake laptop reviews that we can load in
laptop_reviews = TextLoader(
    f"{DA.paths.datasets}/reviews/fake_laptop_reviews.txt", encoding="utf8"
)
document = laptop_reviews.load()
display(document)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 2 - Chunking and Embeddings
# MAGIC
# MAGIC Now that we have the data in document format, we will split data into chunks using a `CharacterTextSplitter` and embed this data using Hugging Face's embedding LLM to embed this data for our vector store.

# COMMAND ----------

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import tempfile

tmp_laptop_dir = tempfile.TemporaryDirectory()
tmp_shakespeare_dir = tempfile.TemporaryDirectory()

# First we split the data into manageable chunks to store as vectors. There isn't an exact way to do this, more chunks means more detailed context, but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(document)
# Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. We'll use the sentence-transformers model for out embeddings. https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models/
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, cache_folder=DA.paths.datasets
)  # Use a pre-cached model
# Finally we make our Index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(
    texts, embeddings, persist_directory=tmp_laptop_dir.name
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 3 - Creating our Document QA LLM Chain
# MAGIC With our data now in vector form we need an LLM and a chain to take our queries and create tasks for our LLM to perform. 

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# We want to make this a retriever, so we need to convert our index.  This will create a wrapper around the functionality of our vector database so we can search for similar documents/chunks in the vectorstore and retrieve the results:
retriever = chromadb_index.as_retriever()

# This chain will be used to do QA on the document. We will need
# 1 - A LLM to do the language interpretation
# 2 - A vector database that can perform document retrieval
# 3 - Specification on how to deal with this data (more on this soon)

hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        "temperature": 0,
        "max_length": 128,
        "cache_dir": DA.paths.datasets,
    },
)

chain_type = "stuff"  # Options: stuff, map_reduce, refine, map_rerank
laptop_qa = RetrievalQA.from_chain_type(
    llm=hf_llm, chain_type="stuff", retriever=retriever
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Step 4 - Talking to Our Data
# MAGIC Now we are ready to send prompts to our LLM and have it use our prompt, the access to our data, and read the information, process, and return with a response.

# COMMAND ----------

# Let's ask the chain about the product we have.
laptop_name = laptop_qa.run("What is the full name of the laptop?")
display(laptop_name)

# COMMAND ----------

# Now we'll ask the chain about the product.
laptop_features = laptop_qa.run("What are some of the laptop's features?")
display(laptop_features)

# COMMAND ----------

# Finally let's ask the chain about the reviews.
laptop_reviews = laptop_qa.run("What is the general sentiment of the reviews?")
display(laptop_reviews)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exercise: Working with larger documents
# MAGIC This document was relatively small. So let's see if we can work with something bigger. To show how well we can scale the vector database, let's load in a larger document. For this we'll get data from the [Gutenberg Project](https://www.gutenberg.org/) where thousands of free-to-access texts. We'll use the complete works of William Shakespeare.
# MAGIC
# MAGIC Instead of a local text document, we'll download the complete works of Shakespeare using the `GutenbergLoader` that works with the Gutenberg project: https://www.gutenberg.org

# COMMAND ----------

from langchain.document_loaders import GutenbergLoader

loader = GutenbergLoader(
    "https://www.gutenberg.org/cache/epub/100/pg100.txt"
)  # Complete works of Shakespeare in a txt file

all_shakespeare_text = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 1
# MAGIC
# MAGIC Now it's your turn! Based on what we did previously, fill in the missing parts below to build your own QA LLMChain.

# COMMAND ----------

# TODO
text_splitter = <FILL_IN> #hint try chunk sizes of 1024 and an overlap of 256 (this will take approx. 10mins with this model to build our vector database index)
texts = <FILL_IN>

model_name = <FILL_IN> #hint, try "sentence-transformers/all-MiniLM-L6-v2" as your model
embeddings = <FILL_IN>
chromadb_index = <FILL_IN>

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion3_1(embeddings, chromadb_index)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Question 2
# MAGIC
# MAGIC Let's see if we can do what we did with the laptop reviews. 
# MAGIC
# MAGIC Think about what is likely to happen now. Will this command succeed? 
# MAGIC
# MAGIC (***Hint: think about the maximum sequence length of a model***)

# COMMAND ----------

# TODO
# Let's start with the simplest method: "Stuff" which puts all of the data into the prompt and asks a question of it:
qa = RetrievalQA.from_chain_type(<FILL_IN>)
query = "What happens in the play Hamlet?"
# Run the query
query_results_hamlet = <FILL_IN>

query_results_hamlet

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion3_2(qa, query_results_hamlet)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Question 3
# MAGIC
# MAGIC Now that we're working with larger documents, we should be mindful of the input sequence limitations that our LLM has. 
# MAGIC
# MAGIC Chain Types for document loader:
# MAGIC
# MAGIC - [`stuff`](https://docs.langchain.com/docs/components/chains/index_related_chains#stuffing) - Stuffing is the simplest method, whereby you simply stuff all the related data into the prompt as context to pass to the language model.
# MAGIC - [`map_reduce`](https://docs.langchain.com/docs/components/chains/index_related_chains#map-reduce) - This method involves running an initial prompt on each chunk of data (for summarization tasks, this could be a summary of that chunk; for question-answering tasks, it could be an answer based solely on that chunk).
# MAGIC - [`refine`](https://docs.langchain.com/docs/components/chains/index_related_chains#refine) - This method involves running an initial prompt on the first chunk of data, generating some output. For the remaining documents, that output is passed in, along with the next document, asking the LLM to refine the output based on the new document.
# MAGIC - [`map_rerank`](https://docs.langchain.com/docs/components/chains/index_related_chains#map-rerank) - This method involves running an initial prompt on each chunk of data, that not only tries to complete a task but also gives a score for how certain it is in its answer. The responses are then ranked according to this score, and the highest score is returned.
# MAGIC   * NOTE: For this exercise, `map_rerank` will [error](https://github.com/hwchase17/langchain/issues/3970).

# COMMAND ----------

# TODO
qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type=<FILL_IN>, retriever=chromadb_index.as_retriever())
query = "Who is the main character in the Merchant of Venice?"
query_results_venice = <FILL_IN>

query_results_venice

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion3_3(qa, query_results_venice)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Question 4
# MAGIC

# COMMAND ----------

# TODO
# That's much better! Let's try another type

qa = RetrievalQA.from_chain_type(llm=hf_llm, chain_type=<FILL_IN>, retriever=chromadb_index.as_retriever())
query = "What happens to romeo and juliet?"
query_results_romeo = <FILL_IN>

query_results_romeo

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion3_4(qa, query_results_romeo)

# COMMAND ----------

# MAGIC %md ## Submit your Results (edX Verified Only)
# MAGIC
# MAGIC To get credit for this lab, click the submit button in the top right to report the results. If you run into any issues, click `Run` -> `Clear state and run all`, and make sure all tests have passed before re-submitting. If you accidentally deleted any tests, take a look at the notebook's version history to recover them or reload the notebooks.

# COMMAND ----------

tmp_laptop_dir.cleanup()
tmp_shakespeare_dir.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
