# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Embeddings, Vector Databases, and Search
# MAGIC
# MAGIC Converting text into embedding vectors is the first step to any text processing pipeline. As the amount of text gets larger, there is often a need to save these embedding vectors into a dedicated vector index or library, so that developers won't have to recompute the embeddings and the retrieval process is faster. We can then search for documents based on our intended query and pass these relevant documents into a language model (LM) as additional context. We also refer to this context as supplying the LM with "state" or "memory". The LM then generates a response based on the additional context it receives! 
# MAGIC
# MAGIC In this notebook, we will implement the full workflow of text vectorization, vector search, and question answering workflow. While we use [FAISS](https://faiss.ai/) (vector library) and [ChromaDB](https://docs.trychroma.com/) (vector database), and a Hugging Face model, know that you can easily swap these tools out for your preferred tools or models!
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/updated_vector_search.png" width=1000 target="_blank" > 
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Implement the workflow of reading text, converting text to embeddings, saving them to FAISS and ChromaDB 
# MAGIC 2. Query for similar documents using FAISS and ChromaDB 
# MAGIC 3. Apply a Hugging Face language model for question answering!

# COMMAND ----------

# MAGIC %pip install faiss-cpu==1.7.4 chromadb==0.3.21

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Reading data
# MAGIC
# MAGIC In this section, we are going to use the data on <a href="https://newscatcherapi.com/" target="_blank">news topics collected by the NewsCatcher team</a>, who collect and index news articles and release them to the open-source community. The dataset can be downloaded from <a href="https://www.kaggle.com/kotartemiy/topic-labeled-news-dataset" target="_blank">Kaggle</a>.

# COMMAND ----------

import pandas as pd

pdf = pd.read_csv(f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv", sep=";")
pdf["id"] = pdf.index
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Library: FAISS
# MAGIC
# MAGIC Vector libraries are often sufficient for small, static data. Since it's not a full-fledged database solution, it doesn't have the CRUD (Create, Read, Update, Delete) support. Once the index has been built, if there are more vectors that need to be added/removed/edited, the index has to be rebuilt from scratch. 
# MAGIC
# MAGIC That said, vector libraries are easy, lightweight, and fast to use. Examples of vector libraries are [FAISS](https://faiss.ai/), [ScaNN](https://github.com/google-research/google-research/tree/master/scann), [ANNOY](https://github.com/spotify/annoy), and [HNSM](https://arxiv.org/abs/1603.09320).
# MAGIC
# MAGIC FAISS has several ways for similarity search: L2 (Euclidean distance), cosine similarity. You can read more about their implementation on their [GitHub](https://github.com/facebookresearch/faiss/wiki/Getting-started#searching) page or [blog post](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/). They also published their own [best practice guide here](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).
# MAGIC
# MAGIC If you'd like to read up more on the comparisons between vector libraries and databases, [here is a good blog post](https://weaviate.io/blog/vector-library-vs-vector-database#feature-comparison---library-versus-database).

# COMMAND ----------

# MAGIC %md
# MAGIC The overall workflow of FAISS is captured in the diagram below. 
# MAGIC
# MAGIC <img src="https://miro.medium.com/v2/resize:fit:1400/0*ouf0eyQskPeGWIGm" width=700>
# MAGIC
# MAGIC Source: [How to use FAISS to build your first similarity search by Asna Shafiq](https://medium.com/loopio-tech/how-to-use-faiss-to-build-your-first-similarity-search-bf0f708aa772).

# COMMAND ----------

from sentence_transformers import InputExample

pdf_subset = pdf.head(1000)

def example_create_fn(doc1: pd.Series) -> InputExample:
    """
    Helper function that outputs a sentence_transformer guid, label, and text
    """
    return InputExample(texts=[doc1])

faiss_train_examples = pdf_subset.apply(
    lambda x: example_create_fn(x["title"]), axis=1
).tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Vectorize text into embedding vectors
# MAGIC We will be using `Sentence-Transformers` [library](https://www.sbert.net/) to load a language model to vectorize our text into embeddings. The library hosts some of the most popular transformers on [Hugging Face Model Hub](https://huggingface.co/sentence-transformers).
# MAGIC Here, we are using the `model = SentenceTransformer("all-MiniLM-L6-v2")` to generate embeddings.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2", 
    cache_folder=DA.paths.datasets
)  # Use a pre-cached model
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())
len(faiss_title_embedding), len(faiss_title_embedding[0])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 3: Saving embedding vectors to FAISS index
# MAGIC Below, we create the FAISS index object based on our embedding vectors, normalize vectors, and add these vectors to the FAISS index. 

# COMMAND ----------

import numpy as np
import faiss

pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")

content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

# Index1DMap translates search results to IDs: https://faiss.ai/cpp_api/file/IndexIDMap_8h.html#_CPPv4I0EN5faiss18IndexIDMapTemplateE
# The IndexFlatIP below builds index
index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Search for relevant documents
# MAGIC
# MAGIC We define a search function below to first vectorize our query text, and then search for the vectors with the closest distance. 

# COMMAND ----------

def search_content(query, pdf_to_index, k=3):
    query_vector = model.encode([query])
    faiss.normalize_L2(query_vector)

    # We set k to limit the number of vectors we want to return
    top_k = index_content.search(query_vector, k)
    ids = top_k[1][0].tolist()
    similarities = top_k[0][0].tolist()
    results = pdf_to_index.loc[ids]
    results["similarities"] = similarities
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC Tada! Now you can query for similar content! Notice that you did not have to configure any database networks beforehand nor pass in any credentials. FAISS works locally with your code.

# COMMAND ----------

display(search_content("animal", pdf_to_index))

# COMMAND ----------

# MAGIC %md
# MAGIC Up until now, we haven't done the last step of conducting Q/A with a language model yet. We are going to demonstrate this with Chroma, a vector database.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Database: Chroma
# MAGIC
# MAGIC Chroma is an open-source embedding database. The company just raised its [seed funding in April 2023](https://www.trychroma.com/blog/seed) and is quickly becoming popular to support LLM-based applications. 

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
# MAGIC ### Chroma Concept: Collection
# MAGIC
# MAGIC Chroma `collection` is akin to an index that stores one set of your documents. 
# MAGIC
# MAGIC According to the [docs](https://docs.trychroma.com/getting-started): 
# MAGIC > Collections are where you will store your embeddings, documents, and additional metadata
# MAGIC
# MAGIC The nice thing about ChromaDB is that if you don't supply a model to vectorize text into embeddings, it will automatically load a default embedding function, i.e. `SentenceTransformerEmbeddingFunction`. It can handle tokenization, embedding, and indexing automatically for you. If you would like to change the embedding model, read [here on how to do that](https://docs.trychroma.com/embeddings). TLDR: you can add an optional `model_name` argument. 
# MAGIC
# MAGIC You can read [the documentation here](https://docs.trychroma.com/usage-guide#using-collections) on rules for collection names.

# COMMAND ----------

collection_name = "my_news"

# If you have created the collection before, you need to delete the collection first
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

print(f"Creating collection: '{collection_name}'")
collection = chroma_client.create_collection(name=collection_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Add data to collection
# MAGIC
# MAGIC Since we are re-using the same data, we can skip the step of reading data. As mentioned in the text above, Chroma can take care of text vectorization for us, so we can directly add text to the collection and Chroma will convert the text into embeddings behind the scene. 

# COMMAND ----------

display(pdf_subset)

# COMMAND ----------

# MAGIC %md
# MAGIC Each document must have a unique `id` associated with it and it is up to you to check that there are no duplicate ids. 
# MAGIC
# MAGIC Adding data to collection will take some time to run, especially when there is a lot of data. In the cell below, we intentionally write only a subset of data to the collection to speed things up. 

# COMMAND ----------

collection.add(
    documents=pdf_subset["title"][:100].tolist(),
    metadatas=[{"topic": topic} for topic in pdf_subset["topic"][:100].tolist()],
    ids=[f"id{x}" for x in range(100)],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Query for 10 relevant documents on "space"
# MAGIC
# MAGIC We will return 10 most relevant documents. You can think of `10` as 10 nearest neighbors. You can also change the number of results returned as well. 

# COMMAND ----------

import json

results = collection.query(query_texts=["space"], n_results=10)

print(json.dumps(results, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Add filter statement
# MAGIC
# MAGIC In addition to conducting relevancy search, we can also add filter statements. Refer to the [documentation](https://docs.trychroma.com/usage-guide#using-where-filters) for more information.

# COMMAND ----------

collection.query(query_texts=["space"], where={"topic": "SCIENCE"}, n_results=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Update data in a collection
# MAGIC
# MAGIC Unlike a vector library, vector databases support changes to the data so we can update or delete data. 
# MAGIC
# MAGIC Indeed, we can update or delete data in a Chroma collection. 

# COMMAND ----------

collection.delete(ids=["id0"])

# COMMAND ----------

# MAGIC %md
# MAGIC The record with `ids=0` is no longer present.

# COMMAND ----------

collection.get(
    ids=["id0"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also update a specific data point.

# COMMAND ----------

collection.get(
    ids=["id2"],
)

# COMMAND ----------

collection.update(
    ids=["id2"],
    metadatas=[{"topic": "TECHNOLOGY"}],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt engineering for question answering 
# MAGIC
# MAGIC Now that we have identified documents about space from the news dataset, we can pass these documents as additional context for a language model to generate a response based on them! 
# MAGIC
# MAGIC We first need to pick a `text-generation` model. Below, we use a Hugging Face model. You can also use OpenAI as well, but you will need to get an Open AI token and [pay based on the number of tokens](https://openai.com/pricing). 

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=DA.paths.datasets)
lm_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=DA.paths.datasets)

pipe = pipeline(
    "text-generation",
    model=lm_model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    device_map="auto",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Here's where prompt engineering, which is developing prompts, comes in. We pass in the context in our `prompt_template` but there are numerous ways to write a prompt. Some prompts may generate better results than the others and it requires some experimentation to figure out how best to talk to the model. Each language model behaves differently to prompts. 
# MAGIC
# MAGIC Our prompt template below is inspired from a [2023 paper on program-aided language model](https://arxiv.org/pdf/2211.10435.pdf). The authors have provided their sample prompt template [here](https://github.com/reasoning-machines/pal/blob/main/pal/prompt/date_understanding_prompt.py).
# MAGIC
# MAGIC The following links also provide some helpful guidance on prompt engineering: 
# MAGIC - [Prompt engineering with OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
# MAGIC - [GitHub repo that compiles best practices to interact with ChatGPT](https://github.com/f/awesome-chatgpt-prompts)

# COMMAND ----------

question = "What's the latest news on space development?"
context = " ".join([f"#{str(i)}" for i in results["documents"][0]])
prompt_template = f"Relevant context: {context}\n\n The user's question: {question}"

# COMMAND ----------

lm_response = pipe(prompt_template)
print(lm_response[0]["generated_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Yay, you have just completed the implementation of your first text vectorization, search, and question answering workflow (that requires prompt engineering)!
# MAGIC
# MAGIC In the lab, you will apply your newly gained knowledge to a different dataset. You can also check out the optional modules on Pinecone and Weaviate to learn how to set up vector databases that offer enterprise offerings.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
