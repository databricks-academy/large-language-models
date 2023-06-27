# Databricks notebook source
# MAGIC %md Note: run this once per workspace in order to cache the datasets and models.

# COMMAND ----------

# MAGIC %pip install transformers datasets sacremoses==0.0.53

# COMMAND ----------

import os
os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

from datasets.utils.info_utils import VerificationMode
from datasets import load_dataset

load_dataset("Helsinki-NLP/tatoeba_mt", "eng-jpn", split="test", verification_mode=VerificationMode.NO_CHECKS)
load_dataset("poem_sentiment", version="1.0.0")
load_dataset("cnn_dailymail", "3.0.0")
load_dataset("wiki_bio", split="test")
load_dataset("AlexaAI/bold", split="train")

ds_list = [
    "xsum",
    "poem_sentiment",
    "databricks/databricks-dolly-15k",
    "wiki_bio",
    "AlexaAI/bold",
]

for ds in ds_list:
  print(ds)
  load_dataset(ds)

# COMMAND ----------

model_list = [
    "nickwong64/bert-base-uncased-poems-sentiment",
    "Helsinki-NLP/opus-mt-en-es",
    "t5-small",
    "t5-base",
    "cross-encoder/nli-deberta-v3-small",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-70m-deduped",
    "sentence-transformers/all-MiniLM-L6-v2",
    "gpt2",
    "bert-base-uncased",
    "facebook/roberta-hate-speech-dynabench-r4-target",
    "sasha/regardv3",
    "dslim/bert-base-NER",
]

# COMMAND ----------

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")


# COMMAND ----------

from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/local_disk0/hf")  

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

for m in model_list:
  print(m)
  AutoModel.from_pretrained(m)
  AutoTokenizer.from_pretrained(m)

# COMMAND ----------

!mkdir -p  /dbfs/mnt/dbacademy-datasets/large-language-models/v01/hf_cache

# COMMAND ----------

!rm -f /local_disk0/hf.tar

# COMMAND ----------

!cd /local_disk0/ && tar -cvf hf.tar hf/

# COMMAND ----------

!ls -lah /local_disk0/

# COMMAND ----------

!cp /local_disk0/hf.tar /dbfs/mnt/dbacademy-datasets/large-language-models/v01/hf_cache

# COMMAND ----------

!ls -lah /dbfs/mnt/dbacademy-datasets/large-language-models/v01/hf_cache

# COMMAND ----------

!tar -tvf /dbfs/mnt/dbacademy-datasets/large-language-models/v01/hf_cache/hf.tar | grep Mini

# COMMAND ----------


