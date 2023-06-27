# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.paths.working_dir = DA.paths.to_vm_path(DA.paths.working_dir)
DA.paths.datasets = DA.paths.to_vm_path(DA.paths.datasets)

# COMMAND ----------

# MAGIC %run ./Test-Framework

# COMMAND ----------

import pathlib
import subprocess
import os
import shutil

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

dbfs_hf_cache = "/dbfs/mnt/dbacademy-datasets/large-language-models/v01/hf_cache/hf.tar"
local_hf_cache_tar = "/local_disk0/hf.tar"
local_hf_cache_dir = "/local_disk0/hf"

if pathlib.Path(dbfs_hf_cache).is_file() and not(pathlib.Path(local_hf_cache_tar).is_file()):
  print("Loading HuggingFace cache. This can take up to 6 minutes to transfer a cache of datasets and models to your cluster to address HuggingFace instability issues and the parallelism of many students accessing these files concurrently...")
  shutil.copyfile(dbfs_hf_cache, local_hf_cache_tar)
  
  print("Extracting HuggingFace cache...")
  cmd = f"cd /local_disk0/ && tar -xvf {local_hf_cache_tar}"
  
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
  process.wait()

else:
  print("Found HuggingFace cache either found on local disk or will download directly from HuggingFace...")

# COMMAND ----------

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe models developed or used in this course are for demonstration and learning purposes only.\nModels may occasionally output offensive, inaccurate, biased information, or harmful instructions.")

