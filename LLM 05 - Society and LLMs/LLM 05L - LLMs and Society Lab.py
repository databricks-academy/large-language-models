# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LLMs and Society Lab
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Learn how to evaluate polarity towards certain demographic groups using `regard`
# MAGIC     - We will first evaluate whether dancers are regarded differently from scientists
# MAGIC     - You will then compute `regard` with other groups of your choice
# MAGIC 2. Test your language model by changing text using `sparknlp` 

# COMMAND ----------

# MAGIC %pip install nlptest==1.4.0

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Regard 
# MAGIC
# MAGIC We will use the [BOLD dataset](https://huggingface.co/datasets/AlexaAI/bold), created by Alexa AI, that allows us to evaluate model fairness in English text generation. Specifically, we will use categories within this dataset to prompt the language model for text completion. Some example categories include:
# MAGIC - gender 
# MAGIC - professions
# MAGIC - religions
# MAGIC
# MAGIC Continuing from the demo, we will use the same `evaluate` library from Hugging Face, but leverage a separate module called `regard`. We evaluate model fairness from the angle of polarity or perception to see if one group is assigned a higher regard score than the other.  

# COMMAND ----------

from datasets import load_dataset

bold = load_dataset(
    "AlexaAI/bold", split="train", cache_dir=DA.paths.datasets
)  # Note: We specify cache_dir to use pre-cached data.

# COMMAND ----------

# MAGIC %md
# MAGIC In this exercise, we will examine whether people in the `dance_occupations` are regarded any differently than those in `scientific_occupations`.

# COMMAND ----------

from random import sample

def generate_samples(category_name: str, n: int) -> list:
    """
    Given a category, returns `n` samples
    """
    bold_samples = sample([p for p in bold if p["category"] == category_name], n)
    return bold_samples

science_bold = generate_samples("scientific_occupations", 10)
dance_bold = generate_samples("dance_occupations", 10)

print("Science example: ", science_bold[0])
print("-" * 60)
print("Dance example: ", dance_bold[0])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Question 1
# MAGIC Now, it's your turn to generate some samples. Run the following cell to get a complete list of categories covered by BOLD.
# MAGIC
# MAGIC If you need inspiration, try `American_actors` and `American_actresses`.

# COMMAND ----------

import numpy as np

np.unique(bold["category"])

# COMMAND ----------

# TODO

# Generate samples from BOLD dataset
group1_bold = generate_samples("<FILL_IN>", 10)
group2_bold = generate_samples("<FILL_IN>", 10)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion5_1(group1_bold, group2_bold)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's get some prompts from each of the categories

# COMMAND ----------

science_prompts = [p["prompts"][0] for p in science_bold]
dance_prompts = [p["prompts"][0] for p in dance_bold]
print("Science prompt example: ", science_prompts[0])
print("Dance prompt example: ", dance_prompts[0])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Question 2
# MAGIC It's your turn to get prompts from the samples.

# COMMAND ----------

# TODO

group1_prompts = [p["prompts"][0] for p in <FILL_IN>]
group2_prompts = [p["prompts"][0] for p in <FILL_IN>]

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion5_2(group1_prompts, group2_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's put GPT-2 to test. Does our model complete the sentences with equal regard for both the scientist and the dancer? 

# COMMAND ----------

from transformers import pipeline, AutoTokenizer

text_generation = pipeline(
    "text-generation", model="gpt2", model_kwargs={"cache_dir": DA.paths.datasets}
)  # Note: We specify cache_dir to use a pre-cached model.

def complete_sentence(text_generation_pipeline: pipeline, prompts: list) -> list:
    """
    Via a list of prompts a prompt list is appended to by the generated `text_generation_pipeline`.
    """
    prompt_continuations = []
    for prompt in prompts:
        generation = text_generation_pipeline(
            prompt, max_length=30, do_sample=False, pad_token_id=50256
        )
        continuation = generation[0]["generated_text"].replace(prompt, "")
        prompt_continuations.append(continuation)
    return prompt_continuations

# COMMAND ----------

# MAGIC %md
# MAGIC We will now complete the sentences for the dancers.

# COMMAND ----------

dance_continuation = complete_sentence(text_generation, dance_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC Then, let's generate text for scientists.

# COMMAND ----------

science_continuation = complete_sentence(text_generation, science_prompts)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Question 3
# MAGIC Your turn to ask the model to complete sentences for each group! 

# COMMAND ----------

# TODO

group1_continuation = complete_sentence(<FILL_IN>)
group2_continuation = complete_sentence(<FILL_IN>)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion5_3(group1_continuation, group2_continuation)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have the prompts and the completion examples by GPT-2, we can evaluate the differences in regard towards both groups. 

# COMMAND ----------

import evaluate

regard = evaluate.load("regard", "compare", cache_dir=DA.paths.datasets)

# COMMAND ----------

# MAGIC %md
# MAGIC Wow, based on the `positive` regard field, we see that people in scientific occupations are regarded much more positively than those in dance (refer to the `positive` field) ! 

# COMMAND ----------

# this returns the regard scores of each string in the input list
regard.compute(data=science_continuation, references=dance_continuation)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Question 4
# MAGIC Now, compute regard score for your groups!

# COMMAND ----------

# TODO

regard.compute(data=<FILL_IN>, references=<FILL_IN>)

# COMMAND ----------

# Test your answer. DO NOT MODIFY THIS CELL.

dbTestQuestion5_4(
    regard.compute(data=group1_continuation, references=group2_continuation)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bonus: NLP Test
# MAGIC
# MAGIC To switch gears a bit, we will now turn to looking at how we can test our NLP models and see how safe and effective they are using `nlptest`. The [library](https://nlptest.org/) is developed by SparkNLP and aims to provide user-friendly APIs to help evaluate models. This library was just released in April 2023. 
# MAGIC
# MAGIC The test categories include:
# MAGIC
# MAGIC - Accuracy
# MAGIC - Bias
# MAGIC - Fairness
# MAGIC - Representation
# MAGIC - Robustness
# MAGIC
# MAGIC Currently, the library supports either `text-classification` or `ner` task.
# MAGIC
# MAGIC To start, we will use the `Harness` class to define what types of tests we would like to conduct on any given NLP model. You can read more about [Harness here](https://nlptest.org/docs/pages/docs/harness). The cell below provides a quick one-liner to show how you can evaluate the model, `dslim/bert-base-NER` from HuggingFace on a Named Entity Recognition (NER) task.
# MAGIC
# MAGIC You can choose to provide your own saved model or load existing models from `spacy` or `John Snow Labs` as well. 

# COMMAND ----------

from nlptest import Harness

# Create a Harness object
h = Harness(task="ner", model="dslim/bert-base-NER", hub="huggingface")

# COMMAND ----------

# MAGIC %md
# MAGIC We won't run the following cell since it could take up to 7 mins. This is a one-liner that runs all tests against the language model you supply. 
# MAGIC
# MAGIC Notice that it consists of three steps: 
# MAGIC 1. Generate test cases
# MAGIC 2. Run the test cases
# MAGIC 3. Generate a report of your test cases

# COMMAND ----------

# h.generate().run().report()

# COMMAND ----------

# MAGIC %md
# MAGIC If you do run `h.generate().run.report()` above, you can see that the report generates different test cases from different `test_type` and `category`. Specifically, it's unsurprising to see that the model fails the `lowercase` test for a NER use case. After all, if we lowercase all names, it would be hard to tell if the names are indeed referring to proper nouns, e.g. "the los angeles time" vs. "the Los Angeles Times".
# MAGIC
# MAGIC You can get a complete list of tests in their [documentation](https://nlptest.org/docs/pages/tests/test). For example, for `add_typo`, it checks whether the NLP model we use can handle input text with typos.

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
