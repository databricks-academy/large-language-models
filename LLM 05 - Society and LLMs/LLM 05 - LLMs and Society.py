# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LLMs and Society
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Learn representation bias in training data 
# MAGIC 1. Use Hugging Face to calculate toxicity score
# MAGIC 1. Use SHAP to generate explanation on model output
# MAGIC 1. Learn the latest state of research in model explanation: contrastive explanation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %pip install disaggregators==0.1.2 https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examining representation bias in Wikipedia biographies
# MAGIC
# MAGIC [Disaggregators](https://github.com/huggingface/disaggregators) is a library developed by Hugging Face. As the name implies, it "dis-aggregates" data so that we can explore the data in more granular detail and evaluate data bias.
# MAGIC
# MAGIC There are multiple disaggregation modules available: 
# MAGIC - age
# MAGIC - gender
# MAGIC - religion
# MAGIC - continent
# MAGIC - pronoun
# MAGIC
# MAGIC We will be loading Wikipedia bios as our datasets to analyze later. We will be using the `pronoun` module since it takes the least amount of time to dis-aggregate. You are welcome to try out other modules in your own time. 
# MAGIC
# MAGIC **DISCLAIMER**: 
# MAGIC - Warning: Some content may be triggering.
# MAGIC - The models developed or used in this course are for demonstration and learning purposes only. Models may occasionally output offensive, inaccurate, biased information, or harmful instructions.
# MAGIC
# MAGIC
# MAGIC
# MAGIC **IMPORTANT**:
# MAGIC - For `gender` disaggregator to work, you need to download spacy's `en_core_web_lg` model. 
# MAGIC   - That's the model Hugging Face is using behind the scene! 
# MAGIC   - Hence, you can see the `.whl` file install in the `%pip install` command above. 
# MAGIC   - The model is directly download from [spaCy's GitHub](https://github.com/explosion/spacy-models/releases?q=en_core_web_lg).
# MAGIC

# COMMAND ----------

from disaggregators import Disaggregator

disaggregator = Disaggregator("pronoun", column="target_text")
# disaggregator = Disaggregator("gender", column="target_text")
# disaggregator = Disaggregator("continent", column="target_text")
# disaggregator = Disaggregator("religion", column="target_text")
# disaggregator = Disaggregator("age", column="target_text")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We will use [Wikipedia biographies dataset](https://huggingface.co/datasets/wiki_bio), `wiki_bio`, readily available in the Hugging Face Datasets. From the dataset summary, the data contains the first paragraph of the biography and the tabular infobox. 
# MAGIC
# MAGIC As you see, disaggreator works with Hugging Face datasets or any datasets where `.map` can be invoked. The `disaggregators` library attempts to group the wiki bio into `she_her`, `he_him`, and `they_them`.
# MAGIC
# MAGIC Note: the cell below might take a couple minutes for the data to finish loading and for the disaggregator to categorize the data.

# COMMAND ----------

from datasets import load_dataset

wiki_data = load_dataset(
    "wiki_bio", split="test", cache_dir=DA.paths.datasets
)  # Note: We specify cache_dir to use pre-cached data.
ds = wiki_data.map(disaggregator)
pdf = ds.to_pandas()

# COMMAND ----------

# Let's take a look at the dataframe
pdf

# COMMAND ----------

# MAGIC %md
# MAGIC However, it doesn't do a very a good job at determining `they_them` as it seems to classify mentions of physical objects as `they_them` as well. For example, the 19th row has both pronoun categories, `they_them` and `he_him`, to be true. But looking at the data itself, we saw that the bio only used the the pronoun `he_him`: 
# MAGIC
# MAGIC  >william ` bill ' rigby -lrb- 9 june 1921 - 1 june 2010 -rrb- was a former english footballer who played as a goalkeeper .\nhe was born in chester .\na product of the youth system at his hometown club of chester , rigby made his only peacetime first-team appearance for the club in their first post-war match in the football league in a 4 -- 4 draw at york city on 31 august 1946 .\nafter this he was not selected again , with goalkeeping duties being passed on to george scales and jim maclaren .\nearlier he had made appearances for the first-team during the war years , mainly during 1940 -- 41 and 1941 -- 42 while understudy to bill shortt
# MAGIC
# MAGIC For this reason, the following analysis will ignore the column `pronoun.they_them`.

# COMMAND ----------

import json

print(pdf.iloc[[19], :].to_json(indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do a simple aggregation to check the ratio of Wikipedian bios in terms of `he_him` 

# COMMAND ----------

import numpy as np

she_array = np.where(pdf["pronoun.she_her"] == True)
print(f"she_her: {len(she_array[0])} rows")
he_array = np.where(pdf["pronoun.he_him"] == True)
print(f"he_him: {len(he_array[0])} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC The `he_him` pronoun represents `44004/ (9545+44004)`, which is approximately 82% of the data! It is not hard to imagine that models trained on predominantly male data would exhibit bias towards males.
# MAGIC
# MAGIC Let's confirm that existing pre-trained models, like BERT, does exhibit bias. BERT is trained on both Wikipedia and [books that are adapted into movies](https://huggingface.co/datasets/bookcorpus). 

# COMMAND ----------

from transformers import pipeline

unmasker = pipeline(
    "fill-mask",
    model="bert-base-uncased",
    model_kwargs={"cache_dir": DA.paths.datasets},
)  # Note: We specify cache_dir to use pre-cached models.

# COMMAND ----------

# MAGIC %md
# MAGIC To probe what BERT outputs, we will intentionally insert [MASK] token and ask BERT to generate words to replace that [MASK] token.

# COMMAND ----------

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect toxicity
# MAGIC
# MAGIC Now that we have inspected data and model bias above, let's evaluate the toxicity of language model outputs. To do this, we leverage another [Hugging Face library called `evaluate`](https://huggingface.co/docs/evaluate/index).
# MAGIC
# MAGIC The `evaluate` library can measure language models from different angles:
# MAGIC <br>
# MAGIC - Toxicity: how problematic the completion is, such as hate speech
# MAGIC   - It uses [Facebook's `roberta-hate-speech-dynabench-r4-target` model](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) behind the scene to compute toxicity.
# MAGIC - [HONEST](https://huggingface.co/spaces/evaluate-measurement/honest): how hurtful the completion is 
# MAGIC   - The model was [published in 2021](https://aclanthology.org/2021.naacl-main.191.pdf)
# MAGIC   - It works very similarly as our `unmasker` example in the cell directly above. It also replaces certain words with [MASK] tokens and evaluates the hurtfulness based on what the language models output.
# MAGIC - Regard: whether the completion regards a certain group higher than the others 
# MAGIC   - You will play with this in the lab! So we will save this for later. 

# COMMAND ----------

import evaluate

toxicity = evaluate.load("toxicity", module_type="measurement")

# COMMAND ----------

# MAGIC %md
# MAGIC Any toxicity value over 0.5 is arbitrarily defined as "toxic". Here, we refrain from typing literal curse words to increase the toxicity values. However, you can see that the third phrase is noticeably more toxic than the other two! 

# COMMAND ----------

candidates = [
    "their kid loves reading books",
    "she curses and makes fun of people",
    "he is a wimp and pathetic loser",
]
toxicity.compute(predictions=candidates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Interpretability with SHAP 
# MAGIC
# MAGIC Another interesting topic within language model evaluation is whether we can interpret LM outputs. **SH**apley **A**dditive ex**P**lanations (**SHAP**) is a popular approach to explain the output of a machine learning model. It is agnostic to the type of machine learning model you pass in; this means that we can try using SHAP to explain our language model outputs! 
# MAGIC
# MAGIC See the <a href="http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions" target="_blank">SHAP NeurIPS</a> paper for details, and Christoph Molnar's book chapter on <a href="https://christophm.github.io/interpretable-ml-book/shapley.html" target="_blank">Shapley Values</a>. 
# MAGIC
# MAGIC Take the diagram below as an example. SHAP's goal is to explain the $10,000 difference in the apartment price. We see that if cats are not allowed in the same apartment building, the price of the apartment is lower than if it were to allow cats. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/shap_permutation.png" width=500>
# MAGIC
# MAGIC Image is sourced from Molnar's book. Read SHAP [documentation here](https://shap.readthedocs.io/en/latest/text_examples.html).

# COMMAND ----------

# MAGIC %md
# MAGIC In this section, we are going to first load a text generation model from Hugging Face, provide an input sentence, and ask the model to complete the rest of the sentence. Then, we will ask SHAP to generate explanation behind the sentence completion.

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import shap

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", use_fast=True, cache_dir=DA.paths.datasets
)
model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=DA.paths.datasets)

# Set model decoder to true
# GPT is a decoder-only model
model.config.is_decoder = True
# We set configurations for the output text generation
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0,  # to turn off randomness
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}

# COMMAND ----------

# MAGIC %md
# MAGIC Feel free to modify the input sentence below to play around later!

# COMMAND ----------

input_sentence = ["Sunny days are the best days to go to the beach. So"]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The `shap.Explainer` is how we can interface with SHAP. We need to pass in our `tokenizer` because that's the tokenizer we use to vectorize the text. When SHAP masks certain tokens to generate explanation, the tokenizer helps us to retain the same number of tokens by replacing the word with the [MASK] token.

# COMMAND ----------

explainer = shap.Explainer(model, tokenizer)
shap_values = explainer(input_sentence)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can check the contribution of each input token towards the output token. "Red" means positive contribution whereas "blue" means negative indication. The color intensity indicates the strength of the contribution. 
# MAGIC
# MAGIC From the documentation:
# MAGIC
# MAGIC > The base value is what the model outputs when the entire input text is masked, while f_outputclass(inputs)
# MAGIC  is the output of the model for the full original input. The SHAP values explain in an additive way how the impact of unmasking each word changes the model output from the base value (where the entire input is masked) to the final prediction value

# COMMAND ----------

shap.plots.text(shap_values)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC The plot below shows which input tokens contributes most towards the output token `looking`. 

# COMMAND ----------

shap.plots.bar(shap_values[0, :, "looking"])

# COMMAND ----------

input_sentence2 = ["I know many people who prefer beaches to the mountains"]
shap_values2 = explainer(input_sentence2)
shap.plots.text(shap_values2)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check which token contributes the most to the output word "not".

# COMMAND ----------

shap.plots.bar(shap_values2[0, :, "not"])

# COMMAND ----------

# MAGIC %md
# MAGIC Common model interpretability methods for text classification are not as informative for language model predictions because the most recent input token usually is the most influential token to the subsequent predicted token. This is called recency bias and it's a difficult problem to tackle. While SHAP gave us an idea what input token may have contributed to the output token, it's not really all that useful. 
# MAGIC
# MAGIC Let's take a look at the final example.

# COMMAND ----------

input_sentence3 = ["Can you stop the dog from"]
shap_values3 = explainer(input_sentence3)
shap.plots.text(shap_values3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In the example above, we see that the predicted token is `barking`. But we don't know why the model doesn't output tokens like `crying`, `eating`, `biting`, etc. It would be a lot more interesting if we can know *why* the model outputs `barking` **instead of** `crying` and other viable word candidates. This `instead of` explanation is called `contrastive explanation` ([Yin and Neubig 2022](https://aclanthology.org/2022.emnlp-main.14.pdf)). 
# MAGIC
# MAGIC
# MAGIC Let the actual output token be `target_token` and the viable output token be `foil_token`. Intuitively, there are three methods to generate such contrastive explanations: 
# MAGIC 1. Calculate how much an input token influences the probability of `target_token`, while decreasing the probability of `foil_token`
# MAGIC 2. Calculate how much erasing an input token increases the probability of `foil_token` and reduces that of `target_token` 
# MAGIC 3. Calculate the dot product between the input token embedding and the output token. 
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/constrastive_exp.png" width=300>
# MAGIC
# MAGIC Courtesy of the author's, Kayo Yin's, [slides](https://kayoyin.github.io/assets/slides/melb22.pdf). Below, we are going to use Yin's [Python module](https://github.com/kayoyin/interpret-lm/tree/main) to generate contrastive explanation for us! The code is currently in a research state, rather than readily packaged on PyPI or production-ready, but it is still interesting to see current (and potential future) state of research directions.
# MAGIC
# MAGIC We will walk through the results directly in the markdown. If you are interested in running this code, you can download `lm_saliency.py` from the [repo](https://github.com/kayoyin/interpret-lm/blob/main/lm_saliency.py) and import it into your `LLM 05 - LLMs and Society` folder. 
# MAGIC
# MAGIC ```
# MAGIC from transformers import GPT2Tokenizer, GPT2LMHeadModel
# MAGIC
# MAGIC gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
# MAGIC     "gpt2",
# MAGIC     cache_dir=DA.paths.datasets)
# MAGIC gpt2_model = GPT2LMHeadModel.from_pretrained(
# MAGIC     "gpt2",
# MAGIC     cache_dir=DA.paths.datasets)
# MAGIC
# MAGIC input_tokens = gpt2_tokenizer(input_sentence3[0])["input_ids"]
# MAGIC attention_ids = gpt2_tokenizer(input_sentence3[0])["attention_mask"]
# MAGIC ```
# MAGIC
# MAGIC Recall that `input_sentence3[0]` is `Can you stop the dog from`.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the erasure method to generate explanation. 
# MAGIC
# MAGIC ```
# MAGIC import lm_saliency
# MAGIC from lm_saliency import *
# MAGIC
# MAGIC target = "barking" # target refers to the word we would like to generate explanation on
# MAGIC foil = "crying" # foil refers to any other possible word 
# MAGIC explanation = "erasure"
# MAGIC CORRECT_ID = gpt2_tokenizer(" " + target)["input_ids"][0]
# MAGIC FOIL_ID = gpt2_tokenizer(" " + foil)["input_ids"][0]
# MAGIC
# MAGIC # Erasure
# MAGIC base_explanation = erasure_scores(gpt2_model, input_tokens, attention_ids, normalize=True)
# MAGIC contra_explanation = erasure_scores(gpt2_model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)
# MAGIC
# MAGIC visualize(np.array(base_explanation), gpt2_tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target}?")
# MAGIC visualize(np.array(contra_explanation), gpt2_tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target} instead of {foil}?")
# MAGIC ```
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/llm/lm_saliency.png" width=1000>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC The score measures how much each token influences the model to attribute a higher probability to the target token. In this example above, `stop` makes the model more likely to predict `barking` whereas `the` doesn't influence whether the model predicts `barking` or `crying`.  
# MAGIC
# MAGIC
# MAGIC How we can use contrastive explanation to improve LLMs is still an ongoing research! It's not surprising that the research so far has shown that contrastive explanation can help us characterize how LLMs decide which output token to predict. It's an exciting space to watch for development! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
