# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluating Large Language Models (LLMs)
# MAGIC This notebook demonstrates methods for evaluating LLMs.  We focus on the task of summarization and cover accuracy, ROUGE-N, and perplexity.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Know how to compute ROUGE-N and other metrics.
# MAGIC 2. Gain an intuitive understanding of ROUGE-N.
# MAGIC 3. Test various models and model sizes on the same data, and compare their results.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %pip install rouge_score==0.1.2

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## How can we evaluate summarization?
# MAGIC
# MAGIC Suppose you are developing a smartphone news app and need to display automatically generated summaries of breaking news articles.  How can you evaluate whether or not the summaries you are generating are good?
# MAGIC
# MAGIC ![](https://drive.google.com/uc?export=view&id=1V6cMD1LgivCb850JDhva1DO9EWVH8rJ7)

# COMMAND ----------

# MAGIC %md ## Dataset
# MAGIC
# MAGIC We will use a subset of the `cnn_dailymail` dataset from See et al., 2017, downloadable from the [Hugging Face `datasets` hub](https://huggingface.co/datasets/cnn_dailymail).
# MAGIC
# MAGIC This dataset provides news article paired with summaries (in the "highlights" column).  Let's load the data and take a look at some examples.

# COMMAND ----------

import torch
from datasets import load_dataset

full_dataset = load_dataset(
    "cnn_dailymail", "3.0.0", cache_dir=DA.paths.datasets
)  # Note: We specify cache_dir to use pre-cached data.

# Use a small sample of the data during this lab, for speed.
sample_size = 100
sample = (
    full_dataset["train"]
    .filter(lambda r: "CNN" in r["article"][:25])
    .shuffle(seed=42)
    .select(range(sample_size))
)
sample

# COMMAND ----------

display(sample.to_pandas())

# COMMAND ----------

example_article = sample["article"][0]
example_summary = sample["highlights"][0]
print(f"Article:\n{example_article}\n")
print(f"Summary:\n{example_summary}")

# COMMAND ----------

# MAGIC %md ## Summarization

# COMMAND ----------

import pandas as pd
import torch
import gc
from transformers import AutoTokenizer, T5ForConditionalGeneration

# COMMAND ----------

def batch_generator(data: list, batch_size: int):
    """
    Creates batches of size `batch_size` from a list.
    """
    s = 0
    e = s + batch_size
    while s < len(data):
        yield data[s:e]
        s = e
        e = min(s + batch_size, len(data))


def summarize_with_t5(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Compute summaries using a T5 model.
    This is similar to a `pipeline` for a T5 model but does tokenization manually.

    :param model_checkpoint: Name for a model checkpoint in Hugging Face, such as "t5-small" or "t5-base"
    :param articles: List of strings, where each string represents one article.
    :return: List of strings, where each string represents one article's generated summary
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = T5ForConditionalGeneration.from_pretrained(
        model_checkpoint, cache_dir=DA.paths.datasets
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, model_max_length=1024, cache_dir=DA.paths.datasets
    )

    def perform_inference(batch: list) -> list:
        inputs = tokenizer(
            batch, max_length=1024, return_tensors="pt", padding=True, truncation=True
        )

        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=40,
        )
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    res = []

    summary_articles = list(map(lambda article: "summarize: " + article, articles))
    for batch in batch_generator(summary_articles, batch_size=batch_size):
        res += perform_inference(batch)

        torch.cuda.empty_cache()
        gc.collect()

    # clean up
    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return res

# COMMAND ----------

t5_small_summaries = summarize_with_t5("t5-small", sample["article"])

# COMMAND ----------

reference_summaries = sample["highlights"]

# COMMAND ----------

display(
    pd.DataFrame.from_dict(
        {
            "generated": t5_small_summaries,
            "reference": reference_summaries,
        }
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC You may see some warning messages in the output above.  While pipelines are handy, they provide less control over the tokenizer and model; we will dive deeper later.
# MAGIC
# MAGIC But first, let's see how our summarization pipeline does!  We'll compute 0/1 accuracy, a classic ML evaluation metric.

# COMMAND ----------

accuracy = 0.0
for i in range(len(reference_summaries)):
    generated_summary = t5_small_summaries[i]
    if generated_summary == reference_summaries[i]:
        accuracy += 1.0
accuracy = accuracy / len(reference_summaries)

print(f"Achieved accuracy {accuracy}!")

# COMMAND ----------

# MAGIC %md Accuracy zero?!?  We can see that the (very generic) metric of 0/1 accuracy is not useful for summarization.  Thinking about this more, small variations in wording may not matter much, and many different summaries may be equally valid.  So how can we evaluate summarization?

# COMMAND ----------

# MAGIC %md ## ROUGE
# MAGIC
# MAGIC Now that we can generate summaries---and we know 0/1 accuracy is useless here---let's look at how we can compute a meaningful metric designed to evaluate summarization: ROUGE.
# MAGIC
# MAGIC Recall-Oriented Understudy for Gisting Evaluation (ROUGE) is a set of evaluation metrics designed for comparing summaries from Lin et al., 2004.  See [Wikipedia](https://en.wikipedia.org/wiki/ROUGE_&#40;metric&#41;) for more info.  Here, we use the Hugging Face Evaluator wrapper to call into the `rouge_score` package.  This package provides 4 scores:
# MAGIC
# MAGIC * `rouge1`: ROUGE computed over unigrams (single words or tokens)
# MAGIC * `rouge2`: ROUGE computed over bigrams (pairs of consecutive words or tokens)
# MAGIC * `rougeL`: ROUGE based on the longest common subsequence shared by the summaries being compared
# MAGIC * `rougeLsum`: like `rougeL`, but at "summary level," i.e., ignoring sentence breaks (newlines)

# COMMAND ----------

import evaluate
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

rouge_score = evaluate.load("rouge")

# COMMAND ----------

# MAGIC %md You can call `rouge_score` evaluator directly, but we provide a convenience function below to handle the expected input format.

# COMMAND ----------

def compute_rouge_score(generated: list, reference: list) -> dict:
    """
    Compute ROUGE scores on a batch of articles.

    This is a convenience function wrapping Hugging Face `rouge_score`,
    which expects sentences to be separated by newlines.

    :param generated: Summaries (list of strings) produced by the model
    :param reference: Ground-truth summaries (list of strings) for comparison
    """
    generated_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in generated]
    reference_with_newlines = ["\n".join(sent_tokenize(s.strip())) for s in reference]
    return rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
    )

# COMMAND ----------

# ROUGE scores for our batch of articles
compute_rouge_score(t5_small_summaries, reference_summaries)

# COMMAND ----------

# MAGIC %md ## Understanding ROUGE scores

# COMMAND ----------

# Sanity check: What if our predictions match the references exactly?
compute_rouge_score(reference_summaries, reference_summaries)

# COMMAND ----------

# And what if we fail to predict anything?
compute_rouge_score(
    generated=["" for _ in range(len(reference_summaries))],
    reference=reference_summaries,
)

# COMMAND ----------

# MAGIC %md Stemming predictions and references can help to ignore minor differences.
# MAGIC
# MAGIC We will use `rouge_score.compute()` directly for these hand-constructed examples.

# COMMAND ----------

rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large language models beating world records"],
    use_stemmer=False,
)

# COMMAND ----------

rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large language models beating world records"],
    use_stemmer=True,
)

# COMMAND ----------

# MAGIC %md Let's look at how the ROUGE score behaves in various situations.

# COMMAND ----------

# What if we predict exactly 1 word correctly?
rouge_score.compute(
    predictions=["Large language models beat world record"],
    references=["Large"],
    use_stemmer=True,
)

# COMMAND ----------

# The ROUGE score is symmetric with respect to predictions and references.
rouge_score.compute(
    predictions=["Large"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)

# COMMAND ----------

# What about 2 words?  Note how 'rouge1' and 'rouge2' compare with the case when we predict exactly 1 word correctly.
rouge_score.compute(
    predictions=["Large language"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)

# COMMAND ----------

# Note how rouge1 differs from the rougeN (N>1) scores when we predict word subsequences correctly.
rouge_score.compute(
    predictions=["Models beat large language world record"],
    references=["Large language models beat world record"],
    use_stemmer=True,
)

# COMMAND ----------

# MAGIC  %md ## Compare small and large models
# MAGIC
# MAGIC  We've been working with the `t5-small` model so far.  Let's compare several models with different architectures in terms of their ROUGE scores and some example generated summaries.

# COMMAND ----------

def compute_rouge_per_row(
    generated_summaries: list, reference_summaries: list
) -> pd.DataFrame:
    """
    Generates a dataframe to compare rogue score metrics.
    """
    generated_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in generated_summaries
    ]
    reference_with_newlines = [
        "\n".join(sent_tokenize(s.strip())) for s in reference_summaries
    ]
    scores = rouge_score.compute(
        predictions=generated_with_newlines,
        references=reference_with_newlines,
        use_stemmer=True,
        use_aggregator=False,
    )
    scores["generated"] = generated_summaries
    scores["reference"] = reference_summaries
    return pd.DataFrame.from_dict(scores)

# COMMAND ----------

# MAGIC %md ### T5-small
# MAGIC
# MAGIC The [T5](https://huggingface.co/docs/transformers/model_doc/t5) [[paper]](https://arxiv.org/pdf/1910.10683.pdf) family of models are text-to-text transformers that have been trained on a multi-task mixture of unsupervised and supervised tasks. They are well suited for task such as summarization, translation, text classification, question answering, and more.
# MAGIC
# MAGIC The t5-small version of the T5 models has 60 million parameters.

# COMMAND ----------

# We computed t5_small_summaries above already.
compute_rouge_score(t5_small_summaries, reference_summaries)

# COMMAND ----------

t5_small_results = compute_rouge_per_row(
    generated_summaries=t5_small_summaries, reference_summaries=reference_summaries
)
display(t5_small_results)

# COMMAND ----------

# MAGIC %md ### T5-base
# MAGIC
# MAGIC The [T5-base](https://huggingface.co/t5-base) model has 220 million parameters.

# COMMAND ----------

t5_base_summaries = summarize_with_t5(
    model_checkpoint="t5-base", articles=sample["article"]
)
compute_rouge_score(t5_base_summaries, reference_summaries)

# COMMAND ----------

t5_base_results = compute_rouge_per_row(
    generated_summaries=t5_base_summaries, reference_summaries=reference_summaries
)
display(t5_base_results)

# COMMAND ----------

# MAGIC %md ### GPT-2
# MAGIC
# MAGIC The [GPT-2](https://huggingface.co/gpt2) model is a generative text model that was trained in a self-supervised fashion. Its strengths are in using a 'completing the sentence' for a given prompt.  It has 124 million parameters.

# COMMAND ----------

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def summarize_with_gpt2(
    model_checkpoint: str, articles: list, batch_size: int = 8
) -> list:
    """
    Convenience function for summarization with GPT2 to handle these complications:
    - Append "TL;DR" to the end of the input to get GPT2 to generate a summary.
    https://huggingface.co/course/chapter7/5?fw=pt
    - Truncate input to handle long articles.
    - GPT2 uses a max token length of 1024.  We use a shorter 512 limit here.

    :param model_checkpoint: reference to checkpointed model
    :param articles: list of strings
    :return: generated summaries, with the input and "TL;DR" removed
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_checkpoint, padding_side="left", cache_dir=DA.paths.datasets
    )
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model = GPT2LMHeadModel.from_pretrained(
        model_checkpoint,
        pad_token_id=tokenizer.eos_token_id,
        cache_dir=DA.paths.datasets,
    ).to(device)

    def perform_inference(batch: list) -> list:
        tmp_inputs = tokenizer(
            batch, max_length=500, return_tensors="pt", padding=True, truncation=True
        )
        tmp_inputs_decoded = tokenizer.batch_decode(
            tmp_inputs.input_ids, skip_special_tokens=True
        )
        inputs = tokenizer(
            [article + " TL;DR:" for article in tmp_inputs_decoded],
            max_length=512,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        summary_ids = model.generate(
            inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            num_beams=2,
            min_length=0,
            max_length=512 + 32,
        )
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    decoded_summaries = []
    for batch in batch_generator(articles, batch_size=batch_size):
        decoded_summaries += perform_inference(batch)

        # batch clean up
        torch.cuda.empty_cache()
        gc.collect()

    # post-process decoded summaries
    summaries = [
        summary[summary.find("TL;DR:") + len("TL;DR: ") :]
        for summary in decoded_summaries
    ]

    # cleanup
    del tokenizer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return summaries

# COMMAND ----------

gpt2_summaries = summarize_with_gpt2(
    model_checkpoint="gpt2", articles=sample["article"]
)
compute_rouge_score(gpt2_summaries, reference_summaries)

# COMMAND ----------

gpt2_results = compute_rouge_per_row(
    generated_summaries=gpt2_summaries, reference_summaries=reference_summaries
)
display(gpt2_results)

# COMMAND ----------

# MAGIC %md ### Comparing all models
# MAGIC
# MAGIC We use a couple of helper functions to compare the above models, first by their evaluation metrics (quantitative) and second by their generated summaries (qualitative).

# COMMAND ----------

def compare_models(models_results: dict) -> pd.DataFrame:
    """
    :param models_results: dict of "model name" string mapped to pd.DataFrame of results computed by `compute_rouge_per_row`
    :return: pd.DataFrame with 1 row per model, with columns: model, rouge1, rouge2, rougeL, rougeLsum
    where metrics are averages over input results for each model
    """
    agg_results = []
    for r in models_results:
        model_results = models_results[r].drop(
            labels=["generated", "reference"], axis=1
        )
        agg_metrics = [r]
        agg_metrics[1:] = model_results.mean(axis=0)
        agg_results.append(agg_metrics)
    return pd.DataFrame(
        agg_results, columns=["model", "rouge1", "rouge2", "rougeL", "rougeLsum"]
    )

# COMMAND ----------

display(
    compare_models(
        {
            "t5-small": t5_small_results,
            "t5-base": t5_base_results,
            "gpt2": gpt2_results,
        }
    )
)

# COMMAND ----------

def compare_models_summaries(models_summaries: dict) -> pd.DataFrame:
    """
    Aggregates results from `models_summaries` and returns a dataframe.
    """
    comparison_df = None
    for model_name in models_summaries:
        summaries_df = models_summaries[model_name]
        if comparison_df is None:
            comparison_df = summaries_df[["generated"]].rename(
                {"generated": model_name}, axis=1
            )
        else:
            comparison_df = comparison_df.join(
                summaries_df[["generated"]].rename({"generated": model_name}, axis=1)
            )
    return comparison_df

# COMMAND ----------

# In the output table below, scroll to the right to see all models.
display(
    compare_models_summaries(
        {
            "t5_small": t5_small_results,
            "t5_base": t5_base_results,
            "gpt2": gpt2_results,
        }
    )
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
