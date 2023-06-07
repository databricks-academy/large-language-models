# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LLMs with Hugging Face
# MAGIC
# MAGIC In this notebook, we'll take a whirlwind tour of some top applications using Large Language Models (LLMs):
# MAGIC * Summarization
# MAGIC * Sentiment analysis
# MAGIC * Translation
# MAGIC * Zero-shot classification
# MAGIC * Few-shot learning
# MAGIC
# MAGIC We will see how existing, open-source (and proprietary) models can be used out-of-the-box for many applications.  For this, we will use [Hugging Face models](https://huggingface.co/models) and some simple prompt engineering.
# MAGIC
# MAGIC We will then look at Hugging Face APIs in more detail to understand how to configure LLM pipelines.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Use a variety of existing models for a variety of common applications.
# MAGIC 1. Understand basic prompt engineering.
# MAGIC 1. Understand search vs. sampling for LLM inference.
# MAGIC 1. Get familiar with the main Hugging Face abstractions: datasets, pipelines, tokenizers, and models.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Libraries:
# MAGIC * [sacremoses](https://github.com/alvations/sacremoses) is for the translation model `Helsinki-NLP/opus-mt-en-es`

# COMMAND ----------

# MAGIC %pip install sacremoses==0.0.53

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md ## Common LLM applications
# MAGIC
# MAGIC The goal of this section is to get your feet wet with several LLM applications and to show how easy it can be to get started with LLMs.
# MAGIC
# MAGIC As you go through the examples, note the datasets, models, APIs, and options used.  These simple examples can be starting points when you need to build your own application.

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline

# COMMAND ----------

# MAGIC %md ### Summarization
# MAGIC
# MAGIC Summarization can take two forms:
# MAGIC * `extractive` (selecting representative excerpts from the text)
# MAGIC * `abstractive` (generating novel text summaries)
# MAGIC
# MAGIC Here, we will use a model which does *abstractive* summarization.
# MAGIC
# MAGIC **Background reading**: The [Hugging Face summarization task page](https://huggingface.co/docs/transformers/tasks/summarization) lists model architectures which support summarization. The [summarization course chapter](https://huggingface.co/course/chapter7/5) provides a detailed walkthrough.
# MAGIC
# MAGIC In this section, we will use:
# MAGIC * **Data**: [xsum](https://huggingface.co/datasets/xsum) dataset, which provides a set of BBC articles and summaries.
# MAGIC * **Model**: [t5-small](https://huggingface.co/t5-small) model, which has 60 million parameters (242MB for PyTorch).  T5 is an encoder-decoder model created by Google which supports several tasks such as summarization, translation, Q&A, and text classification.  For more details, see the [Google blog post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html), [code on GitHub](https://github.com/google-research/text-to-text-transfer-transformer), or the [research paper](https://arxiv.org/pdf/1910.10683.pdf).

# COMMAND ----------

xsum_dataset = load_dataset(
    "xsum", version="1.2.0", cache_dir=DA.paths.datasets
)  # Note: We specify cache_dir to use predownloaded data.
xsum_dataset  # The printed representation of this object shows the `num_rows` of each dataset split.

# COMMAND ----------

# MAGIC %md This dataset provides 3 columns:
# MAGIC * `document`: the BBC article text
# MAGIC * `summary`: a "ground-truth" summary --> Note how subjective this "ground-truth" is.  Is this the same summary you would write?  This a great example of how many LLM applications do not have obvious "right" answers.
# MAGIC * `id`: article ID

# COMMAND ----------

xsum_sample = xsum_dataset["train"].select(range(10))
display(xsum_sample.to_pandas())

# COMMAND ----------

# MAGIC %md We next use the Hugging Face `pipeline` tool to load a pre-trained model.  In this LLM pipeline constructor, we specify:
# MAGIC * `task`: This first argument specifies the primary task.  See [Hugging Face tasks](https://huggingface.co/tasks) for more information.
# MAGIC * `model`: This is the name of the pre-trained model from the [Hugging Face Hub](https://huggingface.co/models).
# MAGIC * `min_length`, `max_length`: We want our generated summaries to be between these two token lengths.
# MAGIC * `truncation`: Some input articles may be too long for the LLM to process.  Most LLMs have fixed limits on the length of input sequences.  This option tells the pipeline to truncate the input if needed.

# COMMAND ----------

summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True,
    model_kwargs={"cache_dir": DA.paths.datasets},
)  # Note: We specify cache_dir to use predownloaded models.

# COMMAND ----------

# Apply to 1 article
summarizer(xsum_sample["document"][0])

# COMMAND ----------

# Apply to a batch of articles
results = summarizer(xsum_sample["document"])

# COMMAND ----------

# Display the generated summary side-by-side with the reference summary and original document.
# We use Pandas to join the inputs and outputs together in a nice format.
import pandas as pd

display(
    pd.DataFrame.from_dict(results)
    .rename({"summary_text": "generated_summary"}, axis=1)
    .join(pd.DataFrame.from_dict(xsum_sample))[
        ["generated_summary", "summary", "document"]
    ]
)

# COMMAND ----------

# MAGIC %md ### Sentiment analysis
# MAGIC
# MAGIC Sentiment analysis is a text classification task of estimating whether a piece of text is positive, negative, or another "sentiment" label.  The precise set of sentiment labels can vary across applications.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on text classification](https://huggingface.co/tasks/text-classification) or [Wikipedia on sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis).
# MAGIC
# MAGIC In this section, we will use:
# MAGIC * **Data**: [poem sentiment](https://huggingface.co/datasets/poem_sentiment) dataset, which provides lines from poems tagged with sentiments `negative` (0), `positive` (1), `no_impact` (2), or `mixed` (3).
# MAGIC * **Model**: [fine-tuned version of BERT](https://huggingface.co/nickwong64/bert-base-uncased-poems-sentiment).  BERT, or Bidirectional Encoder Representations from Transformers, is an encoder-only model from Google usable for 11+ tasks such as sentiment analysis and entity recognition.  For more details, see this [Hugging Face blog post](https://huggingface.co/blog/bert-101) or the [Wikipedia page](https://en.wikipedia.org/wiki/BERT_&#40;language_model&#41;).

# COMMAND ----------

poem_dataset = load_dataset(
    "poem_sentiment", version="1.0.0", cache_dir=DA.paths.datasets
)
poem_sample = poem_dataset["train"].select(range(10))
display(poem_sample.to_pandas())

# COMMAND ----------

# MAGIC %md We load the pipeline using the task `text-classification` since we want to classify text with a fixed set of labels.

# COMMAND ----------

sentiment_classifier = pipeline(
    task="text-classification",
    model="nickwong64/bert-base-uncased-poems-sentiment",
    model_kwargs={"cache_dir": DA.paths.datasets},
)

# COMMAND ----------

results = sentiment_classifier(poem_sample["verse_text"])

# COMMAND ----------

# Display the predicted sentiment side-by-side with the ground-truth label and original text.
# The score indicates the model's confidence in its prediction.

# Join predictions with ground-truth data
joined_data = (
    pd.DataFrame.from_dict(results)
    .rename({"label": "predicted_label"}, axis=1)
    .join(pd.DataFrame.from_dict(poem_sample).rename({"label": "true_label"}, axis=1))
)

# Change label indices to text labels
sentiment_labels = {0: "negative", 1: "positive", 2: "no_impact", 3: "mixed"}
joined_data = joined_data.replace({"true_label": sentiment_labels})

display(joined_data[["predicted_label", "true_label", "score", "verse_text"]])

# COMMAND ----------

# MAGIC %md ### Translation
# MAGIC
# MAGIC Translation models may be designed for specific pairs of languages, or they may support more than two languages.  We will see both below.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on translation](https://huggingface.co/tasks/translation) or the [Wikipedia page on machine translation](https://en.wikipedia.org/wiki/Machine_translation).
# MAGIC
# MAGIC In this section, we will use:
# MAGIC * **Data**: We will use some example hard-coded sentences.  However, there are a variety of [translation datasets](https://huggingface.co/datasets?task_categories=task_categories:translation&sort=downloads) available from Hugging Face.
# MAGIC * **Models**:
# MAGIC    * [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) is used for the first example of English ("en") to Spanish ("es") translation.  This model is based on [Marian NMT](https://marian-nmt.github.io/), a neural machine translation framework developed by Microsoft and other researchers.  See the [GitHub page](https://github.com/Helsinki-NLP/Opus-MT) for code and links to related resources.
# MAGIC    * [t5-small](https://huggingface.co/t5-small) model, which has 60 million parameters (242MB for PyTorch).  T5 is an encoder-decoder model created by Google which supports several tasks such as summarization, translation, Q&A, and text classification.  For more details, see the [Google blog post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html), [code on GitHub](https://github.com/google-research/text-to-text-transfer-transformer), or the [research paper](https://arxiv.org/pdf/1910.10683.pdf).  For our purposes, it supports translation for English, French, Romanian, and German.

# COMMAND ----------

# MAGIC %md Some models are designed for specific language-to-language translation.  Below, we use an English-to-Spanish model.

# COMMAND ----------

en_to_es_translation_pipeline = pipeline(
    task="translation",
    model="Helsinki-NLP/opus-mt-en-es",
    model_kwargs={"cache_dir": DA.paths.datasets},
)

# COMMAND ----------

en_to_es_translation_pipeline(
    "Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)

# COMMAND ----------

# MAGIC %md Other models are designed to handle multiple languages.  Below, we show this with `t5-small`.  Note that, since it supports multiple languages (and tasks), we give it an explicit instruction to translate from one language to another.

# COMMAND ----------

t5_small_pipeline = pipeline(
    task="text2text-generation",
    model="t5-small",
    max_length=50,
    model_kwargs={"cache_dir": DA.paths.datasets},
)

# COMMAND ----------

t5_small_pipeline(
    "translate English to French: Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)

# COMMAND ----------

t5_small_pipeline(
    "translate English to Romanian: Existing, open-source (and proprietary) models can be used out-of-the-box for many applications."
)

# COMMAND ----------

# MAGIC %md ### Zero-shot classification
# MAGIC
# MAGIC Zero-shot classification (or zero-shot learning) is the task of classifying a piece of text into one of a few given categories or labels, without having explicitly trained the model to predict those categories beforehand.  The idea appeared in literature before modern LLMs, but recent advances in LLMs have made zero-shot learning much more flexible and powerful.
# MAGIC
# MAGIC **Background reading**: See the Hugging Face [task page on zero-shot classification](https://huggingface.co/tasks/zero-shot-classification) or [Wikipedia on zero-shot learning](https://en.wikipedia.org/wiki/Zero-shot_learning).
# MAGIC
# MAGIC In this section, we will use:
# MAGIC * **Data**: a few example articles from the [xsum](https://huggingface.co/datasets/xsum) dataset used in the Summarization section above.  Our goal is to label news articles under a few categories.
# MAGIC * **Model**: [nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small), a fine-tuned version of the DeBERTa model.  The DeBERTa base model was developed by Microsoft and is one of several models derived from BERT; for more details on DeBERTa, see the [Hugging Face doc page](https://huggingface.co/docs/transformers/model_doc/deberta), the [code on GitHub](https://github.com/microsoft/DeBERTa), or the [research paper](https://arxiv.org/abs/2006.03654).

# COMMAND ----------

zero_shot_pipeline = pipeline(
    task="zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small",
    model_kwargs={"cache_dir": DA.paths.datasets},
)


def categorize_article(article: str) -> None:
    """
    This helper function defines the categories (labels) which the model must use to label articles.
    Note that our model was NOT fine-tuned to use these specific labels,
    but it "knows" what the labels mean from its more general training.

    This function then prints out the predicted labels alongside their confidence scores.
    """
    results = zero_shot_pipeline(
        article,
        candidate_labels=[
            "politics",
            "finance",
            "sports",
            "science and technology",
            "pop culture",
            "breaking news",
        ],
    )
    # Print the results nicely
    del results["sequence"]
    display(pd.DataFrame(results))

# COMMAND ----------

categorize_article(
    """
Simone Favaro got the crucial try with the last move of the game, following earlier touchdowns by Chris Fusaro, Zander Fagerson and Junior Bulumakau.
Rynard Landman and Ashton Hewitt got a try in either half for the Dragons.
Glasgow showed far superior strength in depth as they took control of a messy match in the second period.
Home coach Gregor Townsend gave a debut to powerhouse Fijian-born Wallaby wing Taqele Naiyaravoro, and centre Alex Dunbar returned from long-term injury, while the Dragons gave first starts of the season to wing Aled Brew and hooker Elliot Dee.
Glasgow lost hooker Pat McArthur to an early shoulder injury but took advantage of their first pressure when Rory Clegg slotted over a penalty on 12 minutes.
It took 24 minutes for a disjointed game to produce a try as Sarel Pretorius sniped from close range and Landman forced his way over for Jason Tovey to convert - although it was the lock's last contribution as he departed with a chest injury shortly afterwards.
Glasgow struck back when Fusaro drove over from a rolling maul on 35 minutes for Clegg to convert.
But the Dragons levelled at 10-10 before half-time when Naiyaravoro was yellow-carded for an aerial tackle on Brew and Tovey slotted the easy goal.
The visitors could not make the most of their one-man advantage after the break as their error count cost them dearly.
It was Glasgow's bench experience that showed when Mike Blair's break led to a short-range score from teenage prop Fagerson, converted by Clegg.
Debutant Favaro was the second home player to be sin-binned, on 63 minutes, but again the Warriors made light of it as replacement wing Bulumakau, a recruit from the Army, pounced to deftly hack through a bouncing ball for an opportunist try.
The Dragons got back within striking range with some excellent combined handling putting Hewitt over unopposed after 72 minutes.
However, Favaro became sinner-turned-saint as he got on the end of another effective rolling maul to earn his side the extra point with the last move of the game, Clegg converting.
Dragons director of rugby Lyn Jones said: "We're disappointed to have lost but our performance was a lot better [than against Leinster] and the game could have gone either way.
"Unfortunately too many errors behind the scrum cost us a great deal, though from where we were a fortnight ago in Dublin our workrate and desire was excellent.
"It was simply error count from individuals behind the scrum that cost us field position, it's not rocket science - they were correct in how they played and we had a few errors, that was the difference."
Glasgow Warriors: Rory Hughes, Taqele Naiyaravoro, Alex Dunbar, Fraser Lyle, Lee Jones, Rory Clegg, Grayson Hart; Alex Allan, Pat MacArthur, Zander Fagerson, Rob Harley (capt), Scott Cummings, Hugh Blake, Chris Fusaro, Adam Ashe.
Replacements: Fergus Scott, Jerry Yanuyanutawa, Mike Cusack, Greg Peterson, Simone Favaro, Mike Blair, Gregor Hunter, Junior Bulumakau.
Dragons: Carl Meyer, Ashton Hewitt, Ross Wardle, Adam Warren, Aled Brew, Jason Tovey, Sarel Pretorius; Boris Stankovich, Elliot Dee, Brok Harris, Nick Crosswell, Rynard Landman (capt), Lewis Evans, Nic Cudd, Ed Jackson.
Replacements: Rhys Buckley, Phil Price, Shaun Knight, Matthew Screech, Ollie Griffiths, Luc Jones, Charlie Davies, Nick Scott.
"""
)

# COMMAND ----------

categorize_article(
    """
The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.
Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.
Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct.
Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.
First Minister Nicola Sturgeon visited the area to inspect the damage.
The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.
Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.
However, she said more preventative work could have been carried out to ensure the retaining wall did not fail.
"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten," she said.
"That may not be true but it is perhaps my perspective over the last few days.
"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"
Meanwhile, a flood alert remains in place across the Borders because of the constant rain.
Peebles was badly hit by problems, sparking calls to introduce more defences in the area.
Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.
The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.
He said it was important to get the flood protection plan right but backed calls to speed up the process.
"I was quite taken aback by the amount of damage that has been done," he said.
"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."
He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.
Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.
"""
)

# COMMAND ----------

# MAGIC %md ### Few-shot learning
# MAGIC
# MAGIC In few-shot learning tasks, you give the model an instruction, a few query-response examples of how to follow that instruction, and then a new query.  The model must generate the response for that new query.  This technique has pros and cons: it is very powerful and allows models to be reused for many more applications, but it can be finicky and require significant prompt engineering to get good and reliable results.
# MAGIC
# MAGIC **Background reading**: See the [Wikipedia page on few-shot learning](https://en.wikipedia.org/wiki/Few-shot_learning_&#40;natural_language_processing&#41;) or [this Hugging Face blog about few-shot learning](https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api).
# MAGIC
# MAGIC In this section, we will use:
# MAGIC * **Task**: Few-shot learning can be applied to many tasks.  Here, we will do sentiment analysis, which was covered earlier.  However, you will see how few-shot learning allows us to specify custom labels, whereas the previous model was tuned for a specific set of labels.  We will also show other (toy) tasks at the end.  In terms of the Hugging Face `task` specified in the `pipeline` constructor, few-shot learning is handled as a `text-generation` task.
# MAGIC * **Data**: We use a few examples, including a tweet example from the blog post linked above.
# MAGIC * **Model**: [gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B), a version of the GPT-Neo model discussed in the blog linked above.  It is a transformer model with 1.3 billion parameters developed by Eleuther AI.  For more details, see the [code on GitHub](https://github.com/EleutherAI/gpt-neo) or the [research paper](https://arxiv.org/abs/2204.06745).

# COMMAND ----------

# We will limit the response length for our few-shot learning tasks.
few_shot_pipeline = pipeline(
    task="text-generation",
    model="EleutherAI/gpt-neo-1.3B",
    max_new_tokens=10,
    model_kwargs={"cache_dir": DA.paths.datasets},
)

# COMMAND ----------

# MAGIC %md ***Tip***: In the few-shot prompts below, we separate the examples with a special token "###" and use the same token to encourage the LLM to end its output after answering the query.  We will tell the pipeline to use that special token as the end-of-sequence (EOS) token below.

# COMMAND ----------

# Get the token ID for "###", which we will use as the EOS token below.
eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

# COMMAND ----------

# Without any examples, the model output is inconsistent and usually incorrect.
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# COMMAND ----------

# With only 1 example, the model may or may not get the answer right.
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# COMMAND ----------

# With 1 example for each sentiment, the model is more likely to understand!
results = few_shot_pipeline(
    """For each tweet, describe its sentiment:

[Tweet]: "I hate it when my phone battery dies."
[Sentiment]: Negative
###
[Tweet]: "My day has been 👍"
[Sentiment]: Positive
###
[Tweet]: "This is the link to the article"
[Sentiment]: Neutral
###
[Tweet]: "This new music video was incredible"
[Sentiment]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# COMMAND ----------

# MAGIC %md Just for fun, we show a few more examples below.

# COMMAND ----------

# The model isn't ready to serve drinks!
results = few_shot_pipeline(
    """For each food, suggest a good drink pairing:

[food]: tapas
[drink]: wine
###
[food]: pizza
[drink]: soda
###
[food]: jalapenos poppers
[drink]: beer
###
[food]: scone
[drink]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# COMMAND ----------

# This example sometimes works and sometimes does not, when sampling.  Too abstract?
results = few_shot_pipeline(
    """Given a word describing how someone is feeling, suggest a description of that person.  The description should not include the original word.

[word]: happy
[description]: smiling, laughing, clapping
###
[word]: nervous
[description]: glancing around quickly, sweating, fidgeting
###
[word]: sleepy
[description]: heavy-lidded, slumping, rubbing eyes
###
[word]: confused
[description]:""",
    eos_token_id=eos_token_id,
)

print(results[0]["generated_text"])

# COMMAND ----------

# We override max_new_tokens to generate longer answers.
# These book descriptions were taken from their corresponding Wikipedia pages.
results = few_shot_pipeline(
    """Generate a book summary from the title:

[book title]: "Stranger in a Strange Land"
[book description]: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
###
[book title]: "The Adventures of Tom Sawyer"
[book description]: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
###
[book title]: "Dune"
[book description]: "This novel is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or spice, a drug that extends life and enhances mental abilities.  The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice."
###
[book title]: "Blue Mars"
[book description]:""",
    eos_token_id=eos_token_id,
    max_new_tokens=50,
)

print(results[0]["generated_text"])

# COMMAND ----------

# MAGIC %md **Prompt engineering** is a new but critical technique for working with LLMs.  You saw some brief examples above.  As you use more general and powerful models, constructing good prompts becomes ever more important.  Some great resources to learn more are:
# MAGIC * [Wikipedia](https://en.wikipedia.org/wiki/Prompt_engineering) for a brief overview
# MAGIC * [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)
# MAGIC * [🧠 Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts) for fun examples with ChatGPT

# COMMAND ----------

# MAGIC %md ## Hugging Face APIs
# MAGIC
# MAGIC In this section, we dive into some more details on Hugging Face APIs.
# MAGIC * Search and sampling to generate text
# MAGIC * Auto* loaders for tokenizers and models
# MAGIC * Model-specific loaders
# MAGIC
# MAGIC Recall the `xsum` dataset from the **Summarization** section above:

# COMMAND ----------

display(xsum_sample.to_pandas())

# COMMAND ----------

# MAGIC %md ### Search and sampling in inference
# MAGIC
# MAGIC You may see parameters like `num_beams`, `do_sample`, etc. specified in Hugging Face pipelines.  These are inference configurations.
# MAGIC
# MAGIC LLMs work by predicting (generating) the next token, then the next, and so on.  The goal is to generate a high probability sequence of tokens, which is essentially a search through the (enormous) space of potential sequences.
# MAGIC
# MAGIC To do this search, LLMs use one of two main methods:
# MAGIC * **Search**: Given the tokens generated so far, pick the next most likely token in a "search."
# MAGIC    * **Greedy search** (default): Pick the single next most likely token in a greedy search.
# MAGIC    * **Beam search**: Greedy search can be extended via beam search, which searches down several sequence paths, via the parameter `num_beams`.
# MAGIC * **Sampling**: Given the tokens generated so far, pick the next token by sampling from the predicted distribution of tokens.
# MAGIC    * **Top-K sampling**: The parameter `top_k` modifies sampling by limiting it to the `k` most likely tokens.
# MAGIC    * **Top-p sampling**: The parameter `top_p` modifies sampling by limiting it to the most likely tokens up to probability mass `p`.
# MAGIC
# MAGIC You can toggle between search and sampling via parameter `do_sample`.
# MAGIC
# MAGIC For more background on search and sampling, see [this Hugging Face blog post](https://huggingface.co/blog/how-to-generate).
# MAGIC
# MAGIC We will illustrate these various options below using our summarization pipeline.

# COMMAND ----------

# We previously called the summarization pipeline using the default inference configuration.
# This does greedy search.
summarizer(xsum_sample["document"][0])

# COMMAND ----------

# We can instead do a beam search by specifying num_beams.
# This takes longer to run, but it might find a better (more likely) sequence of text.
summarizer(xsum_sample["document"][0], num_beams=10)

# COMMAND ----------

# Alternatively, we could use sampling.
summarizer(xsum_sample["document"][0], do_sample=True)

# COMMAND ----------

# We can modify sampling to be more greedy by limiting sampling to the top_k or top_p most likely next tokens.
summarizer(xsum_sample["document"][0], do_sample=True, top_k=10, top_p=0.8)

# COMMAND ----------

# MAGIC %md ### Auto* loaders for tokenizers and models
# MAGIC
# MAGIC We have already seen the `dataset` and `pipeline` abstractions from Hugging Face.  While a `pipeline` is a quick way to set up an LLM for a given task, the slightly lower-level abstractions `model` and `tokenizer` permit a bit more control over options.  We will show how to use those briefly, following this pattern:
# MAGIC
# MAGIC * Given input articles.
# MAGIC * Tokenize them (converting to token indices).
# MAGIC * Apply the model on the tokenized data to generate summaries (represented as token indices).
# MAGIC * Decode the summaries into human-readable text.
# MAGIC
# MAGIC We will first look at the [Auto* classes](https://huggingface.co/docs/transformers/model_doc/auto) for tokenizers and model types which can simplify loading pre-trained tokenizers and models.
# MAGIC
# MAGIC API docs:
# MAGIC * [AutoTokenizer](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer)
# MAGIC * [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSeq2SeqLM)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pre-trained tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir=DA.paths.datasets)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir=DA.paths.datasets)

# COMMAND ----------

# For summarization, T5-small expects a prefix "summarize: ", so we prepend that to each article as a prompt.
articles = list(map(lambda article: "summarize: " + article, xsum_sample["document"]))
display(pd.DataFrame(articles, columns=["prompts"]))

# COMMAND ----------

# Tokenize the input
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)
print("input_ids:")
print(inputs["input_ids"])
print("attention_mask:")
print(inputs["attention_mask"])

# COMMAND ----------

# Generate summaries
summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=2,
    min_length=0,
    max_length=40,
)
print(summary_ids)

# COMMAND ----------

# Decode the generated summaries
decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
display(pd.DataFrame(decoded_summaries, columns=["decoded_summaries"]))

# COMMAND ----------

# MAGIC %md ### Model-specific tokenizer and model loaders
# MAGIC
# MAGIC You can also more directly load specific tokenizer and model types, rather than relying on `Auto*` classes to choose the right ones for you.
# MAGIC
# MAGIC API docs:
# MAGIC * [T5Tokenizer](https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5Tokenizer)
# MAGIC * [T5ForConditionalGeneration](https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration)

# COMMAND ----------

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=DA.paths.datasets)
model = T5ForConditionalGeneration.from_pretrained(
    "t5-small", cache_dir=DA.paths.datasets
)

# COMMAND ----------

# The tokenizer and model can then be used similarly to how we used the ones loaded by the Auto* classes.
inputs = tokenizer(
    articles, max_length=1024, return_tensors="pt", padding=True, truncation=True
)
summary_ids = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    num_beams=2,
    min_length=0,
    max_length=40,
)
decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

display(pd.DataFrame(decoded_summaries, columns=["decoded_summaries"]))

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC
# MAGIC We've covered some common LLM applications and seen how to get started with them quickly using pre-trained models from the Hugging Face Hub.  We've also see how to tweak some configurations.
# MAGIC
# MAGIC But how did we find those models for our tasks?  In the lab, you will find new pre-trained models for tasks, using the Hugging Face Hub.  You will also explore tweaking model configurations to gain intuition about their effects.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
