# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Fine-tuning LLMs
# MAGIC  Many LLMs are general purpose models trained on a broad range of data and use cases. This enables them to perform well in a variety of applications, as shown in previous modules. It is not uncommon though to find situations where applying a general purpose model performs unacceptably for specific dataset or use case. This often does not mean that the general purpose model is unusable. Perhaps, with some new data and additional training the model could be improved, or fine-tuned, such that it produces acceptable results for the specific use case.
# MAGIC  
# MAGIC  Fine-tuning uses a pre-trained model as a base and continues to train it with a new, task targeted dataset. Conceptually, fine-tuning leverages that which has already been learned by a model and aims to focus its learnings further for a specific task.
# MAGIC
# MAGIC  It is important to recognize that fine-tuning is model training. The training process remains a resource intensive, and time consuming effort. Albeit fine-tuning training time is greatly shortened as a result of having started from a pre-trained model. The model training process can be accelerated through the use of tools like Microsoft's [DeepSpeed](https://github.com/microsoft/DeepSpeed).
# MAGIC
# MAGIC  This notebook will explore how to perform fine-tuning at scale.
# MAGIC
# MAGIC ### ![Dolly](https://files.training.databricks.com/images/llm/dolly_small.png) Learning Objectives
# MAGIC 1. Prepare a novel dataset
# MAGIC 1. Fine-tune the `t5-small` model to classify movie reviews.
# MAGIC 1. Leverage DeepSpeed to enhance training process.

# COMMAND ----------

assert "gpu" in spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"), "THIS LAB REQUIRES THAT A GPU MACHINE AND RUNTIME IS UTILIZED."

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Later sections of this notebook will leverage the DeepSpeed package. DeepSpeed has some additional dependencies that need to be installed in the Databricks environment. The dependencies vary based upon which MLR runtime is being used. The below commands add the necessary libraries accordingly. It is convenient to perform this step at the start of the Notebook to avoid future restarts of the Python kernel.

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /tmp/externals/cuda
# MAGIC
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.50-1_amd64.deb -P /tmp/externals/cuda
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -P /tmp/externals/cuda
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -P /tmp/externals/cuda
# MAGIC wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -P /tmp/externals/cuda
# MAGIC
# MAGIC dpkg -i /tmp/externals/cuda/libcurand-dev-11-7_10.2.10.50-1_amd64.deb
# MAGIC dpkg -i /tmp/externals/cuda/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb
# MAGIC dpkg -i /tmp/externals/cuda/libcublas-dev-11-7_11.10.1.25-1_amd64.deb
# MAGIC dpkg -i /tmp/externals/cuda/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install deepspeed==0.9.1 py-cpuinfo==9.0.0

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Working Directory: {DA.paths.working_dir}")

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a local temporary directory on the Driver. This will serve as a root directory for the intermediate model checkpoints created during the training process. The final model will be persisted to DBFS.

# COMMAND ----------

import tempfile

tmpdir = tempfile.TemporaryDirectory()
local_training_root = tmpdir.name

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tuning

# COMMAND ----------

import os
import pandas as pd
import transformers as tr
from datasets import load_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 - Data Preparation
# MAGIC
# MAGIC The first step of the fine-tuning process is to identify a specific task and supporting dataset. In this notebook, we will consider the specific task to be classifying movie reviews. This idea is generally simple task where a movie review is provided as plain-text and we would like to determine whether or not the review was positive or negative.
# MAGIC
# MAGIC The [IMDB dataset](https://huggingface.co/datasets/imdb) can be leveraged as a supporting dataset for this task. The dataset conveniently provides both a training and testing dataset with labeled binary sentiments, as well as a dataset of unlabeled data.

# COMMAND ----------

imdb_ds = load_dataset("imdb")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 - Select pre-trained model
# MAGIC
# MAGIC The next step of the fine-tuning process is to select a pre-trained model. We will consider using the [T5](https://huggingface.co/docs/transformers/model_doc/t5) [[paper]](https://arxiv.org/pdf/1910.10683.pdf) family of models for our fine-tuning purposes. The T5 models are text-to-text transformers that have been trained on a multi-task mixture of unsupervised and supervised tasks. They are well suited for tasks such as summarization, translation, text classification, question answering, and more.
# MAGIC
# MAGIC The `t5-small` version of the T5 models has 60 million parameters. This slimmed down version will be sufficient for our purposes.

# COMMAND ----------

model_checkpoint = "t5-small"

# COMMAND ----------

# MAGIC %md
# MAGIC Recall from Module 1, Hugging Face provides the [Auto*](https://huggingface.co/docs/transformers/model_doc/auto) suite of objects to conveniently instantiate the various components associated with a pre-trained model. Here, we use the [AutoTokenizer](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer) to load in the tokenizer that is associated with the `t5-small` model.

# COMMAND ----------

# load the tokenizer that was used for the t5-small model
tokenizer = tr.AutoTokenizer.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)  # Use a pre-cached model

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC As mentioned above, the IMDB dataset is a binary sentiment dataset. Its labels therefore are encoded as (-1 - unknown; 0 - negative; 1 - positive) values. In order to use this dataset with a text-to-text model like T5, the label set needs to be represented as a string. There are a number of ways to accomplish this. Here, we will simply translate each label id to its corresponding string value.

# COMMAND ----------

def to_tokens(
    tokenizer: tr.models.t5.tokenization_t5_fast.T5TokenizerFast, label_map: dict
) -> callable:
    """
    Given a `tokenizer` this closure will iterate through `x` and return the result of `apply()`.
    This function is mapped to a dataset and returned with ids and attention mask.
    """

    def apply(x) -> tr.tokenization_utils_base.BatchEncoding:
        """From a formatted dataset `x` a batch encoding `token_res` is created."""
        target_labels = [label_map[y] for y in x["label"]]
        token_res = tokenizer(
            x["text"],
            text_target=target_labels,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        return token_res

    return apply


imdb_label_lookup = {0: "negative", 1: "positive", -1: "unknown"}

# COMMAND ----------

imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(
    imdb_to_tokens, batched=True, remove_columns=["text", "label"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 - Setup Training
# MAGIC
# MAGIC The model training process is highly configurable. The [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) class effectively exposes the configurable aspects of the process allowing one to customize them accordingly. Here, we will focus on setting up a training process that performs a single epoch of training with a batch size of 16. We will also leverage `adamw_torch` as the optimizer.

# COMMAND ----------

checkpoint_name = "test-trainer"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    local_checkpoint_path,
    num_train_epochs=1,  # default number of epochs to train is 3
    per_device_train_batch_size=16,
    optim="adamw_torch",
    report_to=["tensorboard"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC The pre-trained `t5-small` model can be loaded using the [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) class.

# COMMAND ----------

# load the pre-trained model
model = tr.AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)  # Use a pre-cached model

# COMMAND ----------

# used to assist the trainer in batching the data
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 - Train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Before starting the training process, let's turn on Tensorboard. This will allow us to monitor the training process as checkpoint logs are created.

# COMMAND ----------

tensorboard_display_dir = f"{local_checkpoint_path}/runs"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

# MAGIC %md
# MAGIC Start the fine-tuning process.

# COMMAND ----------

trainer.train()

# save model to the local checkpoint
trainer.save_model()
trainer.save_state()

# COMMAND ----------

# persist the fine-tuned model to DBFS
final_model_path = f"{DA.paths.working_dir}/llm04_fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 - Predict

# COMMAND ----------

fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

# COMMAND ----------

reviews = [
    """
'Despicable Me' is a cute and funny movie, but the plot is predictable and the characters are not very well-developed. Overall, it's a good movie for kids, but adults might find it a bit boring.""",
    """ 'The Batman' is a dark and gritty take on the Caped Crusader, starring Robert Pattinson as Bruce Wayne. The film is a well-made crime thriller with strong performances and visuals, but it may be too slow-paced and violent for some viewers.
""",
    """
The Phantom Menace is a visually stunning film with some great action sequences, but the plot is slow-paced and the dialogue is often wooden. It is a mixed bag that will appeal to some fans of the Star Wars franchise, but may disappoint others.
""",
    """
I'm not sure if The Matrix and the two sequels were meant to have a tigh consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say.
""",
]
inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True)
pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)

# COMMAND ----------

pdf = pd.DataFrame(
    zip(reviews, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DeepSpeed
# MAGIC
# MAGIC As model architectures evolve and grow, they continually push the limits of available computational resources. For example, some large LLMs having hundreds of billions of parameters making them too large to fit, in some cases, in available GPU memory. Models of this scale therefore need to leverage distributed processing or high-end hardware, and sometimes even both, to support training efforts. This makes large model training a costly undertaking, and therefore accelerating the training process is highly desirable.
# MAGIC
# MAGIC As mentioned above, one such framework that can be leveraged to accelerate the model training process is Microsoft's [DeepSpeed](https://github.com/microsoft/DeepSpeed) [[paper]](https://arxiv.org/pdf/2207.00032.pdf). This framework provides advances in compression, distributed training, mixed precision, gradient accumulation, and checkpointing.
# MAGIC
# MAGIC It is worth noting that DeepSpeed is intended for large models that do not fit into device memory. The `t5-base` model we are using is not a large model, and therefore DeepSpeed is not expected to provide a benefit.
# MAGIC
# MAGIC ### !! Please do not attempt this in Vocareum as it will take more than 5 hours to run and exhaust your compute budget!!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Environment Setup
# MAGIC
# MAGIC The intended use for DeepSpeed is in a distributed compute environment. As such, each node of the environment is assigned a `rank` and `local_rank` in relation to the size of the distributed environment.
# MAGIC
# MAGIC Here, since we are testing with a single node/GPU environment we will set the `world_size` to 1, and both `ranks` to 0.

# COMMAND ----------

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration
# MAGIC
# MAGIC There are a number of [configuration options](https://www.deepspeed.ai/docs/config-json/) that can be set to enhance the training and inference processes. The [ZeRO optimization](https://www.deepspeed.ai/training/#memory-efficiency) settings target reducing the memory footprint allowing for larger models to be efficiently trained on limited resources. 
# MAGIC
# MAGIC The Hugging Face `TrainerArguments` accept the configuration either from a JSON file or a dictionary. Here, we will define the dictionary. 

# COMMAND ----------

zero_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto"
}

# COMMAND ----------

model_checkpoint = "t5-base"
tokenizer = tr.AutoTokenizer.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)

imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(
    imdb_to_tokens, batched=True, remove_columns=["text", "label"]
)

model = tr.AutoModelForSeq2SeqLM.from_pretrained(
    model_checkpoint, cache_dir=DA.paths.datasets
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC There are only two changes made to the training setup from above. The first is to set a new checkpoint name. The second is to add the `deepspeed` configuration to the `TrainingArguments`.
# MAGIC
# MAGIC Note: at this time the `deepspeed` argument is considered an experimental feature and may evolve in the future.

# COMMAND ----------

checkpoint_name = "test-trainer-deepspeed"
checkpoint_location = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    checkpoint_location,
    num_train_epochs=3,  # default number of epochs to train is 3
    per_device_train_batch_size=8,
    deepspeed=zero_config,  # add the deepspeed configuration
    report_to=["tensorboard"],
)

data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# COMMAND ----------

tensorboard_display_dir = f"{checkpoint_location}/runs"

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

trainer.train()

trainer.save_model()
trainer.save_state()

# COMMAND ----------

# persist the fine-tuned model to DBFS
final_model_path = f"{DA.paths.working_dir}/llm04_fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict

# COMMAND ----------

fine_tuned_model = tr.AutoModelForSeq2SeqLM.from_pretrained(final_model_path)

# COMMAND ----------

review = [
    """
           I'm not sure if The Matrix and the two sequels were meant to have a tight consistency but I don't think they quite fit together. They seem to have a reasonably solid arc but the features from the first aren't in the second and third as much, instead the second and third focus more on CGI battles and more visuals. I like them but for different reasons, so if I'm supposed to rate the trilogy I'm not sure what to say."""
]
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True)

pred = fine_tuned_model.generate(
    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
)

# COMMAND ----------

pdf = pd.DataFrame(
    zip(review, tokenizer.batch_decode(pred, skip_special_tokens=True)),
    columns=["review", "classification"],
)
display(pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

tmpdir.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
