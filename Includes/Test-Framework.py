# Databricks notebook source
print("Importing lab testing framework.")

# COMMAND ----------

# DEFINING HELPER FUNCTIONS

def createDirStructure():
  '''
  This creates the directories needed for the test handler.
  Note that `lesson_question_d` is lesson num: num of questions.
    Modify this when changing the number of questions
  '''
  from pathlib import Path

  lesson_question_d = {
    1: 3, # TODO: confirm these questions once tests are finalized 
    2: 6,
    3: 4,
    4: 9,
    5: 5,
  }
  path = getUsernameFromEnv("")

  for lesson, questions in lesson_question_d.items():
    for question in range(1, questions+1):
      final_path = f"{path}lesson{lesson}/question{question}"
      Path(final_path).mkdir(parents=True, exist_ok=True)

def questionPassed(userhome_for_testing, lesson, question):
  '''
  Helper function that writes an empty file named `PASSED` to the designated path
  '''
  from pathlib import Path

  print(f"\u001b[32mPASSED\x1b[0m: All tests passed for {lesson}, {question}")

  path = f"{userhome_for_testing}/{question}"
  Path(path).mkdir(parents=True, exist_ok=True)
  with open(f"{path}/PASSED", "wb") as handle:
      pass # just write an empty file
  
  print ("\u001b[32mRESULTS RECORDED\x1b[0m: Click `Submit` when all questions are completed to log the results.")

def getUsernameFromEnv(lesson):
  '''
  Exception handling for when the working directory is not in the scope
  (i.e. the Classroom-Setup was not run)
  '''
  try:
    return f"{DA.paths.working_dir}-testing-files/{lesson}"
  except NameError:
    raise NameError("Working directory not found. Please re-run the Classroom-Setup at the beginning of the notebook.")

createDirStructure()

# COMMAND ----------

# LLM 01L - LLMs with Hugging Face Lab

def dbTestQuestion1_1(summarizer, summarization_results, summarizer_inputs):
  lesson, question = "lesson1", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(summarizer.task) == "summarization", "Test NOT passed: Pipeline should be built for task `summarization`"
  assert isinstance(summarization_results, list), "Test NOT passed: Result should be a list."
  assert len(summarization_results) == len(summarizer_inputs), "Test NOT passed: Result should be a list of length equal to the input dataset size."
  assert min([len(s) for s in summarization_results]) > 0, "Test NOT passed: Summaries should be non-empty."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_2(translation_pipeline, translation_results, translation_inputs):
  lesson, question = "lesson1", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert "translation" in str(translation_pipeline.task), "Test NOT passed: Pipeline should be built for task `translation`"
  assert isinstance(translation_results, list), "Test NOT passed: Result should be a list."
  assert len(translation_results) == len(translation_inputs), "Test NOT passed: Result should be a list of length equal to the input dataset size."
  assert min([len(s) for s in translation_results]) > 0, "Test NOT passed: Translations should be non-empty."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion1_3(few_shot_pipeline, few_shot_prompt, few_shot_results):
  lesson, question = "lesson1", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert isinstance(few_shot_prompt, str), "Test NOT passed: Prompt should be a string."
  assert isinstance(few_shot_results, str), "Test NOT passed: Results should be a string."
  assert len(few_shot_prompt) > 0, "Test NOT passed: Prompt should be non-empty."
  assert few_shot_results.find(few_shot_prompt) == 0, "Test NOT passed: Results should be prefixed by the prompt."
  assert len(few_shot_results) > len(few_shot_prompt), "Test NOT passed: Results should include new text, beyond the prompt."

  questionPassed(userhome_for_testing, lesson, question)

# COMMAND ----------

# LLM 02L - Embeddings, Vector Databases, and Search

def dbTestQuestion2_1(collection_name):
  lesson, question = "lesson2", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  collection_name=="my_talks", "Test NOT passed: The collection_name should be my_talks." 

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_2(talks_collection):
  lesson, question = "lesson2", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  str(type(talks_collection)) == "<class 'chromadb.api.models.Collection.Collection'>", "Test NOT passed: Result should be of type `chromadb.api.models.Collection.Collection`"

  assert talks_collection.count() > 0, "Test NOT passed: The collection should be non-empty."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_3(results):
  lesson, question = "lesson2", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  len(results) > 0, "Test NOT passed: The result must be non-empty, check `query_texts` and `n_results`"

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion2_4(pipe):
  lesson, question = "lesson2", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  str(type(pipe)) == "<class 'transformers.pipelines.text_generation.TextGenerationPipeline'>", "Test NOT passed: Result should be of type `transformers.pipelines.text_generation.TextGenerationPipeline`"

  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion2_5(_question, context, prompt_template):
  # using _question given that `question` is reserved for `questionPassed`
  lesson, question = "lesson2", "question5"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert isinstance(_question, str), "Test NOT passed: `question` should be a `str` type."
  assert isinstance(context, str), "Test NOT passed: `context` should be a `str` type."
  assert _question in prompt_template, "Test NOT passed: Your `question` should appear inside the prompt." 
  assert context in prompt_template, "Test NOT passed: Your `context` should appear inside the prompt." 

  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion2_6(lm_response):
  lesson, question = "lesson2", "question6"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert lm_response[0]["generated_text"] is not None, "Test NOT passed:  `lm_response` should not be empty" 

  questionPassed(userhome_for_testing, lesson, question) 

# COMMAND ----------

# LLM 03L - Building LLM Chains Lab

def dbTestQuestion3_1(embeddings, docsearch):
  lesson, question = "lesson3", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(embeddings)) == "<class 'langchain.embeddings.huggingface.HuggingFaceEmbeddings'>", "Test NOT passed: Result is not of type `langchain.embeddings.huggingface.HuggingFaceEmbeddings`"
  assert str(type(docsearch)) == "<class 'langchain.vectorstores.chroma.Chroma'>", "Test NOT passed: Result is not of type `langchain.vectorstores.chroma.Chroma`"
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion3_2(qa, query_results_hamlet):
  lesson, question = "lesson3", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(qa)) == "<class 'langchain.chains.retrieval_qa.base.RetrievalQA'>", "Test NOT passed: Result is not of type `langchain.chains.retrieval_qa.base.RetrievalQA`"
  assert type(query_results_hamlet) == str, "Test NOT passed: Query results not a string"
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion3_3(qa, query_results_venice):
  lesson, question = "lesson3", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(qa)) == "<class 'langchain.chains.retrieval_qa.base.RetrievalQA'>", "Test NOT passed: Result is not of type `langchain.chains.retrieval_qa.base.RetrievalQA`"
  assert type(query_results_venice) == str, "Test NOT passed: Query results not a string"
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion3_4(qa, query_results_romeo):
  lesson, question = "lesson3", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(qa)) == "<class 'langchain.chains.retrieval_qa.base.RetrievalQA'>", "Test NOT passed: Result is not of type `langchain.chains.retrieval_qa.base.RetrievalQA`"
  assert type(query_results_romeo) == str, "Test NOT passed: Query results not a string"
  
  questionPassed(userhome_for_testing, lesson, question)


# COMMAND ----------

# LLM 04L - Fine-tuning LLMs

def dbTestQuestion4_1(ds):
  lesson, question = "lesson4", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(ds.keys()) == "dict_keys(['train'])", "Test NOT passed: `ds` should be of type `datasets.dataset_dict.DatasetDict`"
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion4_2(model_checkpoint):
  lesson, question = "lesson4", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  model_checkpoint == "EleutherAI/pythia-70m-deduped", "Test NOT passed: `model_checkpoint` should be `EleutherAI/pythia-70m-deduped`."
  
  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion4_3(tokenizer):
  lesson, question = "lesson4", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(tokenizer)) == "<class 'transformers.models.gpt_neox.tokenization_gpt_neox_fast.GPTNeoXTokenizerFast'>", "Test NOT passed: `tokenizer` is not of type `transformers.models.gpt_neox.tokenization_gpt_neox_fast.GPTNeoXTokenizerFast`"
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_4(tokenized_dataset):
  lesson, question = "lesson4", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert str(type(tokenized_dataset)) == "<class 'datasets.dataset_dict.DatasetDict'>", "Test NOT passed: `tokenized_dataset` should be of type `datasets.dataset_dict.DatasetDict`"
  assert  len(tokenized_dataset["train"]["input_ids"][0]) == len(tokenized_dataset["train"]["attention_mask"][0]), "Test NOT passed: For each entry the number of `input_ids` and `attention_masks` should be equal"
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_5(training_args):
  lesson, question = "lesson4", "question5"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert training_args.num_train_epochs == 10, "Test NOT passed: `num_train_epochs` should be 10."
  assert str(type(training_args.optim)) == "<enum 'OptimizerNames'>", "Test NOT passed: `optim` should be of type `OptimizerNames`."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_6(model):
  lesson, question = "lesson4", "question6"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert model.base_model_prefix == "gpt_neox", "Test NOT passed: `base_model_prefix should be `gpt_neox`, reload your model checkpoint."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_7(trainer):
  lesson, question = "lesson4", "question7"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert trainer.train_dataset.num_rows == 6000, "Test NOT passed: The number of rows in the training data is not equal to `TRAINING_SIZE`."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_8(trainer):
  lesson, question = "lesson4", "question8"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert trainer.state.epoch == 10.0, "Test NOT passed: make sure to run your training for 10 epochs exactly."
  
  questionPassed(userhome_for_testing, lesson, question) 

def dbTestQuestion4_9(rouge_scores):
  lesson, question = "lesson4", "question9"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert type(rouge_scores) == dict, "Test NOT passed: `rouge_scores should be a dict, check your scoring answer."
  
  questionPassed(userhome_for_testing, lesson, question) 

# COMMAND ----------

# #LLM 05L - LLMs and Society Lab

def dbTestQuestion5_1(group1_bold, group2_bold):
  lesson, question = "lesson5", "question1"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert  isinstance(group1_bold, list), "Test NOT passed: `group1_bold` should be of type list."
  assert  isinstance(group2_bold, list), "Test NOT passed: `group2_bold` should be of type list."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion5_2(group1_prompts, group2_prompts):
  lesson, question = "lesson5", "question2"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert isinstance(group1_prompts, list), "Test NOT passed: `group1_prompts` should be of type list."
  assert isinstance(group2_prompts, list), "Test NOT passed: `group2_prompts` should be of type list."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion5_3(group1_continuation, group2_continuation):
  lesson, question = "lesson5", "question3"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert isinstance(group1_continuation, list), "Test NOT passed: `group1_continuation` should be of type list."
  assert isinstance(group2_continuation, list), "Test NOT passed: `group2_continuation` should be of type list."

  questionPassed(userhome_for_testing, lesson, question)

def dbTestQuestion5_4(regard_score):
  lesson, question = "lesson5", "question4"
  userhome_for_testing = getUsernameFromEnv(lesson)

  assert isinstance(regard_score, dict), "Test NOT passed: The regard score should be of type dictionary." 

  questionPassed(userhome_for_testing, lesson, question) 

