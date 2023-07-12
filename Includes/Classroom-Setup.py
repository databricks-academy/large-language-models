# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.paths.working_dir = DA.paths.to_vm_path(DA.paths.working_dir)
DA.paths.datasets = DA.paths.to_vm_path(DA.paths.datasets)
DA.paths.user_db = DA.paths.to_vm_path(DA.paths.user_db)

# COMMAND ----------

# MAGIC %run ./Test-Framework

# COMMAND ----------

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe models developed or used in this course are for demonstration and learning purposes only.\nModels may occasionally output offensive, inaccurate, biased information, or harmful instructions.")

