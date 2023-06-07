# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

lesson_config.create_schema = False                 # We don't need a schema when resetting the environment
lesson_config.create_catalog = False                # Not using UC right now
lesson_config.installing_datasets = False           # We don't want to install datasets when resetting the environment

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_learning_environment()                     # Once initialized, reset the entire learning environment

