# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

lesson_config.create_schema = False                 # We don't need a schema when simply printing the copyrights

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

# Once initialized, just print the copyrights
DA.print_copyrights()


