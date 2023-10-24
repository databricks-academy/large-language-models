# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Run the Universal-Workspace-Setup Job
# MAGIC This will dispatch to the UWS to provide a standard configuration for all labs.
# MAGIC
# MAGIC <span style="font-weight:bold; color:red">WARNING</span>: Running this notebook is <b>no longer required</b> and may actually break your learning environment if executed. This is included here only for testing purposes.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

from dbacademy.dbhelper.universal_worskpace_setup_runner import UniversalWorkspaceSetupRunner

runner = UniversalWorkspaceSetupRunner(
    token=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None),
    endpoint=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None),
    course_config=course_config,
    workspace_name=sc.getConf().get("spark.databricks.workspaceUrl", defaultValue="Unknown")
)

runner.run()

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Datasets
# MAGIC The current version of the UWS won't install user-specific datasets yest so this is a hack to force it to install before moving to round #2 tests that will install the user-specific datasets, but fail the tests due to their asyncrounus nature.

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

