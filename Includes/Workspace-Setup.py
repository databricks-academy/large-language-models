# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Workspace Setup
# MAGIC Instructors should run this notebook to prepare the workspace for a class.
# MAGIC
# MAGIC This creates or updates the following resources:
# MAGIC
# MAGIC |Resource Type|Description|
# MAGIC |---|---|
# MAGIC |User Entitlements|User-specific grants to allow creating databases/schemas against the current catalog when they are not workspace-admins.|
# MAGIC |Instance Pool | **DBAcademy** for use by students and the "student" and "jobs" policies|
# MAGIC |Cluster Policies| **DBAcademy** for clusters running standard notebooks <br> **DBAcademy Jobs** for workflows/jobs <br> **DBAcademy DLT** for DLT piplines (automatically applied)|
# MAGIC |Shared SQL Warehouse|**DBAcademy Warehouse** for Databricks SQL exercises|

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

# Start a timer so we can benchmark execution duration.
setup_start = dbgems.clock_start()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Class Config Parameters
# MAGIC Sets up the following widgets to collect parameters used to configure our environment as a means of controlling class cost.
# MAGIC
# MAGIC - **Configure For** (required) - **All Users**, **Missing Users Only**, or **Current User Only**
# MAGIC - **Description** (optional) - a general purpose description of the class
# MAGIC - **Lab/Class ID** (optional) - **lab_id** is the name assigned to this event/class or alternatively its class number
# MAGIC - **Spark Version** (optional) - **spark_version** is the "preloaded" and thus default version of the Databricks Runtime

# COMMAND ----------

from dbacademy.dbhelper import WorkspaceHelper

# Setup the widgets to collect required parameters.
dbutils.widgets.dropdown("configure_for", WorkspaceHelper.CONFIGURE_FOR_ALL_USERS, 
                         [WorkspaceHelper.CONFIGURE_FOR_ALL_USERS], "Configure For (required)")

# lab_id is the name assigned to this event/class or alternatively its class number
dbutils.widgets.text(WorkspaceHelper.PARAM_LAB_ID, "", "Lab/Class ID (optional)")

# a general purpose description of the class
dbutils.widgets.text(WorkspaceHelper.PARAM_DESCRIPTION, "", "Description (optional)")

# The default spark version
dbutils.widgets.text(WorkspaceHelper.PARAM_SPARK_VERSION, "11.3.x-cpu-ml-scala2.12", "Spark Version (optional)")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Run Init Script & Install Datasets
# MAGIC Main purpose of the next cell is to pre-install the datasets.
# MAGIC
# MAGIC It has the side effect of create our DA object, which includes our REST client.

# COMMAND ----------

lesson_config.create_catalog = False                 # We don't need a schema when configuring the workspace
lesson_config.create_schema = False                 # We don't need a schema when configuring the workspace

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

from dbacademy.dbhelper import ClustersHelper

org_id = dbgems.get_org_id()
lab_id = WorkspaceHelper.get_lab_id() or "UNKNOWN"
spark_version = WorkspaceHelper.get_spark_version()
workspace_name = WorkspaceHelper.get_workspace_name()
workspace_description = WorkspaceHelper.get_workspace_description() or "UNKNOWN"

print(f"org_id:                {org_id}")
print(f"lab_id:                {lab_id}")
print(f"spark_version:         {spark_version}")
print(f"workspace_name:        {workspace_name}")
print(f"workspace_description: {workspace_description}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Class Instance Pools
# MAGIC The following cell configures the instance pool used for this class

# COMMAND ----------

instance_pool_id = DA.workspace.clusters.create_instance_pool(preloaded_spark_version=spark_version,
                                                              org_id=org_id, 
                                                              lab_id=lab_id, 
                                                              workspace_name=workspace_name, 
                                                              workspace_description=workspace_description)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create The Three Class-Specific Cluster Policies
# MAGIC The following cells create the various cluster policies used by the class

# COMMAND ----------

# org_id, lab_id, workspace_name and workspace_description are attached to the
# instance pool and as such, they are not attached to the all-purpose or jobs policies.

ClustersHelper.create_all_purpose_policy(client=DA.client, 
                                         instance_pool_id=instance_pool_id, 
                                         spark_version=spark_version,
                                         autotermination_minutes_max=180,
                                         autotermination_minutes_default=120)

ClustersHelper.create_jobs_policy(client=DA.client, 
                                  instance_pool_id=instance_pool_id, 
                                  spark_version=spark_version)

ClustersHelper.create_dlt_policy(client=DA.client, 
                                 org_id=org_id, 
                                 lab_id=lab_id, 
                                 workspace_name=workspace_name, 
                                 workspace_description=workspace_description)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create Class-Shared Databricks SQL Warehouse/Endpoint
# MAGIC Creates a single wharehouse to be used by all students.
# MAGIC
# MAGIC The configuration is derived from the number of students specified above.

# COMMAND ----------

from dbacademy.dbhelper.warehouses_helper_class import WarehousesHelper

DA.workspace.warehouses.create_shared_sql_warehouse(name=WarehousesHelper.WAREHOUSES_DEFAULT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configure User Entitlements
# MAGIC
# MAGIC This task simply adds the "**databricks-sql-access**" entitlement to the "**users**" group ensuring that they can access the Databricks SQL view.

# COMMAND ----------

WorkspaceHelper.add_entitlement_workspace_access(client=DA.client)
WorkspaceHelper.add_entitlement_databricks_sql_access(client=DA.client)

# COMMAND ----------

print(f"Setup completed {dbgems.clock_stopped(setup_start)}")

