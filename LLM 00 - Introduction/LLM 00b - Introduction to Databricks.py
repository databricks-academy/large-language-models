# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Databricks Platform
# MAGIC
# MAGIC Demonstrate basic functionality and identify terms related to working in the Databricks workspace.
# MAGIC
# MAGIC
# MAGIC ##### Objectives
# MAGIC 1. Create a new cell
# MAGIC 1. Execute code in multiple languages
# MAGIC 1. Create markdown cells
# MAGIC 1. Read data from DBFS (Databricks File System)
# MAGIC 1. Visualize data
# MAGIC 1. Install libraries
# MAGIC
# MAGIC ##### Databricks Notebook Utilities
# MAGIC - Example <a href="https://docs.databricks.com/notebooks/notebooks-use.html#language-magic" target="_blank">magic commands</a>: **`%python`**, **`%sql`**, **`%md`**, **`%fs`**, **`%sh`**, **`%pip`**

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Setup
# MAGIC Run classroom setup to copy Databricks training datasets into your environment.
# MAGIC
# MAGIC Use the **`%run`** magic command to run another notebook within a notebook
# MAGIC
# MAGIC To run the notebook cell below click on the cell containing the `%run` command, this selects the cell, and then push `Shift + Enter` on your keyboard. Cells can also be run by clicking the arrow in the top right corner of the cell (the arrow will appear when hovering your mouse over the cell).

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a new Cell
# MAGIC
# MAGIC Notebook cells can be created by clicking the "`+`" button that appears when you hover your mouse between two cells, or by using keyboard shortcuts. To use the keyboard shortcuts select any cell and press `A` to insert a cell above the selected cell, or `B` to insert a cell below the selected cell.

# COMMAND ----------

# Try creating a cell below me! Click on my cell (not in the text area itself) and then press `B`.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Execute code in multiple languages
# MAGIC
# MAGIC Databricks notbooks support 4 different languages : <a href="https://www.python.org/" target="_blank">Python</a>, <a href="https://www.scala-lang.org/" target="_blank">Scala</a>, <a href="https://en.wikipedia.org/wiki/SQL" target="_blank">SQL</a>, and <a href="https://www.r-project.org/" target="_blank">R</a>. Upon creation of a notebook you'll set a default language for the cells in the notebook to use. The default language of the notebook is displayed in the upper left of your window, to the right of the notebook name.
# MAGIC
# MAGIC
# MAGIC * Each notebook specifies a default language, in this case **Python**.
# MAGIC * Run the cell below using one of the following options:
# MAGIC   * **CTRL+ENTER** or **CMD+RETURN**
# MAGIC   * **SHIFT+ENTER** or **SHIFT+RETURN** to run the cell and move to the next one
# MAGIC   * Using **Run Cell**, **Run All Above** or **Run All Below** as seen here<br/><img style="box-shadow: 5px 5px 5px 0px rgba(0,0,0,0.25); border: 1px solid rgba(0,0,0,0.25);" src="https://files.training.databricks.com/images/notebook-cell-run-cmd.png"/>
# MAGIC
# MAGIC The below cell shows an example of a python command executing.

# COMMAND ----------

print("Run default language")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC But you can also use non-default languages in your notebooks as well! Try running languages specified by their language magic commands: **`%python`**, **`%scala`**, **`%sql`**, **`%r`**.
# MAGIC
# MAGIC Below are examples of using magic commands to execute code in **python** and **sql**:

# COMMAND ----------

# MAGIC %python
# MAGIC print("Run python")

# COMMAND ----------

# MAGIC %sql
# MAGIC select "Run SQL"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Create documentation cells
# MAGIC Render cell as <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">Markdown</a> using the magic command: **`%md`**
# MAGIC
# MAGIC Below are some examples of how you can use Markdown to format documentation. Click this cell and press **`Enter`** to view the underlying Markdown syntax.
# MAGIC
# MAGIC
# MAGIC # Heading 1
# MAGIC ### Heading 3
# MAGIC > block quote
# MAGIC
# MAGIC 1. **bold**
# MAGIC 2. *italicized*
# MAGIC 3. ~~strikethrough~~
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC - <a href="https://www.markdownguide.org/cheat-sheet/" target="_blank">link</a>
# MAGIC - `code`
# MAGIC
# MAGIC ```
# MAGIC {
# MAGIC   "message": "This is a code block",
# MAGIC   "method": "https://www.markdownguide.org/extended-syntax/#fenced-code-blocks",
# MAGIC   "alternative": "https://www.markdownguide.org/basic-syntax/#code-blocks"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ![Spark Logo](https://files.training.databricks.com/images/Apache-Spark-Logo_TM_200px.png)
# MAGIC
# MAGIC | Element         | Markdown Syntax |
# MAGIC |-----------------|-----------------|
# MAGIC | Heading         | `#H1` `##H2` `###H3` `#### H4` `##### H5` `###### H6` |
# MAGIC | Block quote     | `> blockquote` |
# MAGIC | Bold            | `**bold**` |
# MAGIC | Italic          | `*italicized*` |
# MAGIC | Strikethrough   | `~~strikethrough~~` |
# MAGIC | Horizontal Rule | `---` |
# MAGIC | Code            | ``` `code` ``` |
# MAGIC | Link            | `[text](https://www.example.com)` |
# MAGIC | Image           | `![alt text](image.jpg)`|
# MAGIC | Ordered List    | `1. First items` <br> `2. Second Item` <br> `3. Third Item` |
# MAGIC | Unordered List  | `- First items` <br> `- Second Item` <br> `- Third Item` |
# MAGIC | Code Block      | ```` ``` ```` <br> `code block` <br> ```` ``` ````|
# MAGIC | Table           |<code> &#124; col &#124; col &#124; col &#124; </code> <br> <code> &#124;---&#124;---&#124;---&#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br> <code> &#124; val &#124; val &#124; val &#124; </code> <br>|

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Reading data
# MAGIC
# MAGIC When you ran the **Setup** cell at the top of the notebook, some variables were created for you. One of the variables is `DA.paths.datasets` which is the path to datasets which will be used during this course.
# MAGIC
# MAGIC One such dataset is located at **`{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv`**. Let's use `pandas` to read that csv file.

# COMMAND ----------

import pandas as pd

# Specify the location of the csv file
csv_location = f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv"
# Read the dataset
newscatcher = pd.read_csv(csv_location, sep=";")
# Display the datset
newscatcher

# COMMAND ----------

# MAGIC %md
# MAGIC We can now use `matplotlib` to plot aggregate data from our dataset.

# COMMAND ----------

import matplotlib.pyplot as plt

# Count how many articles exist per topic
newscatcher_counts_by_topic = (
    newscatcher
    .loc[:,["topic","title"]]
    .groupby("topic")
    .agg("count")
    .reset_index(drop=False)
)

# Create a bar plot
plt.bar(newscatcher_counts_by_topic["topic"],height=newscatcher_counts_by_topic["title"])
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The `display()` command will pretty-print a large variety of data types, including Apache Spark DataFrames or Pandas DataFrames.
# MAGIC
# MAGIC It will also allow you to make visualizations without writing additional code. For example, after executing the below command click the `+` icon in the results to add a Visualization. Select the **Bar** visualization type and click "Save".

# COMMAND ----------

display(newscatcher_counts_by_topic)

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks Runtime (DBR) environments come with many pre-installed libraries (for example, <a href="https://docs.databricks.com/release-notes/runtime/13.1ml.html#python-libraries-on-cpu-clusters"  target="_blank">DBR 13.1 python libraries</a>), but sometimes you'll want to install some additional ones.
# MAGIC
# MAGIC Additional libraries can be installed directly onto your cluster in the **Compute** tab, or you can install them with a scope specific to your individual notebook using the `%pip` magic command.
# MAGIC
# MAGIC Because sometimes you'll need to restart your python kernel after installing a new library via `%pip` it's considered best practice to put all `%pip` commands at the very top of your notebook.

# COMMAND ----------

# MAGIC %pip install nlptest==1.1.0

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can import the newly installed `nlptest` package.

# COMMAND ----------

import nlptest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Learning More
# MAGIC
# MAGIC We like to encourage you to explore the documentation to learn more about the various features of the Databricks platform and notebooks.
# MAGIC * <a href="https://docs.databricks.com/user-guide/index.html" target="_blank">User Guide</a>
# MAGIC * <a href="https://docs.databricks.com/user-guide/notebooks/index.html" target="_blank">User Guide / Notebooks</a>
# MAGIC * <a href="https://docs.databricks.com/administration-guide/index.html" target="_blank">Administration Guide</a>
# MAGIC * <a href="https://docs.databricks.com/release-notes/index.html" target="_blank">Release Notes</a>
# MAGIC * <a href="https://docs.databricks.com/" target="_blank">And much more!</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
