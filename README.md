## Large Language Models

This repo contains the notebooks and slides for the [Large Language Models: Application through Production](https://www.edx.org/course/large-language-models-application-through-production) course on [edX](https://www.edx.org/professional-certificate/databricks-large-language-models) & Databricks Academy.
 
<details>
<summary> Notebooks</summary>
 
 ## How to Import the Repo into Databricks?

1. You first need to add Git credentials to Databricks. Refer to [documentation here](https://docs.databricks.com/repos/repos-setup.html#add-git-credentials-to-databricks).  

2. Click `Repos` in the sidebar. Click `Add Repo` on the top right.
    
    <img width="400" alt="repo_1" src="https://files.training.databricks.com/images/llm/repo_1.png">

    

3. Clone the "HTTPS" URL from GitHub, or copy `https://github.com/databricks-academy/large-language-models.git` and paste into the box `Git repository URL`. The rest of the fields, i.e. `Git provider` and `Repository name`, will be automatically populated. Click `Create Repo` on the bottom right. 

    <img width="700" alt="add_repo" src="https://files.training.databricks.com/images/llm/add_repo.png">

 ## How to Import the files from `.dbc` releases on GitHub
1. You can download the notebooks from a release by navigating to the releases section on the GitHub page:
 
    <img width="400" alt="dbc_release1" src="https://files.training.databricks.com/images/llm/dbc_release1.png">
 
2. From the releases page, download the `.dbc` file. This contains all of the course notebooks, with the structure and meta data. 
 
    <img width="400" alt="dbc_release2" src="https://files.training.databricks.com/images/llm/dbc_release2.png">

3. In your Databricks workspace, navigate to the Workspace menu, click on Home and select `Import`:
 
    <img width="400" alt="dbc_release3" src="https://files.training.databricks.com/images/llm/dbc_release3.png">

4. Using the import tool, navigate to the location on your computer where the `.dbc` file was dowloaded from Step 1. Once you select the file, click `Import`, and the files will be loaded and extracted to your workspace:
 
    <img width="400" alt="dbc_release4" src="https://files.training.databricks.com/images/llm/dbc_release4.png">



</details>

<details>
 <summary> Cluster settings </summary>
 
## Which Databricks cluster should I use? 

1. First, select `Single Node` 

    <img width="500" alt="single_node" src="https://files.training.databricks.com/images/llm/single_node.png">


2. This courseware has been tested on [Databricks Runtime 13.3 LTS for Machine Learning]([url](https://docs.databricks.com/en/release-notes/runtime/13.3lts-ml.html)). If you do not have access to a 13.3 LTS ML Runtime cluster, you will need to install many additional libraries (as the ML Runtime pre-installs many commonly used machine learning packages), and this courseware is not guaranteed to run. 
    
    <img width="400" alt="cluster" src="https://github.com/databricks-academy/large-language-models/assets/6416014/50dd3080-97d7-40ff-9eda-b91a359fa4ac">


    
    For all of the notebooks except `LLM 04a - Fine-tuning LLMs` and `LLM04L - Fine-tuning LLMs Lab`, you can run them on a CPU just fine. We recommend either `i3.xlarge` or `i3.2xlarge` (i3.2xlarge will have slightly faster performance).  

    <img width="400" alt="cpu_settings" src="https://github.com/databricks-academy/large-language-models/assets/6416014/4c8f6e92-0400-4aba-9107-27b911dd11c1">
    
    For these notebooks: `LLM 04a - Fine-tuning LLMs` and `LLM04L - Fine-tuning LLMs Lab`, you will need the Databricks Runtime 13.3 LTS for Machine Learning **with GPU**. 

    <img width="400" alt="gpu" src="https://github.com/databricks-academy/large-language-models/assets/6416014/2580d6da-f3a5-4562-9b4e-f0f4861e7c23">

    
    Select GPU instance type of `g5.2xlarge`.

    <img width="400" alt="gpu_settings" src="https://github.com/databricks-academy/large-language-models/assets/6416014/3934f739-458b-40db-8d96-02d5d274f58e">
</details>

<details>
 <summary> Install datasets and models </summary>
 
## How do I install the datasets and models locally?
 
1. To improve performance of the code, we highly recommend pre-installing the datasets and models by running the `LLM 00a - Install Datasets` notebook. </br>
    <img width="400" alt="install_datasets_file" src="https://files.training.databricks.com/images/llm/installdatasets1.png">

2. You should run this script before running any of the other notebooks. This can take up to 25mins to complete. 
    <img width="1000" alt="install_datasets_notebook" src="https://files.training.databricks.com/images/llm/installdatasets2.png">
</details>

<details>
 <summary> Slides </summary>
 
 ## Where do I download course slides? 
 
 Please click the latest version under the `Releases` section. You will be able to download the slides in PDF. 
</details>
