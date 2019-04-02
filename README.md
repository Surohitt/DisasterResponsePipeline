# Disaster Response Pipeline Project

![](http://www.giphy.com/gifs/3YIfBJmb0JkFA1sNBz)

## Table of Contents
1. [Installation and Instructions](#Installation)
2. [Project Motivation](#Motivation)
3. [File Descriptions](#Descriptions)
4. [Results](#Results)
5. [Acknowledgements](#Acknowledgements)

# Installation and Instructions <a name="Installation"></a>:
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python which can be found [here](https://www.anaconda.com/). The code should run with no issues using Python versions 3. You may need to install [Flask](http://flask.pocoo.org/), [sqlalchemy](https://www.sqlalchemy.org/) and [Plotly](https://plot.ly/)

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Project Motivation <a name='Motivation'></a>

This project was offered by [Figure Eight](https://www.figure-eight.com/) in association with [Udacity](https://www.udacity.com/). The personal value of this project was to learn and implement basic web dev skills, ETL creation and NLP preprocessing standards.

This project attempts to classify disaster response messages to a high degree of accuracy. With the hope of helping first responders understand the huge volumes of messages they receive whenever major disasters occur. A project like this has the potential to save lives by increasing efficiency of aid workers.

Also, I want to encourage pull requests and further analysis on top of this project! I would love to see what the open source community could contribute to a project like this.

# File Descriptions <a name="Descriptions"></a>

#### Prepfiles
I have compiled the ETL and ML pipelines into two single notebooks in the Prepfiles folder, titled "ETL Pipeline Preparation.ipynb" and "ML Pipeline Preparation.ipynb" respectively. Markdown cells were used to assist in walking through the thought process for individual steps.
#### WebApp
The other folders are reasonably self explanatory. The "run.py" file runs the app itself and extracts the necessary data for the visualisations found on the "master.html" file. The "go.html" file runs the results for the input messages.

The "process_data.py" file uses the aforementioned ETL pipeline we created in the notebook, only now it is refactored. The same goes for the "train_classifier.py" file, however, this relates to the ML pipeline notebook.

# Results <a name='Results'></a>

Stay tuned for a blog post on this whole project!

# Acknowledgements <a name='Acknowledgements'></a>

Thanks must go to [Figure Eights](https://www.figure-eight.com/) for creating this data set and allowing me to use it for this analysis.

Also, I would like to acknowledge [Udacity](https://www.udacity.com/) for reviewing this project and being a guiding hand to help ensure I deliver quality work.
