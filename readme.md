# Project Setup
## Tech stack
The tech stack includes Python 3.10.12, Scikit-learn 1.3.2, NLTK 3.8.1, and Gensim 4.3.2 for the data science components. For the web service part, Flask 3.0.0 is utilized along with other necessary dependencies.

Instead of a Docker setup, a virtual environment is employed to ensure a focused emphasis on the data science aspect.

# Project Data
It is a version of LinearSVC and LogisticRegression models to solve science articles classification using `Title` and `Abstract` text from Kaggle Hackathon: https://www.kaggle.com/datasets/vin1234/janatahack-independence-day-2020-ml-hackathon

# Inspired by
Kaggle notebooks: https://www.kaggle.com/code/venkatkrishnan/simple-nlp-topic-modeling-approach-multiclass
and https://www.kaggle.com/code/nidhitalwar/83-26-research-topic-prediction

# Process

## Data science part
Multiple notebooks were examined to determine the most effective models for the research. It was found that LinearSVC and LogisticRegression exhibited the highest accuracy, and thus were selected for further investigation. Additionally, RandomForestClassifier was chosen for exploratory purposes. Although LSTM and Bert were suggested for achieving optimal accuracy, training Bert proved challenging due to its large size. LSTM is still a work in progress, as there was insufficient time to complete its training.

The stack of these models was tested using StackingClassifier, but the results were inferior to those obtained from the individual models.

During the research phase, data preprocessing techniques were improved by retaining numbers and preserving word case, which led to improved accuracy. Despite these enhancements, RandomForestClassifier could not surpass the accuracy of LinearSVC and LogisticRegression. 

Considering the varying accuracy results of LinearSVC and LogisticRegression across different labels (categories), separate models were trained for each label. These models are utilized individually for inference within the web service.

## Webservice part
Webservice is written in simple Flask app. Two separate modules where written for model loading and for request data preprocessing pipeline.