# Machine Learning for Churn Prediction
<div align="center">
  <a href="https://www.linkedin.com/in/fernando-lacerda-/" target="_blank">
    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="https://github.com/Lacerdash">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

Churn prediction is crucial for businesses to identify potential customers who are likely to discontinue using their service. This repository contains my project aiming to tackle this issue.

## **Project Overview**

**Objectives:** To identify potential churn customers and understand the associated patterns, thus enabling businesses to take proactive measures to retain them.

**Data:** The data is avaible [here]() and the data dictionary [here]()

**Structure:** The project is in 3 parts: 
1. Extract, Transform and load (ETL) and Exploratory Data Analysis (EDA)
2. Creating, Selecting and Optimizing models
3. Creating and Streamlit app to deploy our model

---

### **1 - Extract, Transform and Load (ETL) and Exploratory Data Analysis (EDA)**

*ETL*
- The dataset, in JSON format, is imported into Python and undergoes transformation and cleaning. After processing, the data is saved to a [csv file](). 

*EDA*
- Post-ETL, various visualizations are generated to delve deeper into the patterns within the data, identify potential problems, and better understand the overall structure of the dataset.

Challenges faced: 
- Handling missing values 
- Data enconding
- Correcting data types,
- Ploting relevants graphs for analysis
- Ceation of functions in a python file ([helper.py](https://github.com/Lacerdash/ML-for-Churn-predicting/blob/master/helpers.py)) for a more clean notebook.

Detailed documentation of the ETL and EDA processes can be found in this [notebook](https://github.com/Lacerdash/ML-for-Churn-predicting/blob/master/Churn_Data_Cleansing_and_Exploration.ipynb), which covers data cleaning, handling missing values, data exploration, and preliminary analysis.

---

### **2 - Creating, Selecting and Optimizing models**

The second part of the project is dedicated to: 

- Process the [encoded_churn_data.csv](https://github.com/Lacerdash/ML-for-Churn-predicting/blob/master/data/encoded_churn_data.csv) file generated in the first part: "1 - Extract, Transform and Load" to create and compare models"
    - Creating new columns
    - Scaling data
    - Balacing data
- Creating Baseline Models
    - Creating 9 models (Decision Tree Regressor, Random Forest Regressor, Logistic Regression, KNeighborsClassifier, SVC, GradientBoostingClassifier, GaussianNB, AdaBoostClassifier and MLPClassifier)
- Select best model basead on choosen metric
- Optimize best models Hyperparameters and access its results
    - Usinig nested Cross validation to peform hyperparameter tunning and model assessment (You can check my in depth notebook on [Nested Cross Validation](https://github.com/Lacerdash/Nested-Cross-Validation))
- Save model

All activities performed are documented in this [notebook](https://github.com/Lacerdash/ML-for-Churn-predicting/blob/master/ModelCreation_Selection_Optimization.ipynb).

---

## **3 - Streamlit app and Model deployment**

To deploy the model we created a streamlit app that allows users to interact with it through an easy interface. This includes:

- Model selection
- Data insertion
- Prediction of Churn probability
- Exploratory Data Analysis tab with visualizations

[Streamlit app](https://ml-for-churn-predicting.streamlit.app/)