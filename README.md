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
2. Creating and comparing Classifier Models
3. 

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

Detailed documentation of the ETL and EDA processes can be found in this [notebook](), which covers data cleaning, handling missing values, data exploration, and preliminary analysis.

---

### **2 - Creating and Comparing models**

The second part of the project is dedicated to: 

- Process the data from the first part "1 - Extract, Transform and Load" to create and compare models.
    - Treating null and NaN data;
    - Treating missing data in the zone columns
    - Transforming categorical columns into binary columns (0, 1)
    - Removing useless columns
    - Saving the DataFrame in a parquet file
- Creating Models
    - Vectorizing the data (Vector Assembler)
    - Creating 4 models (Linear Regression, Decision Tree Regressor, Random Forest Regressor and Gradient-boosted tree Regressor)
- Optimizing the best model
    - Cross Validation and Hyperparameters Testing

All activities performed are documented in this [notebook]().

---

## Semana 3

Working ...