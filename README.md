# Machine Learning for Churn Prediction
<div align="center">
  <a href="https://www.linkedin.com/in/fernando-lacerda-/" target="_blank">
    <img src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="https://github.com/Lacerdash">
    <img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
</div>

This repository contains my project 

## **Project Overview**

**Objectivies:** Correctly identify potential Churn costumers.

**Data:** The data is avaible [here]() and the data dictionary [here]()

**Structure:** The project is in 3 parts: ETL (Extract, Transform and load), Creating and comparing Classificators Models and 

---

### **1 - Extract, Transform and Load**

The first part of the project is dedicated to the ETL process of the data. Extracting the data in json format into python, for subsequent transformation/cleaning of the data, followed by loading the data into a [csv file]().

All activities performed are documented in this [notebook]().

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