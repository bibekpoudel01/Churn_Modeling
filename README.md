Customer Churn & Salary Prediction using Deep Learning

Project Overview
This project applies Deep Learning models to solve two important business problems:
Customer Churn Prediction – Identify customers likely to leave a company
Salary Prediction – Predict employee/customer salary based on demographic and professional attributes
Both problems are addressed using Neural Networks built with modern deep learning frameworks.

Objectives
Predict customer churn using deep neural networks
Predict salary using regression-based deep learning models
Compare performance with traditional machine learning approaches
Extract actionable insights from data

Datasets
1.Customer Churn Dataset
Source: Kaggle
Target: Churn (0 = No, 1 = Yes)
Type: Binary Classification

2.Salary Dataset
Source: Kaggle / Public Dataset
Target: Salary (Continuous Value)
Type: Regression

⚠️Challenges Faced
During the development of this project, several challenges were encountered, including class imbalance in the churn dataset, missing and inconsistent values, and model overfitting during initial training. The churn data contained significantly fewer churned customers compared to non-churned ones, which required the use of class weighting and careful metric selection. Feature scaling was essential to ensure stable and faster convergence of the deep learning models, while dropout layers and early stopping were applied to mitigate overfitting.
