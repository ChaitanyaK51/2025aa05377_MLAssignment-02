# 2025aa05377_MLAssignment-02
Machine Learning Assignment 02
  Name: KALE CHAITANYA PRASAD 
  Student Id: 2025AA050377

**DATA SET NAME: Student Academic Placement Performance Dataset**

**Student Academic Placement & Salary Range Prediction using Classification Models**

**A. PROBLEM STATEMENT :-**
The objective of this project is to develop machine learning classification models to predict:
  1. Placement Status of a student (Placed / Not Placed), and
  2. Salary Package Range for placed students, categorized into four predefined salary ranges.

The problem is formulated strictly as a classification task, and no regression techniques are used for salary prediction. Multiple supervised machine learning models are implemented and evaluated using standard performance metrics to identify the most effective model for this dataset.


**B. DATASET DESCRIPTION :-**
Dataset Name: Student Academic Placement Performance Dataset
Source: Public dataset from Kaggle: https://www.kaggle.com/datasets/suvidyasonawane/student-academic-placement-performance-dataset
Number of Instances: 5000 student records
Number of Features: 18 academic and demographic attributes

**Key Features Include:**
  1. Secondary, Higher Secondary and Degree academic percentages
  2. Entrance exam score
  3. Technical and Soft skill score
  4. Work experience months
  5. Certificates
  6. Attendance and backlogs

**Target Variables:**
I. placement_status – Binary classification target
    1 → Placed
    0 → Not Placed
II.salary_package_lpa – Used only for placed students and converted into four salary classes:
    Class 1 → ≤ 3 LPA
    Class 2 → 3 – 5 LPA
    Class 3 → 5 – 10 LPA
    Class 4 → 10 -15 LPA


**C. Models Used and Evaluation Metrics :-**
The following six machine learning classification models were implemented on the same dataset:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. k-Nearest Neighbors (kNN)
  4. Naive Bayes (Gaussian)
  5. Random Forest (Ensemble Model)
  6. XGBoost (Ensemble Model)

Each model was evaluated using the following metrics:
  1. Accuracy
  2. AUC Score
  3. Precision
  4. Recall
  5. F1 Score
  6. Matthews Correlation Coefficient (MCC)

Model Performace Metrics:-
Comparison Table:-
| **ML Model Name**        | **Accuracy** | **AUC**   | **Precision** | **Recall** | **F1 Score** | **MCC**   |
| ------------------------ | ------------ | --------- | ------------- | ---------- | ------------ | --------- |
| Logistic Regression      | 0.891        | 0.936     | 0.719         | 0.607      | 0.658        | 0.597     |
| Decision Tree            | **1.000**    | **1.000** | **1.000**     | **1.000**  | **1.000**    | **1.000** |
| kNN                      | 0.897        | 0.934     | 0.769         | 0.578      | 0.660        | 0.609     |
| Naive Bayes              | 0.933        | 0.984     | 0.942         | 0.653      | 0.771        | 0.750     |
| Random Forest (Ensemble) | **1.000**    | **1.000** | **1.000**     | **1.000**  | **1.000**    | **1.000** |
| XGBoost (Ensemble)       | **1.000**    | **1.000** | **1.000**     | **1.000**  | **1.000**    | **1.000** |




Model Performance Observations :-
| ML Model Name            | Observation about Model Performance                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Served as a strong baseline classifier with stable and interpretable results but was limited in modeling complex non-linear relationships.             |
| Decision Tree            | Provided clear interpretability but showed signs of overfitting, leading to variability in test performance.                                           |
| kNN                      | Performance was sensitive to feature scaling and the chosen value of k. It achieved reasonable accuracy but was computationally less efficient.        |
| Naive Bayes              | Trained very quickly and produced fast predictions but had comparatively lower accuracy due to the independence assumption among features.             |
| Random Forest (Ensemble) | Demonstrated strong and consistent performance by reducing overfitting and effectively capturing feature interactions through ensemble learning.       |
| XGBoost (Ensemble)       | Achieved the best overall performance across Accuracy, AUC, F1 Score, and MCC, indicating superior capability in handling complex decision boundaries. |


CONCLUSION :-

The experimental results show that ensemble-based models, particularly XGBoost, outperform individual classifiers in predicting student placement outcomes.
While simpler models provide interpretability and baseline insights, advanced ensemble techniques offer improved generalization and robustness for real-world academic placement prediction tasks.


