# 2025AA05377_MLAssignment-02
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
| **ML Model Name**            | **Observation about Model Performance**                                                                                                                                                                           |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**      | Served as a strong baseline classifier with stable and interpretable results. However, its linear decision boundary limited its ability to fully capture complex non-linear relationships present in the dataset. |
| **Decision Tree**            | Achieved perfect classification performance by learning explicit decision rules from the data. The results indicate that the dataset is highly separable rather than reflecting evaluation bias.                  |
| **kNN**                      | Performance was sensitive to feature scaling and the choice of the parameter *k*. While it achieved reasonable accuracy, its instance-based nature makes it computationally less efficient for larger datasets.   |
| **Naive Bayes**              | Demonstrated fast training and prediction times with high precision. However, recall was comparatively lower due to the strong assumption of conditional independence among features.                             |
| **Random Forest (Ensemble)** | Delivered excellent and stable performance by aggregating multiple decision trees, effectively capturing deterministic patterns in the data while reducing variance through ensemble averaging.                   |
| **XGBoost (Ensemble)**       | Achieved perfect scores across all evaluation metrics, highlighting its strong capability to model complex feature interactions and hierarchical decision boundaries inherent in the dataset.                     |



CONCLUSION :-

The experimental results indicate that ensemble-based models—particularly Decision Tree, Random Forest, and XGBoost—achieved perfect classification performance due to the deterministic and highly separable nature of the dataset. These models were able to learn exact decision rules that distinguish placement outcomes with complete accuracy.

In contrast, simpler models such as Logistic Regression, kNN, and Naive Bayes provided more conservative yet realistic performance estimates, offering valuable interpretability and baseline insights. Overall, the study demonstrates that while traditional classifiers are useful for understanding feature influence, ensemble learning methods are highly effective for structured academic placement prediction tasks where clear decision boundaries exist.

