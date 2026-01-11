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
    Class 2 → 3 – 6 LPA
    Class 3 → 6 – 10 LPA
    Class 4 → >10 LPA


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

Binary Model Performance:

Multi-class Model Performance:



Model Performance Observations :-
| ML Model Name            | Observation about Model Performance                                                                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Performed well as a baseline classifier with stable results, but struggled to capture complex non-linear relationships present in the dataset.               |
| Decision Tree            | Showed good interpretability but exhibited signs of overfitting, leading to fluctuating performance across evaluation metrics.                               |
| kNN                      | Performance was sensitive to feature scaling and choice of k. It worked reasonably well but was computationally less efficient for larger datasets.          |
| Naive Bayes              | Delivered fast predictions but showed lower accuracy due to the strong assumption of feature independence, which does not fully hold for this dataset.       |
| Random Forest (Ensemble) | Achieved strong performance across most metrics by reducing overfitting through ensemble learning and handling feature interactions effectively.             |
| XGBoost (Ensemble)       | Provided the best overall performance with high AUC, F1 Score, and MCC, demonstrating superior capability in modeling complex patterns and class boundaries. |




