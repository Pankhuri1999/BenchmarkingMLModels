# 📊 BenchmarkingMLModels  
### Unified Machine Learning Benchmarking Framework for Educational Analytics

This repository provides a reproducible machine learning benchmarking framework across multiple education-focused prediction tasks:

- Student Enrollment Prediction  
- Student Performance Analysis  
- Student Retention (Dropout) Prediction  
- University Ranking Prediction  
- Budget Prediction  

In addition to standard model accuracy, this framework introduces an **AI Readiness Index** combining:

- Accuracy  
- Fairness  
- Interpretability (SHAP)  
- Data Quality  
- Scalability  

The objective is to evaluate both predictive performance and deployment readiness of ML models in educational contexts.


## 📁 Repository Structure
BenchmarkingMLModels/

├── StudentEnrollment/

├── StudentPerformance/

├── StudentRetention/

├── UniversityRanking/

├── BudgetPrediction/

Root:

├── singleFramework.py

├── Overall.py

└── README.md



---

## 📂 Folder Contents

Each task folder contains:

- Python files (`*.py`) – ML pipelines  
- Jupyter notebooks (`*.ipynb`) – graphs and visualizations  
- Dataset ZIP files  

Enrollment and Budget folders additionally include:

- Preprocessed datasets  
- Dataset reframing scripts  

Separate notebooks are provided for each task to view visualizations independently.

---

## 📊 Dataset Sources

Datasets were downloaded directly from Kaggle and are also available via:

- Student Performance Factors  
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors  

- College Enrollment in the US (2003–2021)  
https://www.kaggle.com/datasets/eldarsarajlic/college-enrollment-in-the-us-20032021  

- THE World University Rankings (2016–2024)  
https://www.kaggle.com/datasets/raymondtoo/the-world-university-rankings-2016-2024  
https://www.timeshighereducation.com/world-university-rankings  

- Higher Education Predictors of Student Retention  
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention  

- Boxplots for Education Dataset (Budget Prediction)  
https://www.kaggle.com/datasets/jeromeblanchet/drivendatas-boxplots-for-education-dataset  
https://www.drivendata.org/competitions/46/  
https://zenodo.org/records/5777340  

---

## 🧹 Dataset Preparation

### Student Enrollment & Budget Prediction

These datasets required reframing due to:

- Missing or inconsistent targets  
- Non-tabular layouts  
- Mixed categorical/numerical formats  

For these tasks:

- Reframing scripts are provided  
- Post-preprocessing datasets are uploaded  
- Cleaned datasets are directly used by pipelines  

---

### Student Performance, Student Retention, University Ranking

These datasets were already well structured.

For these tasks:

- No external reframing required  
- All preprocessing is performed dynamically inside the pipeline  

---

## ⚙ Unified Framework

### Run Individual Tasks

Use `singleFramework.py` if you want to run any task individually.

### Run All Tasks Together

Use `Overall.py` to execute all five tasks in a single run.

## Methodology

Dataset loading

Optional reframing (Enrollment & Budget only)

Missing value handling and encoding

Automatic task identification

Model training

Optional hyperparameter tuning

Performance evaluation

SHAP-based feature attribution

Fairness assessment

Data quality scoring

Scalability estimation

Readiness Index computation

## Requirements
Python ≥ 3.8

pip install numpy pandas scikit-learn matplotlib shap jupyter

Optional:

pip install xgboost lightgbm

## Usage
Run individual task -

python StudentEnrollment/StudentEnrollmentPythonFileCode.py

Run all tasks -

python Overall.py

## Computing Infrastructure

Experiments were conducted using Jupyter Notebook environments (Google Colab and local CPU systems):

Operating System: Windows / Linux

Python ≥ 3.8

CPU-based execution

RAM: ~8–12 GB

GPU: Not required

## Reproducibility
To reproduce results:

Clone the repository

Install required packages

Run individual task pipelines, singleFramework.py, or execute Overall.py

All datasets, notebooks, preprocessing scripts, and pipelines are included.


### Example Unified Call

```python
results_dropout = comprehensive_ml_pipeline(
    dataset_path="retention/dataset.csv",
    target_column="Target",
    task_type="auto",
    fast_mode="auto",
    do_hyperparam_tuning=True,
    compute_shap=True,
    use_tfidf=False
)```

You may modify: dataset_path, target_column, do_hyperparam_tuning (True/False), compute_shap (True/False), use_tfidf (True/False, only for text features)



