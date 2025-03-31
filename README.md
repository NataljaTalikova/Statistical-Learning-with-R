# Statistical Learning Coursework – Critical Review & Loan Default Classification (2025)

This repository contains my final coursework for the **Statistical Learning with R** module, completed in 2025.

The submission consisted of two parts: a critical review and a machine learning analysis task, both included in the final report.
Despite a 10% late submission penalty (see note below), the work itself was awarded a **full 100% grade**, reflecting both technical depth and analytical clarity. 
---

## 🧠 Task 1: Critical Review of LSTM Neural Network in Finance

This task involved a critical review of a 2022 article published in *Machine Learning and Applications*, where the authors used an **LSTM neural network** to forecast **next-day returns for the S&P 500 index**.

I identified and explained a serious flaw in the paper's methodology — specifically, **data leakage** due to incorrect input preparation, which invalidated the model’s reported performance. 

---

## 📊 Task 2: Loan Default Classification (R)

The second task required applying **two machine learning methods**, preferably from those taught in class, to a real-world dataset.

I built a complete predictive pipeline using a Kaggle loan default dataset (source: [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default)), addressing imbalanced classification, feature encoding, and model evaluation.

Key steps:
- **Stratified sampling** from the full dataset to preserve class distribution
- A second **stratified train/test split** to ensure valid model assessment
- **Data preprocessing**: EDA, correlation analysis, Chi-square tests, one-hot encoding
- **Resampling**: SMOTE used to address class imbalance
- **Modeling**: Logistic Regression (with and without Lasso regularization), Support Vector Machines (RBF and Polynomial kernels)
- **Evaluation**: Custom thresholds, F1-score optimization, saved `.rds` model objects

---

## 🧪 Additional Work: Exploratory Attempt (Not Submitted)

Before the final analysis, I developed an alternate solution using a more complex, real-life dataset from the UCI Machine Learning Repository (source: [UCI Repository](https://archive.ics.uci.edu/dataset/572)). The dataset originates from the Taiwan Economic Journal and covers the period from 1999 to 2009. Bankruptcy status was determined in accordance with the regulatory criteria set by the Taiwan Stock Exchange. 

During the process, it revealed substantial challenges during model evaluation — including **nonreducible outliers** and feature-level inconsistencies. 

Although I experimented with **XGBoost** and other advanced methods, they exceeded the course scope or underperformed under the constraints. With only 36 hours remaining, I made a **strategic decision to switch datasets** and deliver a solution grounded in the taught material.

The exploratory script is included in this repository to reflect the full process and learning curve. I plan to **continue working on both datasets independently**, refining model performance and exploring techniques like ensemble learning and robust outlier handling.

---

## 📁 Repository Structure

- `code/` – Final analysis and unsubmitted exploratory script
- `models/` – Pretrained SVM model (`.rds`) used for evaluation
- `data/` – All project-related data files
- `NT Assignment Statistical Learning with R.pdf` – Final report including both tasks

---

## 🛠️ How to Run the Code

1. Open `Statistical-Learning-with-R.Rproj` in RStudio
2. Place the project-related sample CSV in the `data/` folder and the `.rds` model in `models/`
3. Run `Assignment_..._Data.R` from `code/`

---

## 🔗 Skills Demonstrated

- Deep learning model critique (LSTM in financial forecasting)
- Awareness of data leakage and time series pitfalls
- Complete classification pipeline in R for real-world FinTech use cases
- Threshold tuning and imbalanced class handling
- Code reproducibility and model persistence

---

*Author: [Natalja Talikova]*  
*MSc [Quantitative Finance with Data Science], [Birkbeck, London, UK], 2025*
