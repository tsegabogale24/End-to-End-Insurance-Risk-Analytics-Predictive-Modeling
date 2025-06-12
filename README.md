# AlphaCare Car Insurance Analytics Challenge

##  Project Overview

This project is part of a marketing analytics challenge at **AlphaCare Insurance Solutions (ACIS)**. The objective is to analyze car insurance claim data in South Africa to uncover patterns in risk and profitability, optimize marketing strategy, and identify low-risk customer segments eligible for premium reductions.

## 📊 Business Objectives

- Perform A/B hypothesis testing to validate assumptions on risk and profit variations.
- Conduct Exploratory Data Analysis (EDA) to understand the dataset and detect patterns.
- Build statistical and machine learning models to:
  - Predict total claims based on features.
  - Recommend optimal premium values based on risk levels.

## Folder Structure

acis-car-insurance-analytics/
│
├── data/ # Raw and processed data files
├── notebooks/ # Jupyter notebooks for analysis
├── scripts/ # Python scripts for preprocessing and modeling
├── outputs/ # Visualizations, charts, and exported results
├── .github/workflows/ # CI/CD pipelines (e.g., linting, tests)
├── README.md # Project overview and documentation



##  Dependencies

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- statsmodels
- jupyter
- flake8, black (for code quality)

Install them using:

```bash
pip install -r requirements.txt
   Task Overview
Task	Description
Task 1	EDA and hypothesis testing across demographic and geographic groups
Task 2	Modeling total claims using linear regression
Task 3	Predictive model for optimal premiums
Task 4	Report generation with insights and recommendations