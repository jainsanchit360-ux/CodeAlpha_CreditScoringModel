# Credit Scoring Model

This project builds, trains, and evaluates a machine learning pipeline to predict creditworthiness based on the popular **German Credit Dataset**. The project compares three different classification algorithms to determine the best approach for identifying credit risk.

## Overview

The goal of this model is to classify individuals as either good (1) or bad (0) credit risks using a series of demographic and financial features such as checking account status, credit history, loan purpose, and employment length.

We explore and evaluate the following machine learning models:
- **Logistic Regression**: A baseline linear model for classification.
- **Decision Tree**: A non-linear model capturing decision rules.
- **Random Forest**: An ensemble method to improve accuracy and control overfitting.

## Dataset
The dataset used is the **German Credit dataset** (`credit-g`), fetched automatically via `scikit-learn` from OpenML. 

- **Samples**: 1000
- **Features**: 20 (7 numerical, 13 categorical)
- **Target Variable**: `class` ('good' or 'bad' credit risk)

## Setup and Installation

### Prerequisites
You need **Python 3.x** installed. The project relies on the following standard data science libraries:
- `scikit-learn`
- `pandas`
- `numpy`

### Installation
You can install the required packages using `pip`:
```bash
pip install scikit-learn pandas numpy
```

## Running the Code

To execute the model pipeline, run the main Python script from your terminal:

```bash
python credit_scoring.py
```

### What the Script Does:
1. **Data Acquisition**: Connects to OpenML and downloads the dataset.
2. **Exploratory Data Analysis (EDA)**: Prints basic data shape, checks for missing values, and shows feature summaries.
3. **Feature Engineering**: 
   - Scales numeric variables using `StandardScaler`.
   - Encodes categorical variables using `OneHotEncoder`.
4. **Model Training**: Fits all three classifiers on an 80/20 train/test split.
5. **Evaluation**: Outputs metrics for each model.

## Model Performance

The evaluation metrics calculated are:
- **Accuracy**: Overall correctness of the model.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: The area under the receiver operating characteristic curve, representing the model's ability to distinguish between classes.

Typical baseline results (actual values may vary slightly based on environment differences but should be close to these):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~75% | ~80% | ~85% | ~82% | ~80% |
| Decision Tree | ~70% | ~80% | ~75% | ~77% | ~63% |
| Random Forest | ~78% | ~81% | ~90% | ~85% | ~80% |

> **Note**: While Random Forest generally performs the best and is parallelized on the CPU (`n_jobs=-1`), scikit-learn's standard algorithms are run on the CPU. Utilizing a local RTX GPU would require adapting the code to use libraries like `XGBoost` (with `tree_method='gpu_hist'`) or `cuML`.
