import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print("="*50)
    print("1. Data Acquisition")
    print("="*50)
    print("Fetching the German Credit dataset from OpenML...")
    # Fetch dataset
    credit_data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
    df = credit_data.frame
    
    # Target mapping: 'good' -> 1, 'bad' -> 0
    # In this dataset, target is usually in 'class' column
    target_col = 'class'
    df[target_col] = df[target_col].map({'good': 1, 'bad': 0})
    print("Dataset fetched successfully.\n")

    print("="*50)
    print("2. Exploratory Data Analysis (EDA)")
    print("="*50)
    print(f"Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0]) # Print only columns with missing values if any
    if df.isnull().sum().sum() == 0:
        print("No missing values found.")
        
    print("\nClass Distribution:")
    print(df[target_col].value_counts(normalize=True))
    print("\nSummary Statistics for Numeric Features:")
    print(df.describe())
    print("\n")

    print("="*50)
    print("3. Feature Engineering & Preprocessing")
    print("="*50)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")

    # Preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    print("Preprocessing pipeline created.\n")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}\n")

    print("="*50)
    print("4 & 5. Model Development, Evaluation & Testing")
    print("="*50)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        # Create pipeline with preprocessor and model
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        # Train model
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] # Probabilities for the positive class (1)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })
        print(f"Metrics for {name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}\n")

    # Display results as a DataFrame for a nice summary
    results_df = pd.DataFrame(results)
    print("="*50)
    print("Final Model Comparison")
    print("="*50)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
