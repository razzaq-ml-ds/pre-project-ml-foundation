# End-to-End ML System Template
## Bank Churn Prediction Implementation

This project is an end-to-end machine learning system template for tabular classification problems, demonstrated with a bank customer churn use case.

Instead of stopping at notebook experimentation, this project follows a more complete workflow:

- configuration-driven project structure
- reusable preprocessing pipeline
- feature engineering
- model comparison
- cross-validation
- threshold tuning
- artifact saving
- interactive Streamlit dashboard

The current implementation uses bank churn prediction, but the overall structure is designed so it can be adapted to other tabular ML problems with relatively small changes.

---

## Project Overview

Customer churn prediction helps a business identify customers who are likely to leave, so retention efforts can be targeted more effectively.

In this implementation, the system:

- loads bank customer data
- preprocesses raw features
- engineers additional business-oriented features
- trains and compares multiple models
- evaluates with cross-validation and threshold tuning
- saves the best model and preprocessing objects
- serves predictions through a Streamlit web application

This makes the repository more than just a model notebook. It acts as a reusable ML system template with one concrete example already built.

---

## Why This Project Matters

This repository was built as a practical machine learning engineering project, not just as a one-off model experiment.

It demonstrates how to move from:

- raw CSV data
- to preprocessing
- to model evaluation
- to saved artifacts
- to an interactive app

That makes it a stronger portfolio project than a notebook-only workflow.

---

## Dataset

- **Use case:** Bank churn prediction
- **Task type:** Binary classification
- **Target column:** `Churn`
- **Rows:** 10,000

Main customer features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

---

## End-to-End Workflow

The ML workflow used in this project is:

1. Load raw bank churn data
2. Split the dataset into train and test sets with stratification
3. Clean and preprocess the data
4. Engineer additional features
5. Train multiple candidate models
6. Compare them using test metrics and cross-validation
7. Tune classification thresholds
8. Save the best model and preprocessor
9. Build a Streamlit interface on top of the saved artifacts

---

## Feature Engineering

The preprocessing pipeline includes engineered features such as:

- `HasBalance`
- `IsAge50To60`
- `HasTwoProducts`
- `HasThreePlusProducts`
- `InactivewithBalance`
- `GermanyCustomer`
- `OneProductCustomer`
- `BalanceSalaryRatio`
- `TenureByAge`
- `CreditScoreBucket`

These features were added to convert EDA observations into explicit model inputs.

---

## Models Compared

The project compares multiple classification models, including:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

The final model is selected after threshold tuning and experiment comparison.

---

## Final Model

Based on the saved experiment results:

- **Best Model:** Gradient Boosting
- **Selected Threshold:** 0.25
- **Accuracy:** 0.841
- **Precision:** 0.5899
- **Recall:** 0.7174
- **F1 Score:** 0.6475
- **ROC-AUC:** 0.8690
- **CV F1 Mean:** 0.5765

Threshold tuning was important in this project because the default `0.50` classification cutoff was not the most useful operating point for churn detection.

---

## Streamlit App Features

The Streamlit application includes three main sections:

### 1. Data Overview

An interactive dashboard that shows:

- churn distribution
- churn by geography
- churn by gender
- age distribution by churn
- balance spread by churn
- churn rate by number of products

### 2. Prediction Lab

A live prediction interface where a user can:

- enter customer details
- generate a churn prediction
- view churn probability
- see the threshold-aware decision
- inspect a churn-risk gauge

### 3. Model Overview and Technical Details

This section includes:

- model metric comparison charts
- deployed model summary
- confusion matrix
- model comparison table

---

## Project Structure

```text
data/
models/
notebooks/
src/
app.py
main.py
project_settings.py
requirements.txt
README.md
```

### Important Files

- `project_settings.py`
  Central project configuration for paths, columns, and training settings

- `src/data_loader.py`
  Loads the dataset

- `src/preprocessing.py`
  Handles cleaning, encoding, scaling, and feature engineering

- `src/model_training.py`
  Trains models, compares experiments, performs cross-validation, and tunes thresholds

- `main.py`
  Runs the training pipeline and saves artifacts

- `app.py`
  Streamlit dashboard for visualization and live predictions

- `models/model.pkl`
  Final selected trained model

- `models/preprocessor.pkl`
  Saved preprocessing object

- `models/experiment_results.json`
  Saved experiment metadata, threshold information, metrics, and confusion matrix

---

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model pipeline

```bash
python main.py
```

### 3. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Deployment

This project is designed to be deployed easily with Streamlit Community Cloud.

Use:

- **Repository:** this GitHub repo
- **Branch:** `main`
- **Main file path:** `app.py`

---

## Screenshots

Add deployed app screenshots here after deployment.

Suggested screenshots:

- dashboard top section
- prediction lab
- prediction result with churn gauge
- model overview section

Example:

```md
![Dashboard](assets/dashboard.png)
![Prediction Lab](assets/prediction_lab.png)
![Model Overview](assets/model_overview.png)
```

---

## Live Demo

Add your deployed Streamlit link here after deployment.

Example:

```md
[Live App](https://your-app-name.streamlit.app/)
```

---

## Future Improvements

Possible next improvements for this template:

- separate CSS styling file for cleaner UI structure
- add model explainability
- add API serving with Flask or FastAPI
- improve experiment tracking
- make the template easier to adapt to other datasets
- package the project with Docker

---

## Key Learning Outcome

This project helped me practice building a machine learning system beyond only training a model.

It covers:

- modular project structure
- reusable preprocessing
- model comparison
- threshold tuning
- artifact persistence
- interactive deployment

That makes it a strong starting point for future end-to-end ML projects.

---

## Author

Built by Razza as part of a hands-on machine learning engineering journey.
