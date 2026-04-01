

PROJECT_CONFIG = {
    "project":{
        "name": "Bank Churn Prediction System"
    },
    "data":{
        "path":"data/bank_churn.csv",
        "target_column":"Churn",
        "drop_columns":["Surname","CustomerId"]
    },
    "features":{
        'numerical_columns':['CreditScore','Age','Tenure','Balance','Num Of Products','Has Credit Card','Is Active Member','Estimated Salary'],
        "categorical_columns":['Geography', 'Gender']
    },
    "training":{
        'test_size':0.2,
        'random_state':42
    }
}
