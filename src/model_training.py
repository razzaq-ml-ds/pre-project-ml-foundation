import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import json

class ModelTrainer():
    def __init__(self,config):
        self.config = config
        self.models = {
            "Logistic Regression":LogisticRegression(max_iter=1000),
            "Random Forest":RandomForestClassifier(),
            "Decision Tree":DecisionTreeClassifier()
        }
        self.best_model = None
        self.best_model_name = None


    def train_and_compare(self,X_train,y_train,X_test,y_test):
        results = []
        
        for model_name,model in self.models.items():
            model.fit(X_train,
                      y_train)
            metrics = self.evaluate_model(model,X_test,y_test)
            metrics["model_name"] = model_name
            results.append(metrics)

        best_result = results[0]
        for x in results:
            if x["f1_score"] > best_result["f1_score"]:
                best_result = x

        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]

        return results,best_result 
  

    def evaluate_model(self,model,X_test,y_test):
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_test,y_pred,zero_division=0),
            "recall":recall_score(y_test,y_pred,zero_division=0),
            "f1_score":f1_score(y_test,y_pred,zero_division=0)
        }

        return metrics

    

    def save_model(self,model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.best_model, model_path)

    def save_experiment_results(self, file_path, results,best_result):
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)

        experiment_data = {
            "best_model_name": self.best_model_name,
            "selection_metric": "f1_score",
            "best_results":best_result,
            "results": results
        }

        with open(file_path,"w") as f:
            json.dump(experiment_data,f,indent=4)