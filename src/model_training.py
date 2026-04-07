import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score


class ModelTrainer():
    def __init__(self,config):
        self.config = config
        self.models = {
            "Logistic Regression":LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                C=0.1,
            ),
            "Random Forest":RandomForestClassifier(
                n_estimators=500,
                class_weight='balanced',
                max_depth=None,
                min_samples_leaf=5,
                max_features=0.4,
                min_samples_split=10,
                random_state=42
            ),
            "Decision Tree":DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=30,
                class_weight="balanced",
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,      
                max_depth=4,             
                subsample=0.8,           
                min_samples_leaf=15,      
                random_state=42,
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = None

    def train_and_compare(self,X_train,y_train,X_test,y_test):

        results = []
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
            )
        
        
        for model_name,model in self.models.items():
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1",
            )
            cv_f1_mean = cv_scores.mean()

            model.fit(
                X_train,
                y_train,
                )
            
            metrics = self.evaluate_model(model,X_test,y_test)
            metrics["model_name"] = model_name
            metrics["cv_f1_mean"] = cv_f1_mean
            results.append(metrics)

        best_result = results[0]
        for x in results:
            if x["f1_score"] > best_result["f1_score"]:
                best_result = x

        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]

        return results,best_result 
  
    def evaluate_with_threshold(self,model,X_test,y_test,threshold):
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = (y_prob>=threshold).astype(int)
        
        metrics = {
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_test,y_pred,zero_division=0),
            "recall":recall_score(y_test,y_pred,zero_division=0),
            "f1_score":f1_score(y_test,y_pred,zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        return metrics
    
    def evaluate_model(self,model,X_test,y_test):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

    
        metrics = {
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_test,y_pred,zero_division=0),
            "recall":recall_score(y_test,y_pred,zero_division=0),
            "f1_score":f1_score(y_test,y_pred,zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        return metrics

    

    def save_model(self,model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.best_model, model_path)

    def save_experiment_results(self, file_path, results,best_result,selected_threshold,threshold_metrics):
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)

        experiment_data = {
            "best_model_name": self.best_model_name,
            "selection_metric": "f1_score",
            "best_results":best_result,
            "selected_threshold":selected_threshold,
            "threshold_metrics": threshold_metrics,
            "results": results,
        }

        with open(file_path,"w") as f:
            json.dump(experiment_data,f,indent=4)