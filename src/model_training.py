import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

class ModelTrainer():
    def __init__(self,config):
        self.config = config
        self.models = {
        "Extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight='balanced', 
            max_depth=15,            
            min_samples_leaf=2,      
            max_features=0.45,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight='balanced',
            max_depth=10,
            min_samples_leaf=3,
            max_features=0.4,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=500,        
            learning_rate=0.009,       
            max_depth=4,
            subsample=0.8,           
            min_samples_leaf=3,
            random_state=42
        ),

        "logistic_regression": LogisticRegression(
            penalty='l2',            
            C=0.5,
            solver='saga', 
            
            max_iter=2000,
            class_weight='balanced', 
            random_state=42
        )
    }
        self.best_model = None
        self.best_model_name = None
        self.best_threshold = None

        
    def evaluate_with_threshold(self,model,X_test,y_test,threshold):
        y_prob = model.predict.proba(X_test)[:,1]
        y_pred = (y_prob>=threshold).astype(int)

        
        metrics = {
            "accuracy":float(accuracy_score(y_test,y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        return metrics

    def find_best_threshold(self,model,X_test,y_test,thresholds):
        target_recall = 0.70
        best_threshold = thresholds[0]
        best_metrics = self.evaluate_with_threshold(
            model,
            X_test,
            y_test,
            best_threshold,
        )
        for threshold in thresholds:
            current_metrics = self.evaluate_with_threshold(
                model,
                X_test,
                y_test,
                threshold,
            )
            if current_metrics["recall"]>=target_recall:
                if current_metrics["precision"] > best_metrics["precision"]:
                    best_metrics = current_metrics
                    best_threshold = threshold

        best_metrics["selected_threshold"] = best_threshold

        return best_threshold,best_metrics

    
    def train_and_compare(self,X_train,y_train,X_test,y_test,thresholds):

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
            cv_f1_mean = float(cv_scores.mean())

            model.fit(X_train,y_train,)
            
            best_threshold ,tuned_metrics = self.find_best_threshold(
                model,
                X_test,
                y_test,
                thresholds
            )

            tuned_metrics["model_name"] = model_name
            tuned_metrics["cv_f1_mean"] = cv_f1_mean
            results.append(tuned_metrics)

        best_result = results[0]
        for result in results:
            if result["f1_score"] > best_result["f1_score"]:
                best_result = result

        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]
        self.best_threshold = best_result["selected_threshold"]

        return results,best_result 
  

    
    def get_confusion_matrix(self,model,X_test,y_test,threshold):
        y_prob= model.predict_proba(X_test)[:,1]
        y_pred = (y_prob >= threshold).astype(int)

        return confusion_matrix(y_test,y_pred).tolist()
    def save_model(self,model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.best_model, model_path)

    def save_experiment_results(self, file_path, results,best_result,thresholds,conf_matrix):
        file_path = Path(file_path)
        file_path.parent.mkdir(exist_ok=True)

        experiment_data = {
            "best_model_name": self.best_model_name,
            "selection_metric": "f1_score",
            "thresholds_tested":thresholds,
            "results": results,
            "best_results":best_result,
            "confusion_matrix":conf_matrix,

        }

        with open(file_path,"w") as f:
            json.dump(experiment_data,f,indent=4)