import joblib
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class ModelTrainer():
    def __init__(self,config):
        self.config = config
        self.model = LogisticRegression(max_iter=1000)

    def train_model(self,X_train,y_train):
        self.model.fit(X_train,y_train)

    def evaluate_model(self,X_test,y_test):
        y_pred = self.model.predict(X_test)

        metrics = {
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_test,y_pred),
            "recall":recall_score(y_test,y_pred),
            "f1_score":f1_score(y_test,y_pred)
        }
        return metrics
    def save_model(self,model_path):
        model_path = Path(model_path)
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, model_path)