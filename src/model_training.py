import os
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix

class Modeltrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000) 


    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
        logging.info("model training complete!")

    def evaluate(self,X_test, y_test):
        y_pred = self.model.predict(X_test) 

        accuracy = accuracy_score(y_test,y_pred) 
        report = classification_report(y_test,y_pred)
        matrix = confusion_matrix(y_test,y_pred) 
        
        logging.info(f"model evaluated accuracy: {accuracy:.4f}")

        return accuracy , report , matrix
    
    def save_model(self,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        joblib.dump(self.model,path)
        logging.info(f"model saved at {path}")
     
    def load_model(self,path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no saved model fount at {path}"
                f"run main.py to train and save the model first"
            )
        self.model = joblib.load(path)
        logging.info(f"model loaded from {path}")
        print(f"model loaded from {path}")
        