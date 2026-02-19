from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix

class Modeltrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000) 


    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)

    def evaluate(self,X_test, y_test):
        y_pred = self.model.predict(X_test) 

        accuracy = accuracy_score(y_test,y_pred) 
        report = classification_report(y_test,y_pred)
        matrix = confusion_matrix(y_test,y_pred) 

        return accuracy , report , matrix
     
    