from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix

class Modeltrainer:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000) #why and what is max_iter here


    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)#why there is no self.X_train

    def evaluate(self,X_test, y_test):
        y_pred = self.model.predict(X_test) #what is fundamental difference between train and evalute would usig the test data won't be cheathing here

        accuracy = accuracy_score(y_test,y_pred)#here i don't understand how is accuracy calculated without training data and what exactly this y_pred is 
        report = classification_report(y_test,y_pred)
        matrix = confusion_matrix(y_test,y_pred) #some doubt in report and matrix why and how only we are using y_test and y_pred are what are these three how are they calculated 

        return accuracy , report , matrix
    #  and i am not clearly getting like with these lines of code wihout importing any data from any files of the dir how is the data getting trained 
    