import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
class Model_Creator:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test
    
    def create_xgboost_model(self):
        
        model = XGBClassifier()
        model.fit(self.X_train, self.y_train)
        #test on test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        print("Accuracy: %.2f%%" % (accuracy_score(self.y_test, predictions) * 100.0))
        
        accuracy = cross_val_score(model, self.X_test, self.y_test, scoring='accuracy', cv = 10)
        print("Accuracy: %.2f%% (%.2f%%)" % (accuracy.mean()*100, accuracy.std()*100))
        return model
    
    def create_lgbm_model(self):
        model = LGBMClassifier()
        model.fit(self.X_train, self.y_train)
        #test on test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        print("Accuracy: %.2f%%" % (accuracy_score(self.y_test, predictions) * 100.0))
        accuracy = cross_val_score(model, self.X_test, self.y_test, scoring='accuracy', cv = 10)
        print("Accuracy: %.2f%% (%.2f%%)" % (accuracy.mean()*100, accuracy.std()*100))
        return model
    def create_random_forest_model(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        #test on test data
        y_pred = model.predict(self.X_test)
        predictions = [round(value) for value in y_pred]
        print("Accuracy: %.2f%%" % (accuracy_score(self.y_test, predictions) * 100.0))
        accuracy = cross_val_score(model, self.X_test, self.y_test, scoring='accuracy', cv = 10)
        return model
        