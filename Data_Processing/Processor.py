import joblib
import pandas as pd
import os 

feature_store_path = "/mnt/d/Portfolio/Auto_ML_Pdm/AutoML/Data_Ingestion/Batch_Processing"
class Preprocessor:
    def __init__(self,X_train_path,X_test_path,y_train_path,y_test_path,type="pickle"):
        if type == "pickle":
            self.X_train = joblib.load(X_train_path)
            self.X_test = joblib.load(X_test_path)
            self.y_train = joblib.load(y_train_path)
            self.y_test = joblib.load(y_test_path)
        elif type == "csv":
            self.X_train = pd.read_csv(X_train_path)
            self.X_test = pd.read_csv(X_test_path)
            self.y_train = pd.read_csv(y_train_path)
            self.y_test = pd.read_csv(y_test_path)
        else:
            raise Exception("Invalid type")
    
    def apply_outlier_treatment(self):
        pass

    def apply_feature_selection(self):
        pass
    def apply_isolation_forest_outlier_treatment(self):
        pass
    def apply_correlation_filter(self):
        pass
    def apply_pca(self):
        pass
    def apply_normalization(self):
        pass
    def apply_standardization(self):
        pass
    def apply_text_processing(self):
        pass
    def save_updated(self,type="pickle"):
        pass
    def process(self):
        self.apply_isolation_forest_outlier_treatment()
        self.save_updated()



preprocess = Preprocessor(os.path.join(feature_store_path,"X_train.pkl"),os.path.join(feature_store_path,"X_test.pkl"),os.path.join(feature_store_path,"y_train.pkl"),os.path.join(feature_store_path,"y_test.pkl"))

print(preprocess.X_train.head())
        
