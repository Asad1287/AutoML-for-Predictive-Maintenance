import pickle
from sklearn.model_selection import train_test_split
import pandas as pd

class ModelSelector:
    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df.drop(target, axis=1), df[target], test_size=0.2, random_state=42
        )

    def select_best_model(self):
        lgbm_opt = LightGBMOptimizer(self.X_train, self.X_test, self.y_train, self.y_test)
        xgb_opt = XGBoostOptimizer(self.X_train, self.X_test, self.y_train, self.y_test)
        
        best_lgbm_model, best_lgbm_params = lgbm_opt.optimize()
        best_xgb_model, best_xgb_params = xgb_opt.optimize()

        lgbm_score = best_lgbm_model.score(self.X_test, self.y_test)
        xgb_score = best_xgb_model.score(self.X_test, self.y_test)

        if lgbm_score > xgb_score:
            best_model = best_lgbm_model
            best_params = best_lgbm_params
            print("Best model is LightGBM with score:", lgbm_score)
        else:
            best_model = best_xgb_model
            best_params = best_xgb_params
            print("Best model is XGBoost with score:", xgb_score)
        
        return best_model, best_params

    def save_model(self):
        best_model, best_params = self.select_best_model()
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        with open('best_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)
        with open('features.pkl', 'wb') as f:
            pickle.dump(self.df.drop(self.target, axis=1).columns.tolist(), f)
        print("Models and features saved.")
