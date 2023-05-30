import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class XGBoostModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def base_model(self):
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.3,
            'silent': 1.0,
            'n_estimators': 100
        }
        self.model = xgb.train(params, self.dtrain)
        return self.model

    def hyperopt_train_test(self, params):
        model = xgb.train(params, self.dtrain)
        preds = model.predict(self.dtest)
        accuracy = ((preds > 0.5) == self.y_test).mean()
        return accuracy

    def function_to_minimize(self, params):
        accuracy = self.hyperopt_train_test(params)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def optimize_hyperparameters(self):
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
            'learning_rate': hp.quniform('learning_rate', 0.025, 0.5, 0.025),
            'max_depth': hp.choice('max_depth', range(1,14)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
            'objective': 'binary:logistic',
            'silent': 1
        }
        best = fmin(self.function_to_minimize, space, algo=tpe.suggest, max_evals=100)
        return best

    def train_model(self, params):
        model = xgb.train(params, self.dtrain)
        return model


import unittest
import numpy as np
from sklearn.model_selection import train_test_split

class TestXGBoostModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming we have some data X, y
        X, y = np.random.rand(100, 10), np.random.randint(2, size=100)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_base_model(self):
        model = XGBoostModel(self.X_train, self.X_test, self.y_train, self.y_test)
        base_model = model.base_model()
        self.assertIsNotNone(base_model)

    def test_hyperopt_train_test(self):
        model = XGBoostModel(self.X_train, self.X_test, self.y_train, self.y_test)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.3,
            'silent': 1.0,
            'n_estimators': 100
        }
        accuracy = model.hyperopt_train_test(params)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_optimize_hyperparameters(self):
        model = XGBoostModel(self.X_train, self.X_test, self.y_train, self.y_test)
        best_params = model.optimize_hyperparameters()
        self.assertIn('n_estimators', best_params)
        self.assertIn('learning_rate', best_params)
        self.assertIn('max_depth', best_params)

if __name__ == "__main__":
    unittest.main()
