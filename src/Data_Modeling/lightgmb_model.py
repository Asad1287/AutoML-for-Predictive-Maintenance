from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import numpy as np

class LightGBMModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def base_model(self):
        model = LGBMClassifier()
        model.fit(self.X_train, self.y_train)
        return model

    def hyperopt_train_test(self, params):
        model = LGBMClassifier(**params)
        return cross_val_score(model, self.X_train, self.y_train).mean()

    def function_to_minimize(self, params):
        accuracy = self.hyperopt_train_test(params)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def optimize_hyperparameters(self):
        search_space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
            'max_depth': hp.choice('max_depth', range(5, 30, 1)),
            'num_leaves': hp.choice('num_leaves', range(20, 100, 1)),
            'min_child_samples': hp.choice('min_child_samples', range(20, 100, 1)),
        }

        trials = Trials()
        best_params = fmin(self.function_to_minimize, search_space, algo=tpe.suggest, max_evals=100, trials=trials)
        return best_params

import unittest
import numpy as np
from sklearn.model_selection import train_test_split

class TestLightGBMModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming we have some data X, y
        X, y = np.random.rand(100, 10), np.random.randint(2, size=100)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_base_model(self):
        model = LightGBMModel(self.X_train, self.X_test, self.y_train, self.y_test)
        base_model = model.base_model()
        self.assertIsNotNone(base_model)

    def test_hyperopt_train_test(self):
        model = LightGBMModel(self.X_train, self.X_test, self.y_train, self.y_test)
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 7,
            'num_leaves': 31,
            'min_child_samples': 20
        }
        accuracy = model.hyperopt_train_test(params)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_optimize_hyperparameters(self):
        model = LightGBMModel(self.X_train, self.X_test, self.y_train, self.y_test)
        best_params = model.optimize_hyperparameters()
        self.assertIn('n_estimators', best_params)
        self.assertIn('learning_rate', best_params)
        self.assertIn('max_depth', best_params)
        self.assertIn('num_leaves', best_params)
        self.assertIn('min_child_samples', best_params)

        
if __name__ == '__main__':
    unittest.main()


