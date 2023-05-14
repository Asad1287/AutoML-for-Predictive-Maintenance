import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelector:
    def __init__(self, dataframe, target):
        self.dataframe = dataframe
        self.target = target

    def calculate_correlations(self):
        self.correlations = self.dataframe.corr()[self.target]
        return self.correlations

    def select_high_correlation(self, threshold=0.5):
        high_correlation_features = self.correlations[abs(self.correlations) > threshold].index
        self.dataframe = self.dataframe[high_correlation_features]
        return self.dataframe

    def plot_heatmap(self):
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.dataframe.corr(), annot=True, fmt=".2f")
        plt.show()

# Usage example:
# fs = FeatureSelector(df, 'target')
# fs.calculate_correlations()
# fs.select_high_correlation(threshold=0.5)
# fs.plot_heatmap()