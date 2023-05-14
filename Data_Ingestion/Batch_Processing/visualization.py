import matplotlib.pyplot as plt
import seaborn as sns

class DataPlotter:
    def __init__(self, df, num_features, target='target'):
        self.df = df
        self.num_features = num_features
        self.target = target

    def plot_histograms(self):
        for feature in self.num_features:
            plt.figure(figsize=(10,6))
            plt.title(f'Histogram of {feature}')
            sns.histplot(self.df[feature], bins=30, kde=True)
            plt.show()

    def plot_bar_charts(self):
        non_num_features = [col for col in self.df.columns if col not in self.num_features]
        for feature in non_num_features:
            plt.figure(figsize=(10,6))
            plt.title(f'Bar Chart of {feature}')
            sns.countplot(x=feature, data=self.df)
            plt.show()

    def plot_features_vs_target(self):
        for feature in self.num_features:
            plt.figure(figsize=(10,6))
            plt.title(f'{feature} vs {self.target}')
            sns.regplot(x=feature, y=self.target, data=self.df, line_kws={"color": "red"})
            plt.show()

    def plot_correlation(self):
        corr = self.df[self.num_features].corr()
        plt.figure(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

"""

data_plotter = DataPlotter(df, num_features=['feature1', 'feature2'], target='target_column')
data_plotter.plot_histograms()
data_plotter.plot_bar_charts()
data_plotter.plot_features_vs_target()
data_plotter.plot_correlation()
"""