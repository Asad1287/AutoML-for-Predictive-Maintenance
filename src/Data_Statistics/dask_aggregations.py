import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt

def categorical_agg(dask_df, cat_cols):
    # Function to calculate aggregations for categorical columns
    categorical_agg_df = dask_df[cat_cols].categorize().groupby(cat_cols).size().compute()
    # Save to csv
    categorical_agg_df.to_csv('categorical_aggregations.csv')
    print("Categorical aggregations saved to 'categorical_aggregations.csv'")
    return categorical_agg_df

def numerical_agg(dask_df, num_cols):
    # Function to calculate aggregations for numerical columns
    numerical_agg_df = dask_df[num_cols].describe().compute()
    # Save to csv
    numerical_agg_df.to_csv('numerical_aggregations.csv')
    print("Numerical aggregations saved to 'numerical_aggregations.csv'")
    return numerical_agg_df

def plot_hist_corr(df, num_cols):
    # Function to create histogram and correlation for numerical columns
    for col in num_cols:
        plt.figure(figsize=(9, 6))
        plt.hist(df[col], bins=30, alpha=0.5, color='g', label=col)
        plt.legend(loc='upper right')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    correlation = df[num_cols].corr()
    print("\nCorrelation Matrix:")
    print(correlation)