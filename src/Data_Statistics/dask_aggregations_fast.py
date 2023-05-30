import dask.dataframe as dd
from scipy.stats import norm, lognorm, expon, weibull_min
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def fit_and_plot(data, col_name, sample_frac):
    # Persist the Dask DataFrame to keep computations in memory
    data = data.persist()

    # Sample the DataFrame
    data_sample = data.sample(frac=sample_frac).compute()

    # List of distributions to try
    distributions = [norm, lognorm, expon, weibull_min]
    distribution_names = ['norm', 'lognorm', 'expon', 'weibull_min']

    # Data for fitting
    data_for_fit = data_sample[col_name].dropna().values

    # Initialize variables to store the best distribution
    best_distribution = None
    best_params = None
    best_bic = np.inf

    # Try each distribution
    for distribution, distribution_name in zip(distributions, distribution_names):
        # Fit distribution to data
        params = distribution.fit(data_for_fit)

        # Calculate the BIC of this distribution
        bic = distribution.nnlf(params, data_for_fit) + np.log(len(data_for_fit)) * len(params)

        # If this is the best BIC so far, remember this distribution
        if bic < best_bic:
            best_distribution = distribution
            best_params = params
            best_bic = bic

        # Plot the PDF and CDF of this distribution
        plt.figure(figsize=(12,6))

        plt.subplot(121)
        sns.histplot(data_for_fit, kde=False, norm_hist=True, color='dodgerblue')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = best_distribution.pdf(x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        plt.plot(x, p, 'r', linewidth=2)
        plt.title(f'PDF of {distribution_name}')

        plt.subplot(122)
        plt.hist(data_for_fit, cumulative=True, density=True, bins=100, color='dodgerblue', alpha=0.7)
        p = best_distribution.cdf(x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        plt.plot(x, p, 'r', linewidth=2)
        plt.title(f'CDF of {distribution_name}')

        plt.show()

    # Print the best distribution and its parameters
    print(f'Best fitted distribution: {best_distribution.name}')
    print(f'Best BIC: {best_bic}')
    print(f'Parameters of the best distribution: {best_params}')

    return best_distribution, best_params
