

OUTLIER_COLUMNS = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]"]
def outlier_removal(X_train:dd.DataFrame,OUTLIER_COLUMNS:List[int],lower:float=0.025,upper:float=0.975) -> dd.DataFrame:
  
  lower_bound_dict = {}
  upper_bound_dict = {}
  for col in OUTLIER_COLUMNS:
    lower_bound_dict[col] = X_train[col].quantile(lower)
    upper_bound_dict[col] = X_train[col].quantile(upper)

  for col in OUTLIER_COLUMNS:
      X_train = X_train[(X_train[col] > lower_bound_dict[col]) & (X_train[col] < upper_bound_dict[col])]

  return X_train 


from scipy.stats import zscore
from typing import Tuple,List
# Assuming df is your Dask DataFrame

# Specify your numerical columns
OUTLIER_COLUMNS = ["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]"]
def remove_outliers_zscore(df:dd.DataFrame,OUTLIER_COLUMNS:List[str]) -> dd.DataFrame:
  

  # Define a function to calculate z-scores and filter the DataFrame
  def remove_outliers_partitions(partition):
      for col in OUTLIER_COLUMNS:
          z_scores = zscore(partition[col])
          partition = partition[(z_scores >= -3) & (z_scores <= 3)]  # Filter out outliers
      return partition

  # Apply the function to each partition of the DataFrame
  df = df.map_partitions(remove_outliers_partitions, meta=df)
  return df

class OutlierDection:
  @staticmethod 
  def remove_outliers_zscore(df:dd.DataFrame,OUTLIER_COLUMNS:List[str]) -> dd.DataFrame:
  

  # Define a function to calculate z-scores and filter the DataFrame
    def remove_outliers_partitions(partition):
        for col in OUTLIER_COLUMNS:
            z_scores = zscore(partition[col])
            partition = partition[(z_scores >= -3) & (z_scores <= 3)]  # Filter out outliers
        return partition

  # Apply the function to each partition of the DataFrame
    df = df.map_partitions(remove_outliers_partitions, meta=df)
    return df

  @staticmethod

  def outlier_removal(X_train:dd.DataFrame,OUTLIER_COLUMNS:List[int],lower:float=0.025,upper:float=0.975) -> dd.DataFrame:
  
    lower_bound_dict = {}
    upper_bound_dict = {}
    for col in OUTLIER_COLUMNS:
      lower_bound_dict[col] = X_train[col].quantile(lower)
      upper_bound_dict[col] = X_train[col].quantile(upper)

    for col in OUTLIER_COLUMNS:
        X_train = X_train[(X_train[col] > lower_bound_dict[col]) & (X_train[col] < upper_bound_dict[col])]

    return X_train 