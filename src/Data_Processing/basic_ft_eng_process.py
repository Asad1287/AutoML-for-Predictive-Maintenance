class BasicFeatureEngineering:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def multiply_columns(self, col1, col2, new_col_name):
        if col1 in self.dataframe.columns and col2 in self.dataframe.columns:
            self.dataframe[new_col_name] = self.dataframe[col1] * self.dataframe[col2]
        else:
            raise ValueError(f"One or both columns: {col1}, {col2} not found in dataframe")
        return self.dataframe

    def divide_columns(self, col1, col2, new_col_name):
        if col1 in self.dataframe.columns and col2 in self.dataframe.columns:
            # Adding a small constant to avoid division by zero
            self.dataframe[new_col_name] = self.dataframe[col1] / (self.dataframe[col2] + 1e-7)
        else:
            raise ValueError(f"One or both columns: {col1}, {col2} not found in dataframe")
        return self.dataframe
