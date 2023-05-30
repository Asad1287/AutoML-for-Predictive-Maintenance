

class CustomDaskMinMaxScaler:
    def __init__(self):
        self.numeric_cols = None
        self.mins = None
        self.maxs = None

    def fit(self, df):
        self.numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        self.mins = df[self.numeric_cols].min().compute()
        self.maxs = df[self.numeric_cols].max().compute()

    def transform(self, df):
        if self.mins is None or self.maxs is None:
            raise RuntimeError("Must fit scaler before transforming")

        df_transformed = df.copy()
        for col in self.numeric_cols:
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].map_partitions(lambda s: (s - self.mins[col]) / (self.maxs[col] - self.mins[col]), meta=(col, 'float'))
        return df_transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


