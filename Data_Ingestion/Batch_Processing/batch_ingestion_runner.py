from batch_ingestion_preprocessor import *
from typing import Tuple
def save_to_disk(X_train:dd.DataFrame, X_test:dd.DataFrame, y_train:dd.Series, y_test:dd.Series, path:str,type="pickle") -> bool:
        saver_obj = Saver(X_train, X_test, y_train, y_test)
        if type == "pickle":
            saver_obj.to_pickle_local(path)
            logging.info("Dataframe saved to disk in pickle format")
        elif type == "csv":
            saver_obj.save_csv(path)
            logging.info("Dataframe saved to disk in csv format")
        return True
batch_preprocessor = DataIngestion(CONFIG_FILE,"csv","pickle",None,None)
df = batch_preprocessor.load_data()
def batch_runner(df:dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame, dd.Series, dd.Series]:
    
    
    df = DataIngestionProcessing.set_index(df)
    df = DataIngestionProcessing.optimize_memory_df(df)
    df = DataIngestionProcessing.date_processing(df)
    X_train, X_test, y_train, y_test = DataIngestionProcessing.split_data(df)
    X_train, X_test = DataIngestionProcessing.encoding_categorical(X_train, X_test)
    X_train, X_test = DataIngestionProcessing.imputation(X_train, X_test)
    X_train, X_test = DataIngestionProcessing.scaling_numerical(X_train, X_test)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = batch_runner(df)
save_to_disk(X_train, X_test, y_train, y_test, FEATURE_STORE_PATH)
