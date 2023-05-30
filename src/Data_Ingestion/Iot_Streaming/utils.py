import pandas as pd
import json 
import random 

def read_data(file_path: str,target_name_col: str,columns_to_drop: list) -> pd.DataFrame:
    """
    Reads the data from the given file path and returns a dataframe and removes the target column and columns to drop
    """
    target_df = pd.read_csv(file_path)
    #drop target column and columns to drop
    target_df = target_df.drop(columns=[target_name_col] + columns_to_drop)
    return target_df

def get_feature_names(file_path: str,target_name_col: str,columns_to_drop: list) -> pd.DataFrame:
    """
    Reads the data from the given file path and returns a dataframe and removes the target column and columns to drop
    """
    target_df = pd.read_csv(file_path)
    #drop target column and columns to drop
    target_df = target_df.drop(columns=[target_name_col] + columns_to_drop)
    return target_df.columns.tolist()

CONFIG_FILE = '/workspaces/AutoML-for-Predictive-Maintenance/Data_Ingestion/Iot_Streaming/config.json'

#read variable for SAMPLE_FILE_PATH from config.json, write an encoding for open function
try: json_file = open(CONFIG_FILE, 'r', encoding='utf-8')
except FileNotFoundError as error: print(error)

json_str = json_file.read()
config = json.loads(json_str)
SAMPLE_FILE_PATH = config['SAMPLE_FILE_PATH']
TARGET_NAME = config['TARGET_NAME']
COLUNMS_TO_DROP = config['COLUNMS_TO_DROP']
#randomly choose one of value of MQTT_BROKER_PUBLISHER 
MQTT_BROKER_PUBLISHER = random.choice(config['MQTT_BROKER_PUBLISHER'])
MQTT_BROKER_SUBSCRIBER = random.choice(config['MQTT_BROKER_SUBSCRIBER'])
MQTT_PORT_PUBLISHER = config['MQTT_PORT_PUBLISHER']
MQTT_PORT_SUBSCRIBER = config['MQTT_PORT_SUBSCRIBER']
TOPIC = config['TOPIC']
ID_COLUMN = config['ID_COLUMN']
CSV_TARGET_PATH = config['CSV_TARGET_PATH']


