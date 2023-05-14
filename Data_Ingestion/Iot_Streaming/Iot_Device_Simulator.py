import pandas as pd
import json
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt
import time
from utils import *


class IoTDevice:
    
    def __init__(self, device_id: int, dataframe:pd.DataFrame, mqtt_client_obj: mqtt.Client, topic_mqtt: str):
        self.device_id = device_id
        self.simulated_data_frame = dataframe
        self.mqtt_client = mqtt_client_obj
        self.topic = topic_mqtt

    def generate(self) -> dict:
        row = self.simulated_data_frame.sample().to_dict(orient='records')[0]
        row[ID_COLUMN] = self.device_id
        return row

    def send_data(self):
        while True:
            data = self.generate()
            payload = json.dumps(data)
            self.mqtt_client.publish(self.topic, payload)
            print(f"Sent data: {payload}")
            time.sleep(30)




"""
test code
df = pd.read_csv('/workspaces/AutoML-for-Predictive-Maintenance/ai4i2020.csv')
target_name = 'Machine failure'
#drop the target column and UDI column for feature names of columns
feature_names = df.drop(columns=[target_name,'UDI']).columns.tolist()

mqtt_broker = "mqtt.eclipse.org"  # Replace with your MQTT broker address
mqtt_port = 1883

topic = "sensors/data"

mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port)

device1 = IoTDevice(1, df,feature_names,target_name,mqtt_client)
device2 = IoTDevice(2, df,feature_names,target_name,mqtt_client)
print(device1)
print(device1.generate())
print(device2.generate())
"""