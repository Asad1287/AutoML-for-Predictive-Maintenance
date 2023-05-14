import pandas as pd
import random
import json
import paho.mqtt.client as mqtt
import time
from Iot_Device_Simulator import *
from utils import *
#read variable for SAMPLE_FILE_PATH from config.json, write an encoding for open function




# Create an MQTT client and connect to the broker
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER_PUBLISHER, MQTT_PORT_PUBLISHER)

# Create an IoTDevice instance
device = IoTDevice(1, read_data(SAMPLE_FILE_PATH,TARGET_NAME,COLUNMS_TO_DROP), mqtt_client, TOPIC)

# Send data from the IoTDevice instance
device.send_data()

# Disconnect the MQTT client
mqtt_client.disconnect()