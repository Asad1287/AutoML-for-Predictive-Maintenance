import paho.mqtt.client as mqtt
from collections import deque
import json
import time 
import csv
from utils import *

received_data = deque()

def on_connect(client: mqtt.Client, userdata: object, flags: dict, rc: int):
    print("Connected with result code " + str(rc))
    client.subscribe("sensors/data")
   

def on_message(client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage):
    print("Topic: {}, Received Data: {}".format(msg.topic, msg.payload.decode()))
    data = json.loads(msg.payload.decode())
    received_data.append(data)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(MQTT_BROKER_SUBSCRIBER, MQTT_PORT_SUBSCRIBER, 60)
mqtt_client.loop_start()


feature_names = get_feature_names(SAMPLE_FILE_PATH,TARGET_NAME,COLUNMS_TO_DROP)


try:
    while True:
        time.sleep(30)
        with open(CSV_TARGET_PATH, "a",encoding='utf-8') as csvfile:
            fieldnames = feature_names
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            while received_data:
                writer.writerow(received_data.popleft())
except KeyboardInterrupt:
    print("Exiting...")
finally:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
