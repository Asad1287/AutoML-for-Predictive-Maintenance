import pandas as pd
import json
import asyncio
import paho.mqtt.client as mqtt
from Iot_Device_Simulator import *
from utils import *



async def main():
    df = read_data(SAMPLE_FILE_PATH,TARGET_NAME,COLUNMS_TO_DROP)
    
    

    mqtt_broker = MQTT_BROKER_PUBLISHER
    mqtt_port = MQTT_PORT_PUBLISHER

    topic = TOPIC

    mqtt_client = mqtt.Client()
    mqtt_client.connect(mqtt_broker, mqtt_port)

    devices = [IoTDevice(i, df, mqtt_client, topic) for i in range(1, 51)]

    async def send_data(device: IoTDevice, delay: int):
        while True:
            device.send_data()
            await asyncio.sleep(delay)

    # Run all the devices concurrently with a delay of 1 second between messages
    await asyncio.gather(*(send_data(device, 1) for device in devices))



    mqtt_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
