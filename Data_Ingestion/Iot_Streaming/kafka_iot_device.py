import json
import time
import pandas as pd
from kafka import KafkaProducer
from utils import *


class KafkaIoTDevice:

    def __init__(self, device_id: int, dataframe: pd.DataFrame, kafka_producer: KafkaProducer, topic: str):
        self.device_id = device_id
        self.simulated_data_frame = dataframe
        self.kafka_producer = kafka_producer
        self.topic = topic

    def generate(self) -> dict:
        row = self.simulated_data_frame.sample().to_dict(orient='records')[0]
        row[ID_COLUMN] = self.device_id
        return row

    def send_data(self):
        while True:
            data = self.generate()
            payload = json.dumps(data)
            self.kafka_producer.send(self.topic, payload.encode('utf-8'))
            print(f"Sent data: {payload}")
            time.sleep(30)


if __name__ == "__main__":
    # Create a Kafka producer
    producer = KafkaProducer(bootstrap_servers='localhost:9092')

    # Create a KafkaIoTDevice instance
    device = KafkaIoTDevice(1, read_data(SAMPLE_FILE_PATH, TARGET_NAME, COLUNMS_TO_DROP), producer, 'iot-data')

    # Send data from the KafkaIoTDevice instance
    device.send_data()

    # Close the Kafka producer
    producer.close()
    