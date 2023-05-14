import json
from kafka import KafkaConsumer
from utils import *

# Configure Kafka consumer
consumer = KafkaConsumer('iot-data',
                         bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         group_id='iot-group')

# Process the received data
for message in consumer:
    print(f"Received data: {message.value}")
    