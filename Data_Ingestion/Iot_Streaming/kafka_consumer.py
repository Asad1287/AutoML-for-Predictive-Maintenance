from kafka import KafkaConsumer
import json

consumer = KafkaConsumer('test-topic',
                         bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                         auto_offset_reset='earliest',
                         group_id='test-group')

for message in consumer:
    print("Received message:", message.value)

consumer.close()