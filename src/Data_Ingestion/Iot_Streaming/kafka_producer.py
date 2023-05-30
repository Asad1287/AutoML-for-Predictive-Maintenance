from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

data = {'message': 'Hello, World!'}

producer.send('test-topic', data)
producer.flush()
producer.close()