import threading
from kafka import KafkaProducer
from kafka_iot_device import KafkaIoTDevice
from utils import *

# Initialize the Kafka producer
producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Create a function to run a device in a separate thread
def run_device(device_id):
    device = KafkaIoTDevice(device_id, read_data(SAMPLE_FILE_PATH, TARGET_NAME, COLUNMS_TO_DROP), producer, 'iot-data')
    device.send_data()

# Start 50 devices
device_threads = []

for i in range(50):
    device_id = i + 1
    t = threading.Thread(target=run_device, args=(device_id,))
    device_threads.append(t)
    t.start()

# Wait for all devices to finish
for t in device_threads:
    t.join()

# Close the Kafka producer
producer.close()