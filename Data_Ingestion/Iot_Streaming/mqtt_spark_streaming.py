import json
import paho.mqtt.subscribe as subscribe
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from utils import *
import queue

# MQTT broker and topic configuration
MQTT_BROKER = MQTT_BROKER_SUBSCRIBER
MQTT_PORT = MQTT_PORT_SUBSCRIBER
MQTT_TOPIC = TOPIC

# List of Queues to store MQTT messages
mqtt_queues = [queue.Queue() for _ in range(10)]

# Function to get data from MQTT
def get_mqtt_data():
    queue_index = 0
    while True:
        msg = subscribe.simple(MQTT_TOPIC, hostname=MQTT_BROKER, port=MQTT_PORT)
        mqtt_queues[queue_index].put(json.loads(msg.payload))
        queue_index = (queue_index + 1) % len(mqtt_queues)

# Start the MQTT subscriber in a separate thread
import threading
mqtt_thread = threading.Thread(target=get_mqtt_data)
mqtt_thread.start()

# Initialize Spark context and streaming context
sc = SparkContext("local[2]", "MQTTSparkStreaming")
ssc = StreamingContext(sc, 30)  # 30-second window

# Initialize Spark session
spark = SparkSession.builder.master("local").appName("mqtt-spark").getOrCreate()

# Create a DStream to receive data from MQTT
mqtt_stream = ssc.queueStream(mqtt_queues)

# Process the received data
def process_data(rdd):
    if not rdd.isEmpty():
        # Convert the RDD to a DataFrame
        df = rdd.toDF()

        # Compute the average value for each sensor type
        result = df.groupBy("sensor_type").agg(avg("value").alias("average_value"))

        # Save the result to a CSV file
        result.write.csv("output/result.csv", mode="append")

# Apply the processing function to each RDD in the DStream
mqtt_stream.foreachRDD(process_data)

# Start the streaming context and wait for it to terminate
ssc.start()
ssc.awaitTermination()
