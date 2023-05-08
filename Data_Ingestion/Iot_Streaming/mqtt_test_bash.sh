#!/bin/bash

# Install Mosquitto MQTT broker
#sudo apt-get update
#sudo apt-get install mosquitto mosquitto-clients

# Start Mosquitto MQTT broker
echo "Starting Mosquitto MQTT broker..."
sudo service mosquitto start

# Run the MQTT subscriber script
echo "Starting MQTT subscriber..."
python Data_Ingestion/Iot_Streaming/mqtt_subscriber_one.py &

# Give the subscriber some time to connect before starting the publisher
sleep 5

# Run the MQTT publisher script
echo "Starting MQTT publisher..."
python /workspaces/AutoML-for-Predictive-Maintenance/Data_Ingestion/Iot_Streaming/mqtt_publisher_one.py &

# Stop the Mosquitto MQTT broker after the publisher finishes
echo "Stopping Mosquitto MQTT broker..."
sudo service mosquitto stop

echo "Starting Spark Streaming job..."
python /workspaces/AutoML-for-Predictive-Maintenance/Data_Ingestion/Iot_Streaming/mqtt_spark_streaming.py &

wait

echo "Stopping Mosquitto MQTT broker..."
sudo service mosquitto stop