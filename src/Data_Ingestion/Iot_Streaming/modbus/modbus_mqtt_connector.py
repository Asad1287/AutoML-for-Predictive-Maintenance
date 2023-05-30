import paho.mqtt.client as mqtt
import struct
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder
from pymodbus.payload import BinaryPayloadDecoder

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe("modbus/sensor_data")

def on_message(client, userdata, msg):
    print(msg.topic, msg.payload)

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60)

def read_sensor_values(modbus_client, num_values=4, starting_address=0):
    response = modbus_client.read_holding_registers(starting_address, num_values * 2)
    decoder = BinaryPayloadDecoder.fromRegisters(response.registers, byteorder=Endian.Big, wordorder=Endian.Little)
    sensor_values = [decoder.decode_32bit_float() for _ in range(num_values)]
    return sensor_values

with ModbusTcpClient("localhost", port=5020) as modbus_client:
    sensor_values = read_sensor_values(modbus_client)
    mqtt_client.publish("modbus/sensor_data", str(sensor_values))

mqtt_client.loop_start()