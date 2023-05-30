import struct
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadBuilder

def send_sensor_values(client, sensor_values, starting_address=0):
    builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Little)
    for value in sensor_values:
        builder.add_32bit_float(value)

    payload = builder.to_registers()
    client.write_registers(starting_address, payload)

sensor_values = [25.0, 30.5, 18.7, 22.1]

with ModbusTcpClient("localhost", port=5020) as client:
    send_sensor_values(client, sensor_values)