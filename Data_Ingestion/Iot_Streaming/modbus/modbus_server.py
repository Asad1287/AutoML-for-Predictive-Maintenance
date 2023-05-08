from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.server import (
    StartSerialServer,
    StartTcpServer,
    StartTlsServer,
    StartUdpServer,
)
from pymodbus.constants import Endian
from pymodbus.payload import BinaryPayloadDecoder

def run_server():
    store = ModbusSlaveContext(
        di=None,
        co=None,
        hr=ModbusSequentialDataBlock(0, [0] * 8),  # 4 floating point values require 8 registers
        ir=None
    )
    context = ModbusServerContext(slaves=store, single=True)
    StartTcpServer()


    

if __name__ == "__main__":
    run_server()