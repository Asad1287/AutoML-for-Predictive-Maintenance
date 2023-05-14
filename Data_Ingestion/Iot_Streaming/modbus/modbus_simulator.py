from pymodbus.server import StartAsyncSimulatorServer

async def run():
    simulator = StartAsyncSimulatorServer(
        modbus_server="my server",
        modbus_device="my device",
        modbus_port=5020,
        modbus_address=1,
        modbus_data_store=None,
        modbus_custom_data_store=None,
        modbus_message_store=None,
    )
    await simulator.run_server()
    