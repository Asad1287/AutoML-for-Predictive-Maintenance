import snap7
from snap7.util import DB_Row
from snap7.snap7types import S7AreaDB

def main():
    plc_ip = '192.168.1.1'  # Replace with the IP address of your S7-1500 PLC

    # Create a PLC client and connect
    plc = snap7.client.Client()
    plc.connect(plc_ip, 0, 1)

    # Read data from the PLC (e.g., a bit from a memory address)
    db_number = 1
    address = 0
    size = 1
    data = plc.read_area(S7AreaDB, db_number, address, size)
    print(f"Data at DB{db_number}: {data}")

    # Write data to the PLC (e.g., set a bit in a memory address)
    data[0] ^= 1  # Toggle the first bit in the data bytearray
    plc.write_area(S7AreaDB, db_number, address, data)
    print(f"Updated data at DB{db_number}: {data}")

    # Disconnect from the PLC
    plc.disconnect()
    print(f"Disconnected from PLC {plc_ip}")

if __name__ == "__main__":
    main()
