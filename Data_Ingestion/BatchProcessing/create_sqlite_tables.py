import sqlite3

# Connect to the SQLite database (or create a new one if it doesn't exist)
connection = sqlite3.connect("test_db.db")

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Define the table schema
table_schema = '''
CREATE TABLE IF NOT EXISTS sensors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor_name TEXT NOT NULL,
    sensor_type TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
'''

# Execute the schema creation statement
cursor.execute(table_schema)

# Commit the changes and close the connection
connection.commit()
connection.close()






