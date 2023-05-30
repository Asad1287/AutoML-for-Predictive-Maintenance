import pandas as pd
import os 
# Define the chunk size (you can adjust based on your system's memory)
chunk_size = 10000

# the target csv file
TARGET_CSV_FILE = ""

DESTINATION_DIR = ""


# Read the large csv file with specified chunksize 
df_chunk = pd.read_csv(TARGET_CSV_FILE, chunksize=chunk_size)

chunk_list = []  # append each chunk df here 

# Each chunk is a dataframe
for chunk_number, chunk in enumerate(df_chunk):
    chunk_list.append(chunk)

# Depending on the number of chunks, this may not divide evenly by 10.
# So we will find out how many rows each subdivided dataframe should have
rows_per_chunk = len(pd.concat(chunk_list)) // 10

# Now we create the subdivided chunks
subdivided_chunks = []

# We will use a rolling index to keep track of where to slice the dataframe
rolling_index = 0

# Use a for loop to create each subdivided chunk
for _ in range(10):
    # Slice the dataframe
    subdivided_chunk = pd.concat(chunk_list)[rolling_index:rolling_index + rows_per_chunk]
    # Append it to our list
    subdivided_chunks.append(subdivided_chunk)
    # Update the rolling index
    rolling_index += rows_per_chunk

# Now you can save each subdivided chunk to a new csv
for i, subdivided_chunk in enumerate(subdivided_chunks):
    subdivided_chunk.to_csv(os.path.join(DESTINATION_DIR, f'data_chunk_{i}.csv'), index=False)
