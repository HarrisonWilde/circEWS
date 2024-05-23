import pandas as pd
from tqdm import tqdm

# Define the columns you want to keep
cols_to_keep = ["SUBJECT_ID", "CHARTTIME", "VALUE", "ITEMID"]

# Initialize an empty dataframe to store the filtered data
filtered_data = pd.DataFrame(columns=cols_to_keep)

file_name = "/data/harry/mimiciii/CHARTEVENTS.csv"

# Define the chunk size
chunksize = 10**6

# Get the number of chunks
num_chunks = 330712484 // chunksize

# Read the file in chunks
for chunk in tqdm(pd.read_csv(file_name, chunksize=chunksize), total=num_chunks):
    # Filter the chunk
    filtered_chunk = chunk[chunk["ITEMID"] == 226730][cols_to_keep]
    # Append the filtered chunk to the filtered_data dataframe
    filtered_data = pd.concat([filtered_data, filtered_chunk])

# Now, filtered_data contains only the rows where ITEMID == 226730 and only the columns you want
