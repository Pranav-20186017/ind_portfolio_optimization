import pandas as pd
import json

# Read the Excel file
df = pd.read_excel('master.xlsx')

# Initialize an empty dictionary to store the data
data = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    ticker = row['Ticker']
    entry = {
        "name": row['Name'],
        "isin": row['ISIN'],
        "exchange": row['Exchange']
    }
    if ticker in data:
        # If the ticker already exists, append the new entry to the list
        if isinstance(data[ticker], list):
            data[ticker].append(entry)
        else:
            data[ticker] = [data[ticker], entry]
    else:
        # If the ticker doesn't exist, add it to the dictionary
        data[ticker] = entry

# Write the data to a JSON file
with open('output.json', 'w') as f:
    json.dump(data, f, indent=4)
