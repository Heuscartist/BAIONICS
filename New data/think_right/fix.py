import os
import pandas as pd

# Define input and output directories
input_dir = r'C:\Users\adilm\OneDrive\Desktop\NEW FYP\New data\think_right'
output_dir = os.path.join(input_dir, 'processed')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to process each CSV file
def process_csv(file_path, output_path):
    df = pd.read_csv(file_path)  # Read CSV file
    if len(df) >= 499:
        sampled_df = df.sample(n=500)  # Sample 500 records if available
    else:
        print(f"Ignoring file {file_path} as it has less than 500 records.")
        return  # Ignore file with less than 500 records
    sampled_df.to_csv(output_path, index=False)  # Write sampled data to new CSV file

# Iterate over CSV files in input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.csv', 'processed.csv'))
        process_csv(file_path, output_path)

print("Processing completed.")
