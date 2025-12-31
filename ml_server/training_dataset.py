import os
import pandas as pd
import glob

def concatenate_csvs(input_dir, output_file):
    """
    Finds all CSV files in the input directory and concatenates them into a single CSV file.
    
    Parameters:
    input_dir (str): Directory containing the CSV files to concatenate
    output_file (str): Path to the output CSV file
    """
    # Create the full path to the input directory
    input_path = os.path.join(input_dir, "*.csv")
    
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(input_path)
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to concatenate")
    
    # Initialize an empty list to store the dataframes
    dfs = []
    
    # Read each CSV file and append it to the list
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=";")
            dfs.append(df)
            print(f"Successfully read {file} with {len(df)} rows")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid CSV files were read. Exiting.")
        return
    
    # Concatenate all dataframes in the list
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write the combined dataframe to the output file
    combined_df.to_csv(output_file, index=False, sep=";")
    
    print(f"Successfully created {output_file} with {len(combined_df)} rows")

# Directory containing CSV files
input_directory = "../Files/TM"

# Output file path
output_csv = "../Files/training_dataset.csv"

# Call the function to concatenate CSV files
concatenate_csvs(input_directory, output_csv)
