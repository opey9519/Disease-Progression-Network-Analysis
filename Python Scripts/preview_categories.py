import os
import pandas as pd

# Path to your folder with CSV files
data_dir = "./EHRShot_sampled_2000patients"   # change if needed

# Loop through each CSV file
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        print(f"\n=== {file} ===")
        
        try:
            # Only load 5 rows
            df = pd.read_csv(file_path, nrows=50)
            df.to_csv(os.path.splitext(file)[0] + "_10_rows" + ".csv", index=False)
            
    
        except Exception as e:
            print(f"Error reading {file}: {e}")

