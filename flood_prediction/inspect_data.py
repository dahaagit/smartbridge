import kagglehub
import os
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("s3programmer/flood-risk-in-india")
print("Path to dataset files:", path)

# Find the csv file
csv_file = None
for f in os.listdir(path):
    if f.endswith('.csv'):
        csv_file = os.path.join(path, f)
        break

if csv_file:
    print("Found CSV:", csv_file)
    df = pd.read_csv(csv_file)
    print("Columns:", df.columns.tolist())
    print(df.head())
else:
    print("No CSV found.")
