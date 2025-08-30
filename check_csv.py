import pandas as pd

# Load the dataset
df = pd.read_csv('data/CombinedDataset.csv')

# Reset index to avoid out-of-bounds access and check for missing values
df = df.reset_index(drop=True)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# If there are missing values, you may want to drop or fill them
# For example, drop rows with missing values in either 'text' or 'label'
df = df.dropna(subset=['text', 'label'])

# Ensure that your DataFrame is ready to be passed into the Dataset
print(f"Dataset shape after cleaning: {df.shape}")

# Save the cleaned dataset if needed (optional)
# df.to_csv('CleanedDataset.csv', index=False)
