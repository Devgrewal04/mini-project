import pandas as pd

# Load original CSV
input_path = "data/dataset.csv"  # Replace with your original filename
output_path = "data/CombinedDataset.csv"

# Load CSV
df = pd.read_csv(input_path)

# Combine selected columns into a single text feature
df['text'] = (
    "Distraction: " + df['10. How often do you get distracted by Social media when you are busy doing something?'].astype(str) + ". " +
    "Restlessness: " + df["11. Do you feel restless if you haven't used Social media in a while?"].astype(str) + ". " +
    "Worries: " + df['13. On a scale of 1 to 5, how much are you bothered by worries?'].astype(str) + ". " +
    "Concentration: " + df['14. Do you find it difficult to concentrate on things?'].astype(str)
)

# Binary label: 1 = distress, 0 = not distress (based on feeling depressed or down)
df['label'] = df['18. How often do you feel depressed or down?'].apply(lambda x: 1 if int(x) >= 3 else 0)

# Final clean dataset
df_final = df[['text', 'label']]

# Save to new CSV
df_final.to_csv(output_path, index=False)

print("✅ Data prepared and saved to:", output_path)
