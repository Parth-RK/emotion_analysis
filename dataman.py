import pandas as pd

# Replace 'input.csv' with your CSV file path
df = pd.read_csv('emotion_data.csv')

# Select the top n rows
df_top_100 = df.head(500)

# Save to a new CSV file
df_top_100.to_csv('emotion_data_lite.csv', index=False)
print("Saved data to 'emotion_data_lite.csv'")