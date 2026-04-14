import pandas as pd
from sklearn.model_selection import train_test_split

# Split dataset into tran-test with stratify
# Load dataset
df = pd.read_csv('clickbait_data.csv')

# Split dataset with stratify on 'label' column
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Save split datasets
train_df.to_csv('clickbait_train.csv', index=False)
test_df.to_csv('clickbait_test.csv', index=False)

print("Dataset split completed. Train and test files saved.")

