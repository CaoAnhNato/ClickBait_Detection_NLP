import pandas as pd

df = pd.read_csv('NC_train.csv')

print(df.value_counts('label'))