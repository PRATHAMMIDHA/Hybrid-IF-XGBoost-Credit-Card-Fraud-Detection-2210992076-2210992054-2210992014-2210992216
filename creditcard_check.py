import pandas as pd
df = pd.read_csv('creditcard.csv')
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Class Distribution:")
print(df['Class'].value_counts())
print("Missing values:", df.isnull().sum().sum())
print("Amount stats:\n", df['Amount'].describe())
print("Time stats:\n", df['Time'].describe())
