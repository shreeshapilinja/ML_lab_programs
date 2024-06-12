import pandas as pd
df = pd.read_csv("customer_orders.csv")
print("Original Dataset:",df)

# Remove duplicate rows
df = df.drop_duplicates()

# Print the dataset without duplicates
print("\nDataset without duplicates:",df)

# Remove columns with a single value
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col, axis=1, inplace=True)

print("\nDataset without columns with a single value:",df)

'''
name,age,gender,country,city
John,25,Male,USA,New York
Emily,31,Female,Canada,Toronto
Alex,19,Male,UK,London
Samantha,42,Female,USA,Los Angeles
Bob,28,Male,Canada,Montreal
'''