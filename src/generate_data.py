import json
import random

columns = ["age", "income", "gender", "score", "height", "weight", "city"]
operations = [
    # Selection
    ("Select the column {col}", "df['{col}']"),
    ("Select the first 10 rows of column {col}", "df['{col}'].head(10)"),
    ("Select the last 5 rows of column {col}", "df['{col}'].tail(5)"),
    # Filtering
    ("Select rows where {col} > {val}", "df[df['{col}'] > {val}]"),
    ("Select rows where {col} < {val}", "df[df['{col}'] < {val}]"),
    ("Select rows where {col} == {val}", "df[df['{col}'] == {val}]"),
    ("Select rows where {col} != {val}", "df[df['{col}'] != {val}]"),
    # Sorting
    ("Sort the DataFrame by {col} ascending", "df.sort_values('{col}')"),
    ("Sort the DataFrame by {col} descending", "df.sort_values('{col}', ascending=False)"),
    # Aggregation
    ("Calculate the average of {col}", "df['{col}'].mean()"),
    ("Calculate the sum of {col}", "df['{col}'].sum()"),
    ("Calculate the maximum of {col}", "df['{col}'].max()"),
    ("Calculate the minimum of {col}", "df['{col}'].min()"),
    ("Calculate the median of {col}", "df['{col}'].median()"),
    ("Calculate the standard deviation of {col}", "df['{col}'].std()"),
    # Grouping
    ("Group by {col} and count", "df.groupby('{col}').size()"),
    ("Group by {col} and calculate the average of {val_col}", "df.groupby('{col}')['{val_col}'].mean()"),
    # Cleaning
    ("Drop missing values", "df.dropna()"),
    ("Drop duplicate rows", "df.drop_duplicates()"),
]

value_cols = ["age", "income", "score", "height", "weight"]

data = []
for _ in range(1000):
    template, code_template = random.choice(operations)
    col = random.choice(columns)
    val_col = random.choice(value_cols)
    val = random.randint(10, 100)
    
    instruction = template.format(col=col, val=val, val_col=val_col)
    code = code_template.format(col=col, val=val, val_col=val_col)
    
    data.append({"instruction": instruction, "output": code})

# Save to JSON
with open("./data/train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Generated 1000 English training samples: train_dataset.json")
