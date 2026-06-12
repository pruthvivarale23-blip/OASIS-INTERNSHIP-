import pandas as pd
import numpy as np

print("========== DATA CLEANING PROJECT ==========")

# Load Dataset
df = pd.read_csv("AB_NYC_2019.csv")

# ---------------------------------
# STEP 1: Basic Information
# ---------------------------------
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

# ---------------------------------
# STEP 2: Missing Values
# ---------------------------------
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill Numeric Missing Values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill Text Missing Values
text_cols = df.select_dtypes(include='object').columns

for col in text_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------------
# STEP 3: Duplicate Records
# ---------------------------------
print("\nDuplicate Rows Before Removal:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDuplicate Rows After Removal:")
print(df.duplicated().sum())

# ---------------------------------
# STEP 4: Standardization
# ---------------------------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("\nStandardized Columns:")
print(df.columns)

# ---------------------------------
# STEP 5: Outlier Detection
# ---------------------------------
print("\nOutlier Detection")

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col}: {len(outliers)} outliers")

# ---------------------------------
# STEP 6: Final Validation
# ---------------------------------
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

# ---------------------------------
# STEP 7: Save Clean Dataset
# ---------------------------------
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File Name: cleaned_dataset.csv")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(x=df['price'])
plt.title("Price Outlier Detection")
plt.savefig("Price_Outlier_Detection.png")
plt.close()