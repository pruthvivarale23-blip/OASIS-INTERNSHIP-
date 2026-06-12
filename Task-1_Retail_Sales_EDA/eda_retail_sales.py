import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("retail_sales_dataset.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive Statistics
print("\nStatistics:")
print(df.describe())

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])

# Monthly Sales
df['Month'] = df['Date'].dt.month_name()

monthly_sales = df.groupby('Month')['Total Amount'].sum()

# Plot Monthly Sales Trend
plt.figure(figsize=(10,5))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("monthly_sales_trend.png")
plt.close()

# Product Category Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Product Category')
plt.title("Product Category Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gproduct_category_distribution.png")
plt.close()

# Gender-wise Sales
gender_sales = df.groupby('Gender')['Total Amount'].sum()

plt.figure(figsize=(6,6))
gender_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Gender-wise Sales")
plt.ylabel("")
plt.savefig("gGender-wise_Sales.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(
    df[['Age','Quantity','Price per Unit','Total Amount']].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.savefig("Correlation_Heatmap.png")
plt.close()

print("\nEDA Completed Successfully!")