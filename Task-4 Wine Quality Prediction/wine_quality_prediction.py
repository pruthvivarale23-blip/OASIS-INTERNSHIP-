import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("WineQT.csv")

print("Dataset Shape:", df.shape)

# Missing Values
print(df.isnull().sum())

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("Correlation_Heatmap.png")
plt.close()

# Quality Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.savefig("gWine_Quality_Distribution.png")
plt.close()

# Features and Target
X = df.drop('quality', axis=1)
y = df['quality']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

# SGD
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

print("\nSGD Accuracy:")
print(accuracy_score(y_test, sgd_pred))

# SVC
svc = SVC()
svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)

print("\nSVC Accuracy:")
print(accuracy_score(y_test, svc_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("\nProject Completed Successfully!")