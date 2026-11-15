
# OASIS INFOBYTE - Customer Segmentation Analysis

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create output folder
os.makedirs("outputs", exist_ok=True)

# 1. Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("\n========== DATASET INFORMATION ==========")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
print(df.info())

# 2. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

# 3. Exploratory Data Analysis
print("\n========== EXPLORATORY DATA ANALYSIS ==========")

print("\nStatistical Summary:")
print(df.describe())

print("\nGender Distribution:")
print(df["Gender"].value_counts())

# Gender Distribution Chart
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender")
plt.title("Customer Gender Distribution")
plt.savefig("outputs/gender_distribution.png")
plt.show()

# Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=15, kde=True)
plt.title("Customer Age Distribution")
plt.savefig("outputs/age_distribution.png")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], bins=15, kde=True)
plt.title("Annual Income Distribution")
plt.savefig("outputs/income_distribution.png")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Spending Score (1-100)"], bins=15, kde=True)
plt.title("Customer Spending Score Distribution")
plt.savefig("outputs/spending_distribution.png")
plt.show()

# 4. Feature Selection
features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 5. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 6. Elbow Method
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.savefig("outputs/elbow_method.png")
plt.show()

# 7. Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(scaled_features)

print("\n========== CUSTOMER CLUSTERS ==========")
print(df[["CustomerID", "Gender", "Age",
          "Annual Income (k$)",
          "Spending Score (1-100)", "Cluster"]].head(10))

# 8. Customer Segmentation Visualization
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title("Customer Segmentation Using K-Means")
plt.savefig("outputs/customer_clusters.png")
plt.show()

# 9. Cluster Analysis
cluster_summary = df.groupby("Cluster").agg({
    "CustomerID": "count",
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean"
}).round(2)

cluster_summary = cluster_summary.rename(
    columns={"CustomerID": "Number of Customers"}
)

print("\n========== CLUSTER SUMMARY ==========")
print(cluster_summary)

# 10. Silhouette Score
score = silhouette_score(scaled_features, df["Cluster"])
print("\nSilhouette Score:", round(score, 3))

# 11. Save Results
df.to_csv("outputs/customer_segments.csv", index=False)
cluster_summary.to_csv("outputs/cluster_summary.csv")

print("\nProject completed successfully!")
print("Results saved in the outputs folder.")
