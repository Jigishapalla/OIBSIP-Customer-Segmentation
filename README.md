# Customer Segmentation Analysis

## Project Overview
This project performs Customer Segmentation using Exploratory Data Analysis (EDA) and K-Means Clustering.

The objective is to group mall customers based on their annual income and spending score to understand customer purchasing behaviour.

## Dataset
Mall Customers Dataset

The dataset contains 200 customer records with the following columns:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering

## Project Workflow
1. Load the customer dataset.
2. Perform data cleaning.
3. Analyze customer demographics.
4. Visualize customer distributions.
5. Apply feature scaling.
6. Use the Elbow Method to study the optimal number of clusters.
7. Apply K-Means Clustering.
8. Analyze and visualize five customer segments.

## Customer Segments

| Cluster | Customer Group | Customers |
|---|---|---:|
| 0 | Average Spending Customers | 81 |
| 1 | High Income – High Spending | 39 |
| 2 | Low Income – High Spending | 22 |
| 3 | High Income – Low Spending | 35 |
| 4 | Low Income – Low Spending | 23 |

## Key Findings
- The dataset contains 200 customers.
- No missing values or duplicate rows were found.
- Five customer segments were identified.
- Cluster 0 is the largest customer group.
- Cluster 1 represents high-income customers with high spending scores.
- Cluster 3 represents high-income customers with low spending scores.

## Business Recommendations
- Provide loyalty rewards to high-value customers.
- Offer personalized promotions to customers with high income but low spending.
- Promote affordable products to budget-conscious customers.
- Develop customer retention strategies for average-spending customers.

## Project Outputs
The outputs folder contains:
- Gender distribution chart
- Age distribution chart
- Annual income distribution chart
- Spending score distribution chart
- Elbow method chart
- Customer segmentation chart
- Customer segments CSV
- Cluster summary CSV

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python customer_segmentation.py
```

## Conclusion
This project demonstrates how K-Means Clustering can be used to identify different customer groups and support data-driven marketing decisions.