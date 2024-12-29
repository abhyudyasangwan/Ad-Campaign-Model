# Ad Campaign Performance Analysis

This project analyzes digital marketing campaign performance using advanced data preprocessing, feature engineering, visualization, and machine learning techniques to optimize decision-making and budget allocation. It helps develop a feedback mechanism using GPT-4 LLM techniques.

---

## Table of Contents
1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Visualization](#visualization)
5. [Campaign Decisions](#campaign-decisions)
6. [Model Training](#model-training)
7. [Optimization](#optimization)
8. [Suggested Actions](#suggested-actions)
9. [Usage](#usage)
10. [Requirements](#requirements)

---

## Overview
This project focuses on:
- Cleaning and preprocessing advertising data.
- Engineering key performance metrics like CPC (Cost Per Click) and ROAS (Return on Ad Spend).
- Visualizing data to identify trends and outliers.
- Training a Random Forest model to predict ROAS for future campaigns.
- Optimizing decision-making based on actionable insights.
- Creation of a feedback mechanism utilizing GPT-4 LLM techniques.

---

## Data Preprocessing

### Steps:
1. Removed irrelevant columns: `Date`, `Latitude`, and `Longitude`.

   **Explanation:**

   **Date:** Since all the dates are categorized into seasons in the 'Campaign' column, it provides a broader time frame to evaluate the performance of the campaign across different seasons.

   **Longitude and Latitude:** Since the 'City/Location' feature is already available, we will integrate the regional performance dependency within it, rather than using Longitudes and Latitudes, which can vary significantly.
2. One-hot encoded categorical columns such as `Campaign`, `City/Location`, `Channel`, `Device`, and `Ad`.
3. Renamed columns for clarity.
4. Converted percentage values (e.g., CTR) to numeric format.
5. Dropped redundant one-hot encoded columns for better performance.

**Code Example:**
```python
import pandas as pd

# Load data
df = pd.read_csv("Market.csv")

# Drop unnecessary columns
df = df.drop(['Date', 'Latitude', 'Longitude'], axis=1)

# One-hot encode categorical columns
categorical_columns = ['Campaign', 'City/Location', 'Channel', 'Device', 'Ad']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Clean and rename columns
df.rename(columns={
    'CTR, %': 'CTR', 'Spend, GBP': 'Spend', 'Likes (Reactions)': 'Likes',
    'Total conversion value, GBP': 'conversion_value', 'Daily Average CPC': 'daily_average_cpc'
}, inplace=True)

# Convert CTR to numeric
df['CTR'] = df['CTR'].str.replace('%', '').astype(float)
```

---

## Feature Engineering

### Key Metrics:
1. **CPC (Cost Per Click):** Measures the cost per individual ad click.
   - Formula: `CPC = Spend / Conversions`
2. **ROAS (Return on Ad Spend):** Evaluates revenue generated per ad spend.
   - Formula: `ROAS = conversion_value / Spend`

**Code Example:**
```python
# Compute CPC and ROAS
df['CPC'] = df['Spend'] / df['Conversions']
df['ROAS'] = df['conversion_value'] / df['Spend']

# Handle missing and infinite values
df.fillna(df.mean(), inplace=True)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ROAS', 'CPC'])
```
Standardizing:
---

We will standardize the columns to improve the results. However, the columns that have been one-hot encoded will not be standardized, as they contain binary values (1 or 0). These columns are not individually assigned 0 or 1; instead, they are represented by codes like '01' or '00'. For instance, 'City/Location_London' and 'City/Location_Manchester' have assigned values, while 'City/Location_Birmingham' is not assigned a value. Since the encoding of the others already covers it (e.g., '00' represents Birmingham), it should not be standardized.

## Visualization

Visualizations help identify patterns and outliers in the dataset.

**Techniques:**
- Histograms
- Boxplots
- Pairplots
- Correlation Heatmaps
- Scatterplots

**Code Example:**
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_dataset(df):

    sns.set(style="whitegrid")
    print("Dataset Overview:\n")
    print(df.info())
    print("\nSummary Statistics:\n")
    print(df.describe())

    # 1. Histograms
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
        plt.suptitle('Histograms for Numerical Features', fontsize=16)
        plt.show()

    # 2. Boxplots
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}', fontsize=14)
        plt.show()

    # 3. Pairplot
    if len(num_cols) <= 5:
        sns.pairplot(df[num_cols], diag_kind='kde', plot_kws={'alpha': 0.7})
        plt.suptitle('Pairplot of Numerical Features', fontsize=16)
        plt.show()

    # 4. Correlation Heatmap
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, cbar=True)
        plt.title('Correlation Heatmap', fontsize=16)
        plt.show()

    # 5. Bar Plots for Categorical Columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
        plt.title(f'Bar Plot of {col}', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    # 6. Scatterplots
    if len(num_cols) > 1:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=df[num_cols[i]], y=df[num_cols[j]], alpha=0.6, color='purple')
                plt.title(f'Scatter Plot: {num_cols[i]} vs {num_cols[j]}', fontsize=14)
                plt.xlabel(num_cols[i])
                plt.ylabel(num_cols[j])
                plt.show()

visualize_dataset(df)
```

---

## Campaign Decisions

**Rules:**
1. **Pause Campaign:** When CTR < 1%.

   **Explanation:**

   **What is CTR?**

   Click-Through Rate (**CTR**) is a metric that measures the percentage of users who click on an advertisement after seeing it. A higher CTR indicates better engagement and relevance of the ad to the audience.
   Ranges of CTR for Various Performance Levels CTR can vary based on the platform (e.g., Google Ads, Facebook Ads) and the industry.

   **BENCHMARKS:**

   **Excellent Performance:** CTR > 3%

   Indicates the ad is highly engaging and relevant to the audience.

   **Good Performance:** CTR between 1% and 3%

   Suggests the ad is performing well and meeting expectations.

   **Poor Performance:** CTR < 1%

   Indicates low audience interest or relevance, requiring attention to targeting or ad content.


**Why is 1% a Good Cap?**

Threshold for Relevance:

1) A CTR of less than 1% often means the ad fails to engage the audience, signaling potential issues with targeting, creatives, or messaging. Efficient Use of Budget:

2) Ads with CTR < 1% may waste budget on impressions without generating enough clicks, making the campaign cost-inefficient. Industry Benchmark:

3) In most industries, a 1% CTR is considered the minimum benchmark for an ad to be deemed effective.

2. **Increase Budget:** When ROAS > 4.

   **What is ROAS?**

   Return on Ad Spend (ROAS) is a metric that measures the revenue generated for every unit of currency spent on advertising. A higher ROAS indicates a highly efficient and profitable advertising campaign.
   
   **BENCHMARKS:**

   **Excellent Performance:** ROAS > 4
  
   The campaign is generating significant revenue compared to the spend.

   **Good Performance:** ROAS between 2 and 4

   The campaign is profitable and meeting expectations.

   **Poor Performance:** ROAS < 2

   The campaign is underperforming and may need optimization.

**Why is ROAS > 4 a Good Benchmark to Increase Budget?**

1) A ROAS greater than 4 means that for every ₹1 spent on advertising, at least ₹4 in revenue is generated. This represents a highly profitable campaign. Scalable Performance:

2) Campaigns with high ROAS suggest that the current strategy is effective and scalable. Increasing the budget can amplify returns while maintaining efficiency. Optimal Allocation of Resources:

3) Increasing the budget for high-ROAS campaigns ensures that resources are focused on the most profitable campaigns, maximizing overall revenue.

3. **Decrease Budget:** When ROAS < 1.5.

**Code Example:**
```python
# Campaign decision rules
df['Pause'] = df['CTR'].apply(lambda ctr: 1 if ctr < 1.0 else 0)
df['increase_budget'] = df['ROAS'].apply(lambda roas: 1 if roas > 4.0 else 0)
df['decrease_budget'] = df['ROAS'].apply(lambda roas: 1 if roas < 1.5 else 0)
```

---

## Model Training

Trained a Random Forest Regressor to predict ROAS values.

**Steps:**
1. Preprocessed data by scaling numerical features.
2. Split data into training and testing sets.
3. Trained the model using a Random Forest Regressor.

**Code Example:**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X = df.drop(columns=['conversion_value', 'ROAS', 'Pause', 'increase_budget', 'decrease_budget'])
y = df['ROAS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R² Score: {r2_score(y_test, y_pred)}")
```

---

## Optimization

**Key Improvements:**
1. Removed outlier ranges from the dataset.
2. Increased the number of estimators in the Random Forest model for better accuracy.

**Code Example:**
```python
# Remove outliers
ranges_to_delete = [(2793, 3300), (5200, 6861), (8133, 8754)]
indices_to_drop = [i for start, end in ranges_to_delete for i in range(start, end + 1)]
df = df.drop(indices_to_drop).reset_index(drop=True)

# Retrain model with optimized parameters
model = RandomForestRegressor(n_estimators=2000, random_state=42)
model.fit(X_train, y_train)
```

---

## Suggested Actions

Use the trained model to predict ROAS for future campaigns and optimize ad spend accordingly.

**Code Example:**
```python
def predict_roas(model, scaler, feature_sequence):
    feature_sequence_scaled = scaler.transform([feature_sequence])
    return model.predict(feature_sequence_scaled)[0]
```

---

## Usage
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the scripts to preprocess data, train models, and visualize results.

---

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

