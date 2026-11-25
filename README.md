# Customer Segmentation Project

Segmenting customers by behavior and demographics using K-Means clustering in Python (scikit-learn).

## Method
- Cleaned data and scaled features (age, income, spending score, purchase frequency, order value, recency)
- Chose the number of clusters with the elbow method and silhouette score
- Trained a K-Means model and profiled each segment

## Segments found
- Frequent Young Shoppers: young, high frequency, recent purchases
- High-Value Loyalists: high income, large orders, high spending
- At-Risk / Lapsed: low frequency, long time since last purchase

## Files
- customer_segmentation.py: full code
- segment_profile.csv: average traits of each segment
- insights.txt: findings and marketing strategies
- Images: segment visualization and characteristics heatmap
