"""
Customer Segmentation Project
Segments customers by demographics + purchase behavior using K-Means.

Run:  python customer_segmentation.py
Optional: put your own file named customers.csv next to this script with columns:
  CustomerID, Age, Gender, Annual_Income, Spending_Score, Purchase_Frequency,
  Avg_Order_Value, Recency_Days, Preferred_Category
(If no file is found, realistic sample data is generated automatically.)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT = "output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------- 1. LOAD / GENERATE DATA ----------
def make_data(n=600):
    # four hidden groups so clustering has real structure
    groups = [
        # age, income, spend, freq, aov, recency
        (24, 30, 75, 30, 35, 10),    # young trendy shoppers
        (45, 95, 80, 22, 160, 15),   # affluent big spenders
        (50, 60, 25, 6, 55, 90),     # cautious / occasional
        (35, 40, 20, 3, 30, 200),    # at-risk / lapsed
    ]
    cats = ["Electronics", "Fashion", "Home", "Grocery", "Beauty"]
    pref = [[.2, .5, .05, .05, .2], [.4, .2, .25, .05, .1],
            [.05, .1, .35, .45, .05], [.2, .2, .2, .2, .2]]
    rows = []
    for g, (a, inc, sp, fr, aov, rec) in enumerate(groups):
        k = n // 4
        rows.append(pd.DataFrame({
            "Age": rng.normal(a, 6, k).clip(18, 75).round(),
            "Annual_Income": rng.normal(inc, 10, k).clip(10, 150).round(1),
            "Spending_Score": rng.normal(sp, 8, k).clip(1, 100).round(),
            "Purchase_Frequency": rng.normal(fr, 4, k).clip(1, 60).round(),
            "Avg_Order_Value": rng.normal(aov, aov * .2, k).clip(5, 400).round(2),
            "Recency_Days": rng.normal(rec, rec * .3, k).clip(1, 365).round(),
            "Gender": rng.choice(["Male", "Female"], k),
            "Preferred_Category": rng.choice(cats, k, p=pref[g]),
        }))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1)
    df.insert(0, "CustomerID", range(1001, 1001 + len(df)))
    return df.reset_index(drop=True)

if os.path.exists("customers.csv"):
    df = pd.read_csv("customers.csv")
    print("Loaded customers.csv")
else:
    df = make_data()
    df.to_csv("customers.csv", index=False)
    print("No customers.csv found -> generated sample data (saved as customers.csv)")

print(df.shape, "\n", df.head(), "\n")

# ---------- 2. CLEAN ----------
df = df.drop_duplicates().copy()
num_cols = ["Age", "Annual_Income", "Spending_Score",
            "Purchase_Frequency", "Avg_Order_Value", "Recency_Days"]
num_cols = [c for c in num_cols if c in df.columns]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---------- 3. EDA ----------
df[num_cols].hist(figsize=(12, 7), bins=25, color="#4C78A8")
plt.suptitle("Feature Distributions")
plt.tight_layout(); plt.savefig(f"{OUT}/01_distributions.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
plt.imshow(df[num_cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
plt.yticks(range(len(num_cols)), num_cols)
plt.colorbar(); plt.title("Correlation Heatmap")
plt.tight_layout(); plt.savefig(f"{OUT}/02_correlation.png", dpi=150); plt.close()

# ---------- 4. SCALE + CHOOSE K ----------
X = StandardScaler().fit_transform(df[num_cols])
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(m.inertia_)
    sil.append(silhouette_score(X, m.labels_))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", color="green"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(f"{OUT}/03_choose_k.png", dpi=150); plt.close()

best_k = list(ks)[int(np.argmax(sil))]
print(f"Best k by silhouette: {best_k} (score={max(sil):.3f})")

# ---------- 5. FINAL MODEL ----------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["Segment"] = km.labels_

# ---------- 6. PROFILE SEGMENTS ----------
profile = df.groupby("Segment")[num_cols].mean().round(1)
profile["Customers"] = df["Segment"].value_counts().sort_index()
profile["Share_%"] = (profile["Customers"] / len(df) * 100).round(1)
if "Preferred_Category" in df.columns:
    profile["Top_Category"] = df.groupby("Segment")["Preferred_Category"].agg(lambda s: s.mode()[0])

# auto-name segments from their behavior
def name_segment(r):
    if r["Recency_Days"] > df["Recency_Days"].quantile(.7) and r["Purchase_Frequency"] < df["Purchase_Frequency"].median():
        return "At-Risk / Lapsed"
    if r["Avg_Order_Value"] > df["Avg_Order_Value"].quantile(.7):
        return "High-Value Loyalists"
    if r["Purchase_Frequency"] > df["Purchase_Frequency"].quantile(.6):
        return "Frequent Young Shoppers"
    return "Occasional / Price-Sensitive"

profile["Name"] = profile.apply(name_segment, axis=1)
df["Segment_Name"] = df["Segment"].map(profile["Name"])
print("\nSEGMENT PROFILE\n", profile.to_string())

profile.to_csv(f"{OUT}/segment_profile.csv")
df.to_csv(f"{OUT}/customers_segmented.csv", index=False)

# ---------- 7. VISUALIZE ----------
colors = plt.cm.Set2(np.linspace(0, 1, best_k))

pca = PCA(n_components=2).fit(X)
P = pca.transform(X)
plt.figure(figsize=(8, 6))
for s in range(best_k):
    plt.scatter(P[df.Segment == s, 0], P[df.Segment == s, 1], s=18,
                color=colors[s], label=f"{s}: {profile.loc[s, 'Name']}")
plt.title("Customer Segments (PCA 2D view)"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/04_segments_pca.png", dpi=150); plt.close()

if {"Annual_Income", "Spending_Score"} <= set(df.columns):
    plt.figure(figsize=(8, 6))
    for s in range(best_k):
        d = df[df.Segment == s]
        plt.scatter(d.Annual_Income, d.Spending_Score, s=18, color=colors[s], label=profile.loc[s, "Name"])
    plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score"); plt.legend()
    plt.title("Income vs Spending Score by Segment")
    plt.tight_layout(); plt.savefig(f"{OUT}/05_income_vs_spend.png", dpi=150); plt.close()

plt.figure(figsize=(6, 6))
plt.pie(profile["Customers"], labels=profile["Name"], autopct="%1.0f%%", colors=colors)
plt.title("Segment Size"); plt.tight_layout()
plt.savefig(f"{OUT}/06_segment_size.png", dpi=150); plt.close()

# normalized heatmap of segment traits
norm = (profile[num_cols] - profile[num_cols].min()) / (profile[num_cols].max() - profile[num_cols].min() + 1e-9)
plt.figure(figsize=(9, 4.5))
plt.imshow(norm.values, cmap="YlGnBu", aspect="auto")
plt.xticks(range(len(num_cols)), num_cols, rotation=30, ha="right")
plt.yticks(range(best_k), profile["Name"])
for i in range(best_k):
    for j, c in enumerate(num_cols):
        plt.text(j, i, profile.iloc[i][c], ha="center", va="center", fontsize=8)
plt.title("Segment Characteristics (cell = average value)")
plt.tight_layout(); plt.savefig(f"{OUT}/07_segment_heatmap.png", dpi=150); plt.close()

if "Preferred_Category" in df.columns:
    ct = pd.crosstab(df.Segment_Name, df.Preferred_Category, normalize="index") * 100
    ct.plot(kind="bar", stacked=True, figsize=(9, 5), colormap="tab20")
    plt.ylabel("% of segment"); plt.title("Category Preference by Segment")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(f"{OUT}/08_category_preference.png", dpi=150); plt.close()

# ---------- 8. BUSINESS RECOMMENDATIONS ----------
actions = {
    "High-Value Loyalists": "VIP program, early access, premium bundles, personal outreach.",
    "Frequent Young Shoppers": "Social-media campaigns, referral rewards, trend drops, app push offers.",
    "Occasional / Price-Sensitive": "Discount coupons, free-shipping thresholds, seasonal promos.",
    "At-Risk / Lapsed": "Win-back emails, 'we miss you' discount, survey to learn why they left.",
}
with open(f"{OUT}/insights.txt", "w") as f:
    f.write("CUSTOMER SEGMENTATION - KEY INSIGHTS\n" + "=" * 40 + "\n")
    f.write(f"Customers analysed: {len(df)} | Segments found: {best_k} | Silhouette: {max(sil):.3f}\n\n")
    for s, r in profile.iterrows():
        f.write(f"Segment {s} - {r['Name']} ({r['Customers']} customers, {r['Share_%']}%)\n")
        f.write(f"  Avg age {r['Age']}, income {r['Annual_Income']}k, order value {r['Avg_Order_Value']}, "
                f"frequency {r['Purchase_Frequency']}, recency {r['Recency_Days']} days\n")
        f.write(f"  Strategy: {actions.get(r['Name'], '')}\n\n")

print("\nDone! Everything is in the 'output' folder.")
