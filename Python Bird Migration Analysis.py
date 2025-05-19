import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load and Prepare the Data
# ---------------------------
df = pd.read_csv("bird_migration_data.csv")
sns.set(style="whitegrid")

# ---------------------------------------------------
# 1. Top 10 Most Common Bird Species (Bar Plot)
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
species_counts = df["Species"].value_counts().head(10)
sns.barplot(x=species_counts.values, y=species_counts.index, palette="viridis")
plt.title("Top 10 Most Common Bird Species")
plt.xlabel("Count")
plt.ylabel("Species")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 2. Distribution of Flight Distances
# --------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(df["Flight_Distance_km"], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Flight Distances")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 3. Migration Success Rate by Region (Stacked Bar Chart)
# --------------------------------------------------------
plt.figure(figsize=(10, 6))
success_by_region = df.groupby("Region")["Migration_Success"].value_counts(normalize=True).unstack()
success_by_region.plot(kind="bar", stacked=True, colormap="Set2")
plt.title("Migration Success Rate by Region")
plt.ylabel("Proportion")
plt.xlabel("Region")
plt.legend(title="Migration Success")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 4. Correlation Heatmap of Numeric Variables
# ---------------------------------------------------
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=["float64", "int64"]).corr()
sns.heatmap(numeric_cols, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 5. Average Flight Distance by Top 10 Bird Species
# ---------------------------------------------------
top_species = df["Species"].value_counts().head(10).index
top_df = df[df["Species"].isin(top_species)]
plt.figure(figsize=(10, 6))
sns.barplot(data=top_df, x="Flight_Distance_km", y="Species", estimator='mean', palette="crest")
plt.title("Average Flight Distance by Top 10 Bird Species")
plt.xlabel("Average Distance (km)")
plt.ylabel("Species")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 6. Box Plot of Max Altitude by Habitat
# ---------------------------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="Habitat", y="Max_Altitude_m", palette="pastel")
plt.title("Maximum Altitude by Habitat Type")
plt.xlabel("Habitat")
plt.ylabel("Max Altitude (m)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 7. Count of Migrations by Start Month
# ---------------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Migration_Start_Month", order=df["Migration_Start_Month"].value_counts().index, palette="flare")
plt.title("Migration Counts by Start Month")
plt.xlabel("Month")
plt.ylabel("Number of Migrations")
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 8. Pair Plot of Flight Metrics
# ---------------------------------------------------
flight_metrics = df[["Flight_Distance_km", "Flight_Duration_hours", "Average_Speed_kmph"]]
sns.pairplot(flight_metrics)
plt.suptitle("Pair Plot of Flight Metrics", y=1.02)
plt.show()
