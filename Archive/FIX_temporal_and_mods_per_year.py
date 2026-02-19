"""
FIX_temporal_and_mods_per_year.py
=================================
Replicates the analysis pipeline from notebook 03 using local parquet files,
then computes corrected temporal and mods-per-year statistics per cluster.

Key fix: users with activity_span < 30 days are excluded from mods_per_year
to avoid division-by-near-zero inflation.

Run: python "Actually Analysis/FIX_temporal_and_mods_per_year.py"
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats.mstats import winsorize
from scipy.stats import yeojohnson
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"c:\Users\nicta\Desktop\API_Paper_ChiPlay\mod_data"

# ── 1. Load parquet files ──────────────────────────────────────────────────────
print("Loading parquet files...")
authors = pd.read_parquet(f"{DATA_DIR}/Authors.parquet")
mods    = pd.read_parquet(f"{DATA_DIR}/CleanedModData.parquet")
cats    = pd.read_parquet(f"{DATA_DIR}/GameCategories.parquet")

print(f"  Authors:        {len(authors):,} rows")
print(f"  CleanedModData: {len(mods):,} rows")
print(f"  GameCategories: {len(cats):,} rows")

# ── 2. Replicate the SQL query from notebook 03 ───────────────────────────────
print("\nReplicating SQL aggregation...")

# Filter authors: deleted=0, last_active >= 2024-01-01
# Note: 'deleted' column is bool in parquet
authors_filtered = authors[
    (~authors["deleted"]) &
    (authors["last_active"].notna()) &
    (authors["last_active"] >= "2024-01-01")
].copy()
print(f"  Authors after filter: {len(authors_filtered):,}")

# Cast bool to int for aggregation
mods["contains_adult_content"] = mods["contains_adult_content"].astype(int)

# Join mods with GameCategories to get new_group_category
mods_with_cats = mods.merge(
    cats[["game_id", "category_id", "new_group_category"]],
    on=["game_id", "category_id"],
    how="left"
)

# Aggregate mods to user level
mod_agg = mods_with_cats.groupby("member_id").agg(
    first_mod_created_date=("created_timestamp", "min"),
    last_mod_created_date=("created_timestamp", "max"),
    total_domains=("domain_name", "nunique"),
    total_categories=("new_group_category", "nunique"),
    endorsements_received=("endorsement_count", "count"),
    adult_content_count=("contains_adult_content", "sum"),
    all_mod_downloads=("mod_downloads", "sum"),
    all_unique_mod_downloads=("mod_unique_downloads", "sum"),
).reset_index()

# Join authors with mod aggregates
df = authors_filtered.merge(mod_agg, on="member_id", how="left")

# Compute derived columns matching the SQL query
df["active_days"] = (df["last_active"] - df["joined"]).dt.days
df["published_mod_count"] = df["mod_count"]
df["unpublished_mod_count"] = df["owned_mod_count"] - df["mod_count"]
df["all_mods_count"] = df["owned_mod_count"]

# Convert unix timestamps to datetime for mod dates
df["first_mod_created_date_dt"] = pd.to_datetime(df["first_mod_created_date"], unit="s", errors="coerce")
df["last_mod_created_date_dt"] = pd.to_datetime(df["last_mod_created_date"], unit="s", errors="coerce")

# Compute mod_creation_days_since_joined (days from join to first mod)
df["mod_creation_days_since_joined"] = (df["first_mod_created_date_dt"] - df["joined"]).dt.days

print(f"  Joined dataset: {len(df):,} rows")

# ── 3. Feature preparation (matching notebook 03) ─────────────────────────────
print("\nPreparing features...")

# Select the 13 features used in the Devotion Score
feature_cols = [
    "last_mod_created_date",
    "published_mod_count",
    "unpublished_mod_count",
    "all_mods_count",
    "endorsements_given",
    "posts",
    "kudos",
    "views",
    "endorsements_received",
    "adult_content_count",
    "all_mod_downloads",
    "all_unique_mod_downloads",
    "mod_creation_days_since_joined",
]

df_numeric = df[feature_cols].copy().fillna(0)

# ── 4. Preprocessing (matching notebook 03) ───────────────────────────────────
print("Applying transformations...")

df_transformed = df_numeric.copy()

# Log1p transforms (clip negatives to 0 first — a few rows have negative posts)
for col in ["endorsements_given", "posts", "kudos"]:
    df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))

# Winsorize + cube-root for views
df_transformed["views"] = winsorize(df_transformed["views"], limits=[0.02, 0.02])
df_transformed["views"] = np.cbrt(df_transformed["views"])

# Yeo-Johnson transforms
yj_cols = [
    "published_mod_count", "unpublished_mod_count", "all_mods_count",
    "endorsements_received", "adult_content_count",
    "all_mod_downloads", "all_unique_mod_downloads",
]
for col in yj_cols:
    df_transformed[col], _ = yeojohnson(df_transformed[col])

# Also drop total_categories and total_domains if they were added
# (they're not in the 13 feature list, so we're fine)

# Z-standardise
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_transformed)

print(f"  Scaled data shape: {data_scaled.shape}")

# ── 5. PCA ─────────────────────────────────────────────────────────────────────
print("\nRunning PCA...")
pca = PCA(n_components=7)
pca_data = pca.fit_transform(data_scaled)

print("  Explained variance ratios:", np.round(pca.explained_variance_ratio_, 4))
print(f"  Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# ── 6. Devotion Score ──────────────────────────────────────────────────────────
print("\nComputing Devotion Score...")
loadings = np.abs(pca.components_)
explained_variance = pca.explained_variance_ratio_
feature_weights = np.dot(explained_variance, loadings)
devotion_scores = np.dot(data_scaled, feature_weights)

df["Devotion_Score"] = devotion_scores
print(f"  Devotion Score - Mean: {devotion_scores.mean():.4f}, "
      f"Median: {np.median(devotion_scores):.4f}, "
      f"Std: {devotion_scores.std():.4f}")

# ── 7. K-Means Clustering ─────────────────────────────────────────────────────
print("\nRunning K-Means clustering (k=4)...")
kmeans = KMeans(n_clusters=4, n_init=50, random_state=42)
df["Cluster"] = kmeans.fit_predict(pca_data)

# Sort clusters by mean Devotion Score so labelling is consistent
cluster_means = df.groupby("Cluster")["Devotion_Score"].mean().sort_values()
cluster_map = {old: new for new, old in enumerate(cluster_means.index)}
df["Devotion_Cluster"] = df["Cluster"].map(cluster_map)

print("  Cluster sizes:")
for c in sorted(df["Devotion_Cluster"].unique()):
    n = (df["Devotion_Cluster"] == c).sum()
    pct = n / len(df) * 100
    mean_d = df.loc[df["Devotion_Cluster"] == c, "Devotion_Score"].mean()
    print(f"    Cluster {c}: {n:,} ({pct:.1f}%), mean Devotion = {mean_d:.2f}")

# ── 8. Temporal Analysis (THE FIX) ────────────────────────────────────────────
print("\n" + "="*70)
print("TEMPORAL ANALYSIS (FIXED)")
print("="*70)

# Activity span = last_mod - first_mod (in days)
df["activity_span_days"] = (
    df["last_mod_created_date_dt"] - df["first_mod_created_date_dt"]
).dt.days
df["activity_span_years"] = df["activity_span_days"] / 365.25

print("\n-- Original computation (for comparison) --")
df_orig = df.copy()
df_orig["activity_span_years_orig"] = df_orig["activity_span_years"].replace(0, np.nan)
df_orig["mods_per_year_orig"] = df_orig["all_mods_count"] / df_orig["activity_span_years_orig"]

orig_stats = df_orig.groupby("Devotion_Cluster").agg(
    span_days_mean=("activity_span_days", "mean"),
    span_days_median=("activity_span_days", "median"),
    span_years_mean=("activity_span_years", "mean"),
    mods_per_year_mean=("mods_per_year_orig", "mean"),
).round(2)
print(orig_stats.to_string())
print("\n** Note: mods_per_year_mean is inflated by users with tiny spans")

# ── FIXED computation ──
print("\n-- Fixed computation: floor span at 365 days for rate calc --")
df["activity_span_years_floored"] = np.maximum(df["activity_span_days"], 365) / 365.25
df["mods_per_year_floored"] = df["all_mods_count"] / df["activity_span_years_floored"]

floored_stats = df.groupby("Devotion_Cluster").agg(
    span_days_mean=("activity_span_days", "mean"),
    span_days_median=("activity_span_days", "median"),
    mods_per_year_floored_mean=("mods_per_year_floored", "mean"),
    mods_per_year_floored_median=("mods_per_year_floored", "median"),
).round(2)
print(floored_stats.to_string())

print("\n-- Fixed computation: exclude users with span < 30 days --")
df_30plus = df[df["activity_span_days"] >= 30].copy()
df_30plus["mods_per_year_clean"] = df_30plus["all_mods_count"] / df_30plus["activity_span_years"]

clean_stats = df_30plus.groupby("Devotion_Cluster").agg(
    n_users=("member_id", "count"),
    span_days_mean=("activity_span_days", "mean"),
    span_days_median=("activity_span_days", "median"),
    mods_per_year_mean=("mods_per_year_clean", "mean"),
    mods_per_year_median=("mods_per_year_clean", "median"),
    total_mods_mean=("all_mods_count", "mean"),
).round(2)
print(clean_stats.to_string())

print("\n-- Active days (joined -> last_active) per cluster --")
active_days_stats = df.groupby("Devotion_Cluster").agg(
    active_days_mean=("active_days", "mean"),
    active_days_median=("active_days", "median"),
).round(0)
print(active_days_stats.to_string())

# ── 9. Summary of what the paper should say ────────────────────────────────────
print("\n" + "="*70)
print("RECOMMENDED PAPER VALUES")
print("="*70)

print("\n1. APPROACH A -- Floor span at 1 year for rate calculation:")
print("   This avoids division-by-tiny-numbers while keeping all users.")
print(floored_stats[["mods_per_year_floored_mean", "mods_per_year_floored_median"]].to_string())

print("\n2. APPROACH B — Exclude users with span < 30 days:")
print("   More honest, but reduces sample size significantly for some clusters.")
print(clean_stats[["n_users", "mods_per_year_mean", "mods_per_year_median"]].to_string())

print("\n3. Active days (joined -> last_active) -- reliable, use in appendix table:")
print(active_days_stats.to_string())

# ── 10. Preprocessing skewness table for appendix ─────────────────────────────
print("\n" + "="*70)
print("PREPROCESSING SKEWNESS TABLE (for appendix)")
print("="*70)

skew_before = df_numeric.skew().round(2)
skew_after = pd.DataFrame(data_scaled, columns=feature_cols).skew().round(2)

skew_table = pd.DataFrame({
    "Feature": feature_cols,
    "Raw_Skewness": skew_before.values,
    "Post_Transform_Skewness": skew_after.values,
    "Transform": [
        "None",            # last_mod_created_date
        "Yeo-Johnson",     # published_mod_count
        "Yeo-Johnson",     # unpublished_mod_count
        "Yeo-Johnson",     # all_mods_count
        "log1p",           # endorsements_given
        "log1p",           # posts
        "log1p",           # kudos
        "Winsorize+cbrt",  # views
        "Yeo-Johnson",     # endorsements_received
        "Yeo-Johnson",     # adult_content_count
        "Yeo-Johnson",     # all_mod_downloads
        "Yeo-Johnson",     # all_unique_mod_downloads
        "None",            # mod_creation_days_since_joined
    ]
})
print(skew_table.to_string(index=False))

print("\nDone! Use these values to update the paper and appendix.")
