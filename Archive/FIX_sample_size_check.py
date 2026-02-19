"""Quick diagnostic: trace where the sample size dropped from 615,392 to 73,855."""
import pandas as pd

DATA_DIR = r"c:\Users\nicta\Desktop\API_Paper_ChiPlay\mod_data"
authors = pd.read_parquet(f"{DATA_DIR}/Authors.parquet")
mods = pd.read_parquet(f"{DATA_DIR}/CleanedModData.parquet")

print("=== AUTHORS TABLE ===")
print(f"Total rows: {len(authors):,}")
print(f"deleted=True: {authors['deleted'].sum():,}")
print(f"deleted=False: {(~authors['deleted']).sum():,}")
print(f"last_active is null: {authors['last_active'].isna().sum():,}")
has_active = authors['last_active'].notna()
print(f"last_active >= 2024-01-01: {(has_active & (authors['last_active'] >= '2024-01-01')).sum():,}")

# Apply the SQL WHERE clause step by step
step1 = authors[~authors['deleted']]
print(f"\nStep 1 - deleted=0: {len(step1):,}")

step2 = step1[step1['last_active'].notna()]
print(f"Step 2 - last_active NOT NULL: {len(step2):,}")

step3 = step2[step2['last_active'] >= '2024-01-01']
print(f"Step 3 - last_active >= 2024-01-01: {len(step3):,}")

# Now check: how many of these have mods in CleanedModData?
mod_member_ids = set(mods['member_id'].unique())
has_mods = step3['member_id'].isin(mod_member_ids)
print(f"\nOf {len(step3):,} filtered authors:")
print(f"  With mods in CleanedModData: {has_mods.sum():,}")
print(f"  Without mods in CleanedModData: {(~has_mods).sum():,}")

# The SQL query uses LEFT JOIN ... GROUP BY
# But if users have no mods, they'd still appear (with NULLs for mod fields)
# Unless there's a downstream filter

print(f"\n=== MODS TABLE ===")
print(f"Total mods: {len(mods):,}")
print(f"Unique member_ids in mods: {mods['member_id'].nunique():,}")
print(f"Unique member_ids in Authors: {authors['member_id'].nunique():,}")

# mod_count from Authors table (API-provided, separate from CleanedModData)
print(f"\n=== mod_count in Authors (filtered, step3) ===")
print(f"mod_count == 0: {(step3['mod_count'] == 0).sum():,}")
print(f"mod_count > 0: {(step3['mod_count'] > 0).sum():,}")
print(f"owned_mod_count == 0: {(step3['owned_mod_count'] == 0).sum():,}")
print(f"owned_mod_count > 0: {(step3['owned_mod_count'] > 0).sum():,}")

# Check what the parquet Authors file actually is
# Is this ALL users from the API or already filtered?
print(f"\n=== Is Authors.parquet already filtered? ===")
print(f"Total Authors rows: {len(authors):,}")
print(f"Any deleted=True? {authors['deleted'].any()}")
print(f"Min last_active: {authors['last_active'].min()}")
print(f"Max last_active: {authors['last_active'].max()}")
print(f"Min joined: {authors['joined'].min()}")
print(f"Max joined: {authors['joined'].max()}")

# Look at the downstream filters in notebook 03
# After the SQL query, what additional filtering happens?
print(f"\n=== Checking for additional filters ===")
# Users with mod_count > 0 in Authors BUT no mods in CleanedModData
step3_with_mods = step3[step3['mod_count'] > 0]
in_cleaned = step3_with_mods['member_id'].isin(mod_member_ids)
print(f"Authors with mod_count>0: {len(step3_with_mods):,}")
print(f"  Of those, in CleanedModData: {in_cleaned.sum():,}")
print(f"  Of those, NOT in CleanedModData: {(~in_cleaned).sum():,}")

# Check: what if the 615,392 was on the SQL server (with the full Authors table)?
# The parquet Authors has only 122,859 rows
# The SQL server Authors had 1,028,417 rows
# After filtering: deleted=0, last_active >= 2024-01-01 -> some number
# On the parquet (122,859), after filtering -> 73,855
# So: 122,859 is likely already a subset
print(f"\n=== CONCLUSION ===")
print(f"The parquet Authors.parquet has {len(authors):,} rows")
print(f"The paper says N_raw = 1,028,417")
print(f"So Authors.parquet is likely ALREADY a filtered export,")
print(f"not the full table from the SQL server.")
print(f"")
print(f"After SQL filters (deleted=0, last_active >= 2024-01-01): {len(step3):,}")
print(f"After LEFT JOIN with CleanedModData (users WITH mods): {has_mods.sum():,}")
print(f"")
print(f"The 615,392 likely came from the full SQL server Authors table")
print(f"with the same WHERE clause applied to all 1,028,417 users.")
print(f"The 73,855 represents users who ALSO have mods in CleanedModData.")
