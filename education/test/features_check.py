import pandas as pd

df = pd.read_parquet(
    "download_dataset/new_dataset/new_data/final_openalex_dataset.parquet"
)

check_cols = [
    "citation_velocity",
    "novelty_raw",
    "topic_publication_growth",
    "source_tier",
]

for col in check_cols:
    print(f"\n=== FEATURE: {col} ===")
    stats = df.groupby("pub_year")[col].agg(
        ["count", "mean", "std", lambda x: (x == 0).mean()]
    )
    stats.columns = ["count", "mean", "std", "share_zeros"]
    print(stats.loc[2020:2023])