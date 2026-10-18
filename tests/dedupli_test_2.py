from modules.ingestion import load_dataset

dataset = load_dataset("data/raw")
feed = dataset.feed

record_ids = [
    "RF0213", "RF0183", "RF0203", "RF0193",
    "RF0249", "RF0247",
]

available_columns = [
    column
    for column in [
        "record_id",
        "headline",
        "company",
        "ticker",
        "published_at",
        "source",
        "source_type",
        "content",
    ]
    if column in feed.columns
]

print(feed.loc[
    feed["record_id"].isin(record_ids),
    available_columns,
].sort_values("record_id").to_string(index=False))