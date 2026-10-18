from modules.ingestion import load_dataset
from modules.deduplication import find_near_duplicates

dataset = load_dataset("data/raw")
feed = dataset.feed.set_index("record_id", drop=False)

pairs = find_near_duplicates(dataset.feed, threshold=0.65)

for pair in pairs:
    if pair["candidate_type"] != "repeated_document_or_reissue":
        continue

    first = feed.loc[pair["record_id_1"]]
    second = feed.loc[pair["record_id_2"]]

    print("\n" + "=" * 90)
    print(
        f'{pair["record_id_1"]} <-> {pair["record_id_2"]}'
        f' | {pair["days_apart"]} days apart'
    )
    print("Company:", first.get("company"))
    print("Ticker:", first.get("ticker"))
    print("Headline:", first.get("headline"))
    print("Content identical:", pair["same_content"])
    print("\nFIRST DOCUMENT:\n", first.get("content"))
    print("\nSECOND DOCUMENT:\n", second.get("content"))