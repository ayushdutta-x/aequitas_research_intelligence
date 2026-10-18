from collections import Counter

from modules.ingestion import load_dataset
from modules.deduplication import (
    find_exact_duplicates,
    find_near_duplicates,
)

dataset = load_dataset("data/raw")

exact = find_exact_duplicates(dataset.feed)
pairs = find_near_duplicates(dataset.feed, threshold=0.65)

print("Records:", len(dataset.feed))
print("Exact duplicate groups:", len(exact))
print("Near-duplicate pairs:", len(pairs))

print("\nCandidate classifications:")
print(Counter(pair["candidate_type"] for pair in pairs))

for pair in pairs[:10]:
    print(
        f"\n{pair['record_id_1']} <-> {pair['record_id_2']}"
        f"\nSimilarity: {pair['similarity']:.3f}"
        f"\nClassification: {pair['candidate_type']}"
        f"\nSame company: {pair['same_company']}"
        f"\nDays apart: {pair['days_apart']}"
        f"\nNumeric difference: {pair['numeric_difference']}"
    )