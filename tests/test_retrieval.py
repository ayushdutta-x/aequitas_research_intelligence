from modules.ingestion import load_dataset
from modules.retrieval import build_research_index, search_research

dataset = load_dataset("data/raw")
index = build_research_index(dataset)


# test-1 : index records shape
print("Records:", len(index.records))
print("Matrix shape:", index.matrix.shape)
print("Vocabulary size:", len(index.vectorizer.vocabulary_))

#test2- search for keywords
test1 = search_research("order execution accelerates", index,limit=5)
test2 = search_research("quantumleaf AI transformation", index, limit=5)
test3 = search_research("bank regulatory review", index, limit=5)

#edge cases
#edge1 = search_research("", index)
edge2 = search_research("xyzqwerty", index)
#edge3 = search_research("order", index, limit=0)


print(
    test1[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)
print(
    test2[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)
print(
    test3[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)

#check edge case
# print(
#     edge1[
#         ["record_id", "published_at", "company",
#          "headline", "relevance_score"]
#     ].to_string(index=False)
# )
print(
    edge2[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)

# print(
#     edge3[
#         ["record_id", "published_at", "company",
#          "headline", "relevance_score"]
#     ].to_string(index=False)
# )



#new search test for updated filter mechanics
t0 = search_research(
    "order execution",
    index,
    company="Prakash Motion",
    limit=5,
)

t1 = search_research(
    "regulatory review",
    index,
    sector="Banking",
    limit=5,
)

t2 = search_research(
    "AI transformation",
    index,
    start_date="2026-06-01",
    end_date="2026-09-30",
    limit=5,
    sort_by = "date",
)

print(
    t0[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)
print(
    t1[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)

print(
    t2[
        ["record_id", "published_at", "company",
         "headline", "relevance_score"]
    ].to_string(index=False)
)


