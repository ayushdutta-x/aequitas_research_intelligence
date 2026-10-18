from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.ingestion import ResearchDataset

@dataclass
class ResearchIndex:
    records: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object

#index-building function
def build_research_index(dataset: ResearchDataset) -> ResearchIndex:
    records = dataset.feed.copy()

    records["search_text"] = (
        records["headline"].fillna("")
        + " "
        + records["content"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
    )

    # A malformed/empty corpus should still start cleanly. Fit on a
    # sentinel only when the real corpus has no usable terms.
    try:
        matrix = vectorizer.fit_transform(records["search_text"].fillna(""))
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):
            raise
        records = records.copy()
        records["search_text"] = "research_record"
        matrix = vectorizer.fit_transform(records["search_text"])

    return ResearchIndex(
        records=records,
        vectorizer=vectorizer,
        matrix=matrix,
    )


def search_research(
    query: str,
    index: ResearchIndex,
    limit: int = 10,
    company: str | None = None,
    ticker: str | None = None,
    sector: str | None = None,
    source_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    sort_by: str = "relevance",
) -> pd.DataFrame:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Limit must be an integer of at least 1.")

    query_vector = index.vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        index.matrix,
    ).flatten()

    results = index.records.copy()
    results["relevance_score"] = scores

    results = results[results["relevance_score"] > 0]

    if company:
        results = results[results["company"].fillna("").astype(str).str.contains(
            company, case=False, na=False, regex=False
        )]

    if ticker:
        results = results[results["ticker"].fillna("").astype(str).str.casefold() == ticker.casefold()]

    if sector:
        results = results[results["sector"].fillna("").astype(str).str.contains(
            sector, case=False, na=False, regex=False
        )]

    if source_type:
        results = results[results["source_type"].fillna("").astype(str).str.casefold() == source_type.casefold()]

    if start_date:
        start = pd.to_datetime(start_date, errors="raise", utc=True)
        dates = pd.to_datetime(results["published_at"], errors="coerce", utc=True)
        results = results[dates >= start]

    if end_date:
        end = pd.to_datetime(end_date, errors="raise", utc=True)
        # An end date supplied without a time includes the whole day.
        if isinstance(end_date, str) and len(end_date.strip()) == 10:
            end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        dates = pd.to_datetime(results["published_at"], errors="coerce", utc=True)
        results = results[dates <= end]


    if sort_by == "relevance":
        results = results.sort_values(
            by="relevance_score",
            ascending=False,
            kind="stable",
        )

    elif sort_by == "date":
        results = results.sort_values(
            by="published_at",
            ascending=False,
            kind="stable",
            na_position="last",
        )

    else:
        raise ValueError(
            "sort_by must be 'relevance' or 'date'."
        )

    return results.head(limit).drop(columns=["search_text"])
        