"""Simple, explainable prioritization for recent research records."""
from __future__ import annotations

import pandas as pd


def rank_recent_insights(feed: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Return recent records ranked by a transparent triage score.

    Recency is measured relative to the newest usable date in the supplied
    dataset, so results are reproducible even when the dataset is historical.
    The score is for review prioritization, not investment quality prediction.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an integer of at least 1.")

    required = {"published_at", "headline", "content", "company", "ticker", "source_type"}
    missing = required - set(feed.columns)
    if missing:
        raise ValueError(f"Feed is missing required columns: {sorted(missing)}")

    results = feed.copy()
    dates = pd.to_datetime(results["published_at"], errors="coerce", utc=True)
    latest = dates.max()

    if pd.isna(latest):
        results["priority_score"] = 0.0
        results["priority_reasons"] = "Publication date unavailable; metadata completeness only"
        return results.head(limit)

    age_days = ((latest - dates).dt.total_seconds() / 86400).clip(lower=0)
    recency_score = ((30 - age_days.clip(upper=30)) / 30 * 70).fillna(0)

    source = results["source_type"].fillna("").astype(str).str.casefold()
    source_score = source.map(
        lambda value: 15 if any(term in value for term in
            ("filing", "exchange", "company disclosure", "regulatory"))
        else 8 if any(term in value for term in ("news", "broker", "research"))
        else 0
    )

    fields = ["headline", "content", "company", "ticker"]
    completeness = results[fields].apply(
        lambda column: column.notna() & column.astype(str).str.strip().ne("")
    ).sum(axis=1) / len(fields) * 15

    results["priority_score"] = (recency_score + source_score + completeness).round(1)
    results["_priority_date"] = dates

    def explain(row):
        reasons = []
        date = row["_priority_date"]
        if pd.notna(date):
            age = max(0, (latest - date).total_seconds() / 86400)
            if age <= 7:
                reasons.append("published within 7 days of dataset's newest record")
            elif age <= 30:
                reasons.append("published within 30 days of dataset's newest record")
        source_type = str(row.get("source_type", "") or "").casefold()
        if any(term in source_type for term in
               ("filing", "exchange", "company disclosure", "regulatory")):
            reasons.append("disclosure/regulatory source")
        elif any(term in source_type for term in ("news", "broker", "research")):
            reasons.append("research/news source")
        present = sum(
            pd.notna(row.get(field)) and bool(str(row.get(field)).strip())
            for field in fields
        )
        reasons.append(f"{present}/4 key fields populated")
        return "; ".join(reasons)

    results["priority_reasons"] = results.apply(explain, axis=1)
    results = results.sort_values(
        ["priority_score", "_priority_date"], ascending=[False, False],
        na_position="last", kind="stable",
    )
    return results.head(limit).drop(columns=["_priority_date"])
