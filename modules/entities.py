from dataclasses import dataclass
from typing import Literal
from difflib import SequenceMatcher

import pandas as pd

@dataclass
class EntityMatch:
    query: str
    canonical_company: str | None
    ticker: str | None
    match_type: Literal[
        "exact_name",
        "alias",
        "ticker",
        "fuzzy",
        "unresolved",
    ]
    confidence: float

def normalise_name(name: str) -> str:
    if name is None or pd.isna(name):
        return ""
    return " ".join(str(name).casefold().split())

def build_company_lookup(companies: pd.DataFrame) -> dict:
    lookup = {}

    for _, row in companies.iterrows():
        canonical_name = row["company"]
        ticker = row["ticker"]

        canonical_key = normalise_name(canonical_name)
        if not canonical_key:
            continue
        lookup[canonical_key] = (canonical_name, ticker, "exact_name")

        if pd.notna(ticker) and normalise_name(ticker):
            lookup[normalise_name(ticker)] = (
                canonical_name,
                ticker,
                "ticker",
            )

        aliases = row["aliases"]

        if pd.notna(aliases):
            for alias in aliases.split(";"):
                alias = alias.strip()

                if alias:
                    lookup[normalise_name(alias)] = (
                        canonical_name,
                        ticker,
                        "alias",
                    )

    return lookup

def resolve_company(query: str, lookup: dict) -> EntityMatch:
    normalised_query = normalise_name(query)

    if not normalised_query:
        return EntityMatch(
            query=query,
            canonical_company=None,
            ticker=None,
            match_type="unresolved",
            confidence=0.0,
        )

    match = lookup.get(normalised_query)

    if match is None:
        return EntityMatch(
            query=query,
            canonical_company=None,
            ticker=None,
            match_type="unresolved",
            confidence=0.0,
        )

    canonical_name, ticker, match_type = match

    return EntityMatch(
        query=query,
        canonical_company=canonical_name,
        ticker=ticker,
        match_type=match_type,
        confidence=1.0,
    )

def suggest_company_matches(
    query: str,
    lookup: dict,
    limit: int = 3,
    min_score: float = 0.50,
) -> list[EntityMatch]:

    normalised_query = normalise_name(query)

    if not normalised_query:
        return []

    # A company may have several aliases. Keep only its best score.
    best_matches = {}

    for name, (canonical_name, ticker, _) in lookup.items():
        score = SequenceMatcher(
            None,
            normalised_query,
            name,
        ).ratio()

        if score < min_score:
            continue

        # Do not suggest an exact match as a fuzzy candidate.
        if normalised_query == name:
            continue

        existing = best_matches.get(canonical_name)

        if existing is None or score > existing.confidence:
            best_matches[canonical_name] = EntityMatch(
                query=query,
                canonical_company=canonical_name,
                ticker=ticker,
                match_type="fuzzy",
                confidence=score,
            )

    candidates = sorted(
        best_matches.values(),
        key=lambda match: match.confidence,
        reverse=True,
    )

    return candidates[:limit]