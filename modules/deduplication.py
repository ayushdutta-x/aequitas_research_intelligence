
import math
import re
from collections import defaultdict
from itertools import combinations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalise_headline(headline: str) -> str:
    """Normalize text for comparison without modifying the source."""
    if not isinstance(headline, str):
        return ""

    headline = headline.casefold()
    headline = re.sub(r"[^\w\s]", " ", headline)
    headline = re.sub(r"\s+", " ", headline)

    return headline.strip()


def _normalise_content(content) -> str:
    """Normalize document content for exact-content comparison."""
    if not isinstance(content, str):
        return ""

    return re.sub(r"\s+", " ", content.casefold()).strip()


def extract_numeric_facts(text: str) -> list[str]:
    """Extract numerical expressions and common adjacent units."""
    if not isinstance(text, str):
        return []

    number = r"\d[\d,]*(?:\.\d+)?"

    pattern = re.compile(
        rf"""
        ₹\s*{number}(?:\s*(?:crore|lakh|million|billion))?
        |
        \b{number}\s*%
        |
        \b{number}\s*
        (?:crore|lakh|million|billion|rupees|INR|USD|months|years|days)
        \b
        |
        \b{number}\b
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return [
        re.sub(r"\s+", " ", match.group()).strip().casefold()
        for match in pattern.finditer(text)
    ]


def find_exact_duplicates(feed: pd.DataFrame) -> list[dict]:
    """Group records with identical normalized headlines."""

    required_columns = {"record_id", "headline"}
    missing = required_columns - set(feed.columns)

    if missing:
        raise ValueError(
            f"Feed is missing required columns: {sorted(missing)}"
        )

    records = feed[["record_id", "headline"]].copy()
    records["normalised_headline"] = (
        records["headline"].map(normalise_headline)
    )

    records = records[
        records["normalised_headline"].ne("")
    ]

    groups = defaultdict(list)

    for _, record in records.iterrows():
        groups[record["normalised_headline"]].append(
            record["record_id"]
        )

    return [
        {
            "normalised_headline": headline,
            "record_ids": record_ids,
            "count": len(record_ids),
        }
        for headline, record_ids in groups.items()
        if len(record_ids) > 1
    ]


def _has_value(value) -> bool:
    """Check whether a scalar field contains a usable value."""
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False

    return bool(str(value).strip())


def _compare_company_identity(first, second):
    """Compare company identities, returning True, False or None."""

    ticker_1 = first.get("ticker")
    ticker_2 = second.get("ticker")

    # Tickers are the preferred identity signal when both exist.
    if _has_value(ticker_1) and _has_value(ticker_2):
        return (
            str(ticker_1).strip().casefold()
            == str(ticker_2).strip().casefold()
        )

    company_1 = first.get("company")
    company_2 = second.get("company")

    if _has_value(company_1) and _has_value(company_2):
        return (
            normalise_headline(str(company_1))
            == normalise_headline(str(company_2))
        )

    # Missing identity is unknown, not evidence of a match.
    return None


def _calculate_days_apart(date_1, date_2):
    """Calculate elapsed days, returning None for unusable dates."""

    if not _has_value(date_1) or not _has_value(date_2):
        return None

    parsed_1 = pd.to_datetime(date_1, errors="coerce")
    parsed_2 = pd.to_datetime(date_2, errors="coerce")

    if pd.isna(parsed_1) or pd.isna(parsed_2):
        return None

    try:
        return round(
            abs((parsed_1 - parsed_2).total_seconds()) / 86400,
            2,
        )
    except (TypeError, ValueError):
        # For example, incompatible timezone-aware and naive dates.
        return None


def classify_candidate_pair(pair: dict) -> str:
    """Assign a provisional label using identity, content and timing."""

    same_company = pair["same_company"]
    similarity = pair["similarity"]
    days_apart = pair["days_apart"]
    numeric_difference = pair["numeric_difference"]
    same_content = pair.get("same_content")

    # Similar wording across different companies is not duplicate coverage.
    if same_company is False:
        return "cross_company_similarity"

    # Changed extracted numbers need inspection; they are not
    # necessarily contradictions.
    if numeric_difference:
        return "needs_review"

    if same_company is True:

        # Identical company content published close together.
        if same_content is True and days_apart is not None:
            if days_apart <= 2:
                return "likely_duplicate"

            return "repeated_document_or_reissue"

        # Similar headline, same company, but content differs or is missing.
        if similarity >= 0.95 and days_apart is not None:
            if days_apart <= 2:
                return "likely_duplicate"

            return "possible_follow_up"

        if similarity >= 0.80:
            return "related_company_research"

        return "uncertain"

    # Identical content without known company identity is more likely
    # recurring sector commentary or a reusable template.
    if same_content is True and similarity >= 0.95:
        return "recurring_topic_or_template"

    # Unknown identity plus near-identical wording remains ambiguous.
    if similarity >= 0.95:
        return "recurring_topic_or_template"

    return "uncertain"


def find_near_duplicates(
    feed: pd.DataFrame,
    threshold: float = 0.65,
) -> list[dict]:
    """Find similar headline pairs and attach contextual evidence."""

    required_columns = {"record_id", "headline"}
    missing = required_columns - set(feed.columns)

    if missing:
        raise ValueError(
            f"Feed is missing required columns: {sorted(missing)}"
        )

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Threshold must be a finite number from 0 to 1.")

    optional_columns = [
        "published_at",
        "company",
        "ticker",
        "content",
    ]

    columns = [
        "record_id",
        "headline",
        *[
            column
            for column in optional_columns
            if column in feed.columns
        ],
    ]

    records = feed[columns].copy()
    records["normalised_headline"] = (
        records["headline"].map(normalise_headline)
    )

    # Remove unusable headlines, not duplicate records.
    records = records[
        records["normalised_headline"].ne("")
    ].reset_index(drop=True)

    if len(records) < 2:
        return []

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b",
    )

    try:
        matrix = vectorizer.fit_transform(
            records["normalised_headline"]
        )
    except ValueError as exc:
        if "empty vocabulary" in str(exc):
            return []
        raise

    similarity_matrix = cosine_similarity(matrix)
    results = []

    # Each unordered pair is considered exactly once.
    for i, j in combinations(range(len(records)), 2):
        first = records.iloc[i]
        second = records.iloc[j]

        score = float(similarity_matrix[i, j])

        if score < threshold:
            continue

        same_company = _compare_company_identity(first, second)

        date_1 = first.get("published_at")
        date_2 = second.get("published_at")
        days_apart = _calculate_days_apart(date_1, date_2)

        content_1 = _normalise_content(first.get("content"))
        content_2 = _normalise_content(second.get("content"))

        # None means content is unavailable for at least one record.
        same_content = (
            content_1 == content_2
            if content_1 and content_2
            else None
        )

        facts_1 = extract_numeric_facts(first["headline"])
        facts_2 = extract_numeric_facts(second["headline"])

        numeric_difference = facts_1 != facts_2

        pair = {
            "record_id_1": first["record_id"],
            "record_id_2": second["record_id"],
            "headline_1": first["headline"],
            "headline_2": second["headline"],
            "similarity": round(score, 4),
            "company_1": first.get("company"),
            "company_2": second.get("company"),
            "same_company": same_company,
            "date_1": date_1,
            "date_2": date_2,
            "days_apart": days_apart,
            "numeric_facts_1": facts_1,
            "numeric_facts_2": facts_2,
            "numeric_difference": numeric_difference,
            # Backwards-compatible alias; not proof of contradiction.
            "numeric_conflict": numeric_difference,
            "same_content": same_content,
            "content_available": bool(content_1 and content_2),
        }

        pair["candidate_type"] = classify_candidate_pair(pair)
        results.append(pair)

    return sorted(
        results,
        key=lambda item: item["similarity"],
        reverse=True,
    )
