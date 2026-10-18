
"""
Evidence-linked event histories for equity research intelligence.

Features:
- Preserve every research record.
- Link only approved, same-company relationships.
- Validate company/ticker identity before linking.
- Build chronological research histories.
- Detect and describe document content changes.
- Preserve accepted and rejected relationship evidence.
- Flag transitive clusters that lack direct pairwise evidence.
- Produce deterministic event IDs for a fixed set of record IDs.

Important:
An event history is an evidence-backed grouping of research records.
It is not proof that every record describes the same real-world event.
"""

import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd


LINKABLE_RELATIONSHIPS = {
    "likely_duplicate",
    "repeated_document_or_reissue",
    "possible_follow_up",
}


def _has_value(value) -> bool:
    """Return whether a scalar value is present and usable."""
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False

    return bool(str(value).strip())


def _clean_value(value):
    """Represent missing scalar values as None."""
    return value if _has_value(value) else None


def _normalise_content(content) -> str:
    """Normalize case and whitespace for document comparison."""
    if not isinstance(content, str):
        return ""

    return re.sub(r"\s+", " ", content.casefold()).strip()


def _normalise_identity(value) -> str:
    """Normalize an identity field without modifying the source."""
    if not _has_value(value):
        return ""

    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _company_identity_key(record: dict):
    """Return the strongest available company identity."""
    ticker = _normalise_identity(record.get("ticker"))

    if ticker:
        return ("ticker", ticker)

    company = _normalise_identity(record.get("company"))

    if company:
        return ("company", company)

    return None


def _identity_conflicts(first: dict, second: dict) -> bool:
    """
    Return True only when available company identity fields conflict.

    Missing tickers do not cause a conflict when company names match.
    Matching tickers take precedence over company-name variations.
    """
    ticker_1 = _normalise_identity(first.get("ticker"))
    ticker_2 = _normalise_identity(second.get("ticker"))

    if ticker_1 and ticker_2 and ticker_1 != ticker_2:
        return True

    company_1 = _normalise_identity(first.get("company"))
    company_2 = _normalise_identity(second.get("company"))

    if company_1 and company_2 and company_1 != company_2:
        # The same ticker can legitimately have name variations.
        if ticker_1 and ticker_2 and ticker_1 == ticker_2:
            return False

        return True

    return False


def _has_comparable_identity(first: dict, second: dict) -> bool:
    """Check whether two records share a usable identity field."""
    ticker_1 = _normalise_identity(first.get("ticker"))
    ticker_2 = _normalise_identity(second.get("ticker"))

    if ticker_1 and ticker_2:
        return True

    company_1 = _normalise_identity(first.get("company"))
    company_2 = _normalise_identity(second.get("company"))

    return bool(company_1 and company_2)


def _parse_date(value):
    """Parse a date into a timezone-aware UTC timestamp."""
    if not _has_value(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return None

    return parsed


def _record_sort_key(record: dict):
    """Sort records chronologically, with stable ID tie-breaking."""
    parsed = _parse_date(record.get("published_at"))

    return (
        parsed is None,
        parsed
        if parsed is not None
        else pd.Timestamp.max.tz_localize("UTC"),
        str(record.get("record_id", "")),
    )


def _content_diff(previous, current) -> dict:
    """
    Describe token-level changes between two available documents.

    Similarity is lexical, not semantic. Added and removed terms are
    evidence for analyst inspection, not claims about materiality.
    """
    previous_text = _normalise_content(previous)
    current_text = _normalise_content(current)

    if not previous_text or not current_text:
        return {
            "content_changed": None,
            "content_similarity": None,
            "added_terms": [],
            "removed_terms": [],
        }

    previous_tokens = previous_text.split()
    current_tokens = current_text.split()

    similarity = SequenceMatcher(
        None,
        previous_tokens,
        current_tokens,
        autojunk=False,
    ).ratio()

    matcher = SequenceMatcher(
        None,
        previous_tokens,
        current_tokens,
        autojunk=False,
    )

    added = []
    removed = []

    for operation, first_start, first_end, second_start, second_end in (
        matcher.get_opcodes()
    ):
        if operation in {"delete", "replace"}:
            removed.extend(previous_tokens[first_start:first_end])

        if operation in {"insert", "replace"}:
            added.extend(current_tokens[second_start:second_end])

    return {
        "content_changed": previous_text != current_text,
        "content_similarity": round(similarity, 4),
        "added_terms": added[:30],
        "removed_terms": removed[:30],
    }


def _choose_first_value(records: list[dict], field: str):
    """Return the first non-empty value for a field."""
    for record in records:
        value = _clean_value(record.get(field))

        if value is not None:
            return value

    return None


def _build_record_history(records: list[dict]) -> dict:
    """Build chronology and compare documents with the previous usable one."""
    ordered = sorted(records, key=_record_sort_key)
    history = []
    previous_record = None

    for record in ordered:
        if previous_record is None:
            diff = {
                "content_changed": None,
                "content_similarity": None,
                "added_terms": [],
                "removed_terms": [],
            }
        else:
            diff = _content_diff(
                previous_record.get("content"),
                record.get("content"),
            )

        history.append({
            "record_id": record.get("record_id"),
            "headline": _clean_value(record.get("headline")),
            "published_at": _clean_value(record.get("published_at")),
            "source_type": _clean_value(record.get("source_type")),
            "content": _clean_value(record.get("content")),
            "content_changed_from_previous": diff["content_changed"],
            "content_similarity_to_previous": (
                diff["content_similarity"]
            ),
            "added_terms": diff["added_terms"],
            "removed_terms": diff["removed_terms"],
        })

        # A missing-content record should not erase the last usable
        # comparison point.
        if _normalise_content(record.get("content")):
            previous_record = record

    normalized_contents = [
        _normalise_content(record.get("content"))
        for record in ordered
    ]
    available_contents = [
        content for content in normalized_contents if content
    ]

    if len(available_contents) != len(ordered):
        all_content_identical = None
    else:
        all_content_identical = len(set(available_contents)) <= 1

    dated_records = [
        _parse_date(record.get("published_at"))
        for record in ordered
    ]
    dated_records = [date for date in dated_records if date is not None]

    return {
        "records": history,
        "event_start": (
            min(dated_records).isoformat() if dated_records else None
        ),
        "latest_update": (
            max(dated_records).isoformat() if dated_records else None
        ),
        "has_content_changes": any(
            item["content_changed_from_previous"] is True
            for item in history
        ),
        "all_content_identical": all_content_identical,
    }


def _relationship_evidence(pair: dict) -> dict:
    """Preserve observed evidence behind an accepted relationship."""
    return {
        "record_id_1": pair.get("record_id_1"),
        "record_id_2": pair.get("record_id_2"),
        "relationship": pair.get("candidate_type"),
        "similarity": pair.get("similarity"),
        "same_company": pair.get("same_company"),
        "days_apart": pair.get("days_apart"),
        "same_content": pair.get("same_content"),
        "numeric_difference": pair.get("numeric_difference"),
    }


def _excluded_pair_reason(pair: dict) -> str:
    """Explain why a candidate relationship was not merged."""
    relationship = pair.get("candidate_type")

    if pair.get("same_company") is False:
        return (
            "Different company identities; do not merge company events."
        )

    if relationship == "needs_review":
        return (
            "Numerical expressions differ; manual review is required."
        )

    if relationship == "related_company_research":
        return (
            "Related research is not sufficient proof of one event."
        )

    if relationship == "recurring_topic_or_template":
        return (
            "May represent recurring sector commentary or a template."
        )

    if relationship == "uncertain":
        return "Insufficient evidence for an event-level link."

    if relationship not in LINKABLE_RELATIONSHIPS:
        return "Relationship type is not approved for automatic linking."

    if pair.get("same_company") is not True:
        return (
            "Company identity is unknown or not sufficiently established."
        )

    return "Relationship was not accepted for automatic linking."


def _candidate_review_evidence(pair: dict) -> dict:
    """Describe a candidate relationship that was not used for merging."""
    return {
        **_relationship_evidence(pair),
        "reason_not_linked": _excluded_pair_reason(pair),
    }



def build_event_timeline(
    feed: pd.DataFrame,
    candidate_pairs: list[dict],
) -> list[dict]:
    """
    Build chronological research histories from candidate record relationships.

    Only sufficiently supported, same-company relationships are linked.
    High headline similarity alone is not enough to merge records when
    their content differs or is unavailable.

    Every input record appears in exactly one returned event.
    Candidate pairs referencing unknown records or self-pairs are ignored.

    Important:
        - Linked events are heuristic research histories, not proof that
          records describe the same real-world event.
        - Indirect graph connections are explicitly flagged for review.
        - Rejected candidate relationships remain available for inspection.
    """
    required_columns = {"record_id", "headline"}
    missing = required_columns - set(feed.columns)

    if missing:
        raise ValueError(
            f"Feed is missing required columns: {sorted(missing)}"
        )

    if feed["record_id"].isna().any():
        raise ValueError("Feed contains missing record_id values.")

    if feed["record_id"].duplicated().any():
        raise ValueError("Feed contains duplicate record_id values.")

    records = feed.to_dict("records")
    record_by_id = {
        record["record_id"]: record
        for record in records
    }

    # Empty input is valid and produces no events.
    if not records:
        return []

    # Disjoint-set / union-find structure.
    parent = {
        record_id: record_id
        for record_id in record_by_id
    }
    rank = {record_id: 0 for record_id in parent}

    def find(record_id):
        """Find a component representative with path compression."""
        while parent[record_id] != record_id:
            parent[record_id] = parent[parent[record_id]]
            record_id = parent[record_id]
        return record_id

    def union(first_id, second_id):
        """Merge components using union by rank."""
        root_1 = find(first_id)
        root_2 = find(second_id)

        if root_1 == root_2:
            return

        if rank[root_1] < rank[root_2]:
            root_1, root_2 = root_2, root_1

        parent[root_2] = root_1

        if rank[root_1] == rank[root_2]:
            rank[root_1] += 1

    accepted_pairs = []
    excluded_pairs = []

    # Validate each proposed relationship before merging records.
    for pair in candidate_pairs:
        first_id = pair.get("record_id_1")
        second_id = pair.get("record_id_2")

        # Ignore malformed references rather than creating phantom events.
        if first_id not in record_by_id or second_id not in record_by_id:
            continue

        if first_id == second_id:
            continue

        relationship = pair.get("candidate_type")
        same_content = pair.get("same_content")

        link_is_approved = (
            relationship in LINKABLE_RELATIONSHIPS
            and pair.get("same_company") is True
        )

        if not link_is_approved:
            excluded_pairs.append(
                _candidate_review_evidence(pair)
            )
            continue

        first_record = record_by_id[first_id]
        second_record = record_by_id[second_id]

        # Require identity evidence from the underlying records.
        if not _has_comparable_identity(
            first_record,
            second_record,
        ):
            excluded_pairs.append({
                **_relationship_evidence(pair),
                "reason_not_linked": (
                    "Insufficient comparable company/ticker identity "
                    "in the underlying records."
                ),
            })
            continue

        # Do not trust a candidate label if the underlying identities clash.
        if _identity_conflicts(first_record, second_record):
            excluded_pairs.append({
                **_relationship_evidence(pair),
                "reason_not_linked": (
                    "Underlying company or ticker fields conflict."
                ),
            })
            continue

        # A likely-duplicate label must be supported by identical content.
        # False, missing, or unavailable content evidence requires review.
        if (
            relationship == "likely_duplicate"
            and same_content is not True
        ):
            excluded_pairs.append({
                **_relationship_evidence(pair),
                "reason_not_linked": (
                    "Headline similarity is high, but document content "
                    "is different or unavailable; review before linking."
                ),
            })
            continue

        union(first_id, second_id)
        accepted_pairs.append(pair)

    # Construct connected components after all accepted unions.
    grouped_records = defaultdict(list)

    for record_id, record in record_by_id.items():
        grouped_records[find(record_id)].append(record)

    grouped_pairs = defaultdict(list)

    for pair in accepted_pairs:
        grouped_pairs[find(pair["record_id_1"])].append(pair)

    # Keep excluded relationships attached to both affected records.
    grouped_exclusions = defaultdict(list)

    for pair in excluded_pairs:
        first_id = pair.get("record_id_1")
        second_id = pair.get("record_id_2")

        if first_id in record_by_id:
            grouped_exclusions[first_id].append(pair)

        if second_id in record_by_id:
            grouped_exclusions[second_id].append(pair)

    events = []

    for root, group_records in grouped_records.items():
        group_records = sorted(
            group_records,
            key=_record_sort_key,
        )

        pair_evidence = grouped_pairs[root]
        record_ids = [
            record["record_id"]
            for record in group_records
        ]
        member_ids = set(record_ids)

        # A component is potentially unsafe when members are connected
        # only by a chain and not every member pair has direct evidence.
        direct_edges = {
            frozenset((
                pair["record_id_1"],
                pair["record_id_2"],
            ))
            for pair in pair_evidence
        }

        missing_direct_links = [
            [first_id, second_id]
            for first_id, second_id in combinations(record_ids, 2)
            if frozenset((first_id, second_id)) not in direct_edges
        ]

        transitive_link_risk = bool(missing_direct_links)

        relationship_types = sorted({
            pair.get("candidate_type")
            for pair in pair_evidence
            if pair.get("candidate_type")
        })

        if not pair_evidence:
            relationship = "standalone"
        elif len(relationship_types) == 1:
            relationship = relationship_types[0]
        else:
            relationship = "mixed_relationships"

        history = _build_record_history(group_records)

        # Stable for a fixed set of record IDs; not a permanent database key.
        anchor_id = min(str(record_id) for record_id in record_ids)
        event_id = f"EVT-{anchor_id}"

        identity_conflict = any(
            _identity_conflicts(first, second)
            for first, second in combinations(group_records, 2)
        )

        if len(group_records) == 1:
            resolution_status = "standalone"
            confidence = "not_applicable"
        elif identity_conflict:
            resolution_status = "review_identity_conflict"
            confidence = "low"
        elif transitive_link_risk:
            resolution_status = "review_transitive_link"
            confidence = "medium"
        elif history["all_content_identical"] is True:
            resolution_status = "linked_identical_content"
            confidence = "high"
        else:
            resolution_status = "linked_research_history"
            confidence = "medium"

        # Preserve rejected candidate relationships touching this event.
        event_exclusions = [
            pair
            for record_id in member_ids
            for pair in grouped_exclusions.get(record_id, [])
        ]

        unique_exclusions = []
        seen_exclusions = set()

        for pair in event_exclusions:
            first_id = pair.get("record_id_1")
            second_id = pair.get("record_id_2")

            key = (
                min(str(first_id), str(second_id)),
                max(str(first_id), str(second_id)),
                pair.get("relationship"),
            )

            if key not in seen_exclusions:
                seen_exclusions.add(key)
                unique_exclusions.append(pair)

        events.append({
            "event_id": event_id,
            "company": _choose_first_value(
                group_records,
                "company",
            ),
            "ticker": _choose_first_value(
                group_records,
                "ticker",
            ),
            "relationship": relationship,
            "relationship_types": relationship_types,
            "event_type": (
                "standalone"
                if len(group_records) == 1
                else "linked_research_history"
            ),
            "resolution_status": resolution_status,
            "confidence": confidence,
            "transitive_link_risk": transitive_link_risk,
            "missing_direct_links": missing_direct_links,
            "identity_conflict": identity_conflict,
            "record_count": len(group_records),
            "record_ids": record_ids,
            "event_start": history["event_start"],
            "latest_update": history["latest_update"],
            "has_content_changes": history["has_content_changes"],
            "all_content_identical": history["all_content_identical"],
            "records": history["records"],
            "relationship_evidence": [
                _relationship_evidence(pair)
                for pair in pair_evidence
            ],
            "excluded_relationships": unique_exclusions,
        })

    # Chronological output; events without usable dates appear last.
    events.sort(
        key=lambda event: (
            event["event_start"] is None,
            event["event_start"] or "",
            event["event_id"],
        )
    )

    return events
