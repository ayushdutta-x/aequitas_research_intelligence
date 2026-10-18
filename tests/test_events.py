
import pandas as pd
import pytest

from modules.events import build_event_timeline


def make_feed(records):
    """Build a minimal research feed for event tests."""
    defaults = {
        "headline": "Company reports quarterly results",
        "company": "Acme Industries",
        "ticker": "ACME",
        "published_at": "2026-01-01",
        "content": "Revenue increased and operating margins improved.",
    }

    rows = []
    for record in records:
        row = {**defaults, **record}
        rows.append(row)

    return pd.DataFrame(rows)


def make_pair(
    first_id,
    second_id,
    *,
    candidate_type="likely_duplicate",
    same_company=True,
    same_content=True,
    similarity=1.0,
    days_apart=0,
    numeric_difference=False,
):
    """Build a candidate relationship in the expected input format."""
    return {
        "record_id_1": first_id,
        "record_id_2": second_id,
        "candidate_type": candidate_type,
        "same_company": same_company,
        "same_content": same_content,
        "similarity": similarity,
        "days_apart": days_apart,
        "numeric_difference": numeric_difference,
    }


def get_event_containing(events, record_id):
    """Return the unique event containing a record."""
    matches = [
        event
        for event in events
        if record_id in event["record_ids"]
    ]

    assert len(matches) == 1
    return matches[0]


def test_every_input_record_appears_exactly_once():
    feed = make_feed([
        {"record_id": "R1"},
        {"record_id": "R2"},
        {"record_id": "R3"},
    ])

    pairs = [
        make_pair("R1", "R2"),
    ]

    events = build_event_timeline(feed, pairs)

    represented_ids = [
        record_id
        for event in events
        for record_id in event["record_ids"]
    ]

    assert sorted(represented_ids) == ["R1", "R2", "R3"]
    assert len(represented_ids) == len(set(represented_ids))
    assert sum(event["record_count"] for event in events) == len(feed)


def test_identical_same_company_records_are_linked():
    feed = make_feed([
        {"record_id": "R1"},
        {"record_id": "R2", "published_at": "2026-01-02"},
    ])

    events = build_event_timeline(
        feed,
        [make_pair("R1", "R2")],
    )

    assert len(events) == 1

    event = events[0]
    assert set(event["record_ids"]) == {"R1", "R2"}
    assert event["record_count"] == 2
    assert event["relationship"] == "likely_duplicate"
    assert event["resolution_status"] == "linked_identical_content"
    assert event["all_content_identical"] is True
    assert event["transitive_link_risk"] is False


def test_different_content_likely_duplicate_is_not_linked():
    feed = make_feed([
        {"record_id": "R1", "content": "Revenue increased by 10%."},
        {"record_id": "R2", "content": "Revenue declined by 10%."},
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="likely_duplicate",
                same_content=False,
            )
        ],
    )

    assert len(events) == 2

    event_1 = get_event_containing(events, "R1")
    event_2 = get_event_containing(events, "R2")

    assert event_1["record_count"] == 1
    assert event_2["record_count"] == 1
    assert event_1["resolution_status"] == "standalone"
    assert event_2["resolution_status"] == "standalone"

    # The rejected candidate remains available for review.
    assert event_1["excluded_relationships"]
    assert event_2["excluded_relationships"]

    rejection = event_1["excluded_relationships"][0]
    assert "reason_not_linked" in rejection


def test_missing_content_does_not_allow_likely_duplicate_link():
    feed = make_feed([
        {"record_id": "R1", "content": ""},
        {"record_id": "R2", "content": ""},
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                same_content=None,
            )
        ],
    )

    assert len(events) == 2
    assert get_event_containing(events, "R1")["record_count"] == 1
    assert get_event_containing(events, "R2")["record_count"] == 1


def test_repeated_document_or_reissue_can_be_linked():
    feed = make_feed([
        {"record_id": "R1", "published_at": "2026-01-01"},
        {"record_id": "R2", "published_at": "2026-02-01"},
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="repeated_document_or_reissue",
                days_apart=31,
                same_content=True,
            )
        ],
    )

    assert len(events) == 1

    event = events[0]
    assert event["record_count"] == 2
    assert event["relationship"] == "repeated_document_or_reissue"
    assert event["all_content_identical"] is True


def test_possible_follow_up_can_have_different_content():
    feed = make_feed([
        {
            "record_id": "R1",
            "published_at": "2026-01-01",
            "content": "The company announced a new project.",
        },
        {
            "record_id": "R2",
            "published_at": "2026-01-03",
            "content": "The company reported progress on the new project.",
        },
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="possible_follow_up",
                same_content=False,
                similarity=0.96,
                days_apart=2,
            )
        ],
    )

    assert len(events) == 1

    event = events[0]
    assert event["record_count"] == 2
    assert event["resolution_status"] == "linked_research_history"
    assert event["all_content_identical"] is False
    assert event["has_content_changes"] is True


def test_cross_company_candidate_is_not_linked():
    feed = make_feed([
        {"record_id": "R1", "company": "Acme Industries", "ticker": "ACME"},
        {"record_id": "R2", "company": "Beta Systems", "ticker": "BETA"},
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="cross_company_similarity",
                same_company=False,
            )
        ],
    )

    assert len(events) == 2
    assert get_event_containing(events, "R1")["record_count"] == 1
    assert get_event_containing(events, "R2")["record_count"] == 1


def test_underlying_identity_conflict_prevents_link():
    feed = make_feed([
        {"record_id": "R1", "company": "Acme Industries", "ticker": "ACME"},
        {"record_id": "R2", "company": "Beta Systems", "ticker": "BETA"},
    ])

    # Even if the candidate generator claims the companies match,
    # the underlying records must be checked independently.
    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="repeated_document_or_reissue",
                same_company=True,
            )
        ],
    )

    assert len(events) == 2

    event_1 = get_event_containing(events, "R1")
    event_2 = get_event_containing(events, "R2")

    assert event_1["record_count"] == 1
    assert event_2["record_count"] == 1
    assert event_1["excluded_relationships"]
    assert event_2["excluded_relationships"]


def test_missing_comparable_identity_prevents_link():
    feed = make_feed([
        {"record_id": "R1", "company": "", "ticker": ""},
        {"record_id": "R2", "company": "", "ticker": ""},
    ])

    events = build_event_timeline(
        feed,
        [
            make_pair(
                "R1",
                "R2",
                candidate_type="repeated_document_or_reissue",
                same_company=True,
            )
        ],
    )

    assert len(events) == 2
    assert get_event_containing(events, "R1")["excluded_relationships"]
    assert get_event_containing(events, "R2")["excluded_relationships"]


def test_transitive_chain_is_flagged_for_review():
    feed = make_feed([
        {"record_id": "R1"},
        {"record_id": "R2"},
        {"record_id": "R3"},
    ])

    pairs = [
        make_pair("R1", "R2"),
        make_pair("R2", "R3"),
    ]

    events = build_event_timeline(feed, pairs)

    assert len(events) == 1

    event = events[0]
    assert event["record_count"] == 3
    assert event["transitive_link_risk"] is True
    assert event["resolution_status"] == "review_transitive_link"
    assert event["confidence"] == "medium"
    assert event["missing_direct_links"] == [["R1", "R3"]]


def test_unknown_record_ids_in_candidate_pairs_are_ignored():
    feed = make_feed([
        {"record_id": "R1"},
        {"record_id": "R2"},
    ])

    pairs = [
        make_pair("R1", "UNKNOWN"),
        make_pair("UNKNOWN", "R2"),
    ]

    events = build_event_timeline(feed, pairs)

    assert len(events) == 2
    assert get_event_containing(events, "R1")["record_count"] == 1
    assert get_event_containing(events, "R2")["record_count"] == 1


def test_self_pair_is_ignored():
    feed = make_feed([
        {"record_id": "R1"},
    ])

    events = build_event_timeline(
        feed,
        [make_pair("R1", "R1")],
    )

    assert len(events) == 1
    assert events[0]["record_ids"] == ["R1"]
    assert events[0]["resolution_status"] == "standalone"


def test_duplicate_record_ids_raise_error():
    feed = make_feed([
        {"record_id": "R1"},
        {"record_id": "R1"},
    ])

    with pytest.raises(ValueError, match="duplicate record_id"):
        build_event_timeline(feed, [])


def test_missing_record_ids_raise_error():
    feed = make_feed([
        {"record_id": None},
    ])

    with pytest.raises(ValueError, match="missing record_id"):
        build_event_timeline(feed, [])


def test_missing_required_columns_raise_error():
    feed = pd.DataFrame({
        "record_id": ["R1"],
        "content": ["Some content"],
    })

    with pytest.raises(ValueError, match="headline"):
        build_event_timeline(feed, [])


def test_empty_feed_returns_no_events():
    feed = pd.DataFrame(columns=["record_id", "headline"])

    events = build_event_timeline(feed, [])

    assert events == []


def test_event_ids_are_deterministic_for_same_records():
    feed = make_feed([
        {"record_id": "R2"},
        {"record_id": "R1"},
    ])

    pairs = [make_pair("R1", "R2")]

    first_run = build_event_timeline(feed, pairs)
    second_run = build_event_timeline(feed, pairs)

    first_ids = [event["event_id"] for event in first_run]
    second_ids = [event["event_id"] for event in second_run]

    assert first_ids == second_ids
    assert first_ids == ["EVT-R1"]


def test_events_are_sorted_chronologically():
    feed = make_feed([
        {"record_id": "R2", "published_at": "2026-03-01"},
        {"record_id": "R1", "published_at": "2026-01-01"},
        {"record_id": "R3", "published_at": "2026-02-01"},
    ])

    events = build_event_timeline(feed, [])

    dated_events = [
        event for event in events
        if event["event_start"] is not None
    ]

    dates = [event["event_start"] for event in dated_events]

    assert dates == sorted(dates)
