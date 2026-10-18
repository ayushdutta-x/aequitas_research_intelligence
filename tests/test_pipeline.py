from pathlib import Path

from modules.ingestion import load_dataset
from modules.validation import validate_dataset
from modules.retrieval import build_research_index, search_research
from modules.deduplication import find_near_duplicates
from modules.events import build_event_timeline

ROOT = Path(__file__).resolve().parents[1]


def test_supplied_dataset_loads_and_source_paths_resolve():
    dataset = load_dataset(ROOT / "data" / "raw")
    report = validate_dataset(dataset, ROOT / "data")

    assert len(dataset.feed) == 250
    assert len(dataset.companies) == 15
    assert len(dataset.watchlist) == 8
    assert report.error_count == 0
    # Forty macro/sector-level records are intentionally unassigned; seven
    # company-specific records lack identity metadata and need review.
    assert report.warning_count == 7
    assert report.info_count == 40
    assert report.affected_record_count == 47


def test_retrieval_returns_relevant_results_and_honours_filters():
    dataset = load_dataset(ROOT / "data" / "raw")
    index = build_research_index(dataset)

    results = search_research("order execution", index, limit=5)
    assert not results.empty
    assert len(results) <= 5
    assert "relevance_score" in results.columns

    sector_results = search_research(
        "order execution", index, limit=10, sector="Industrials"
    )
    assert all(
        sector_results["sector"].fillna("").str.contains("Industrials", case=False)
    )


def test_event_histories_preserve_each_record_once():
    dataset = load_dataset(ROOT / "data" / "raw")
    pairs = find_near_duplicates(dataset.feed)
    events = build_event_timeline(dataset.feed, pairs)

    represented = [rid for event in events for rid in event["record_ids"]]
    expected = dataset.feed["record_id"].tolist()
    assert len(pairs) == 189
    assert len(events) == 237
    assert sorted(represented) == sorted(expected)
    assert len(represented) == len(set(represented))
