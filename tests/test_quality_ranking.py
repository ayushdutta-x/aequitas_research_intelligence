from pathlib import Path

import pandas as pd

from modules.ingestion import ResearchDataset, load_dataset
from modules.ranking import rank_recent_insights
from modules.validation import validate_company_mappings, validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_reports_missing_company_identity_without_misclassifying_macro():
    dataset = load_dataset(ROOT / "data" / "raw")
    report = validate_dataset(dataset, ROOT / "data")

    missing_identity = [
        issue for issue in report.issues
        if issue.check == "entity_identity"
    ]
    assert len(missing_identity) == 47
    assert sum(issue.severity == "INFO" for issue in missing_identity) == 40
    assert sum(issue.severity == "WARNING" for issue in missing_identity) == 7
    assert any("Headline mentions" in issue.message for issue in missing_identity
               if issue.severity == "WARNING")


def test_company_ticker_conflict_is_reported():
    companies = pd.DataFrame([
        {"company": "Alpha Limited", "ticker": "ALPHA", "aliases": "Alpha"},
        {"company": "Beta Limited", "ticker": "BETA", "aliases": "Beta"},
    ])
    feed = pd.DataFrame([
        {"record_id": "R1", "company": "Alpha Limited", "ticker": "BETA"}
    ])

    issues = validate_company_mappings(feed, companies)
    assert any("Company–ticker conflict" in issue.message for issue in issues)


def test_recent_insights_ranking_is_explainable_and_limited():
    feed = pd.DataFrame([
        {
            "record_id": "R1", "published_at": "2026-01-01",
            "headline": "Older report", "content": "Content",
            "company": "Alpha Limited", "ticker": "ALPHA",
            "source_type": "News",
        },
        {
            "record_id": "R2", "published_at": "2026-01-30",
            "headline": "New filing", "content": "Content",
            "company": "Alpha Limited", "ticker": "ALPHA",
            "source_type": "Regulatory filing",
        },
    ])

    ranked = rank_recent_insights(feed, limit=1)
    assert len(ranked) == 1
    assert ranked.iloc[0]["record_id"] == "R2"
    assert ranked.iloc[0]["priority_score"] > 0
    assert "regulatory source" in ranked.iloc[0]["priority_reasons"]


def test_unknown_company_name_includes_fuzzy_review_suggestion():
    companies = pd.DataFrame([
        {"company": "Alpha Limited", "ticker": "ALPHA", "aliases": "Alpha"}
    ])
    feed = pd.DataFrame([
        {"record_id": "R2", "company": "Alphaa Limited", "ticker": "ALPHA"}
    ])

    issues = validate_company_mappings(feed, companies)
    name_issue = next(issue for issue in issues if "Company name not found" in issue.message)
    assert "possible matches" in name_issue.message
    assert "Alpha Limited" in name_issue.message
