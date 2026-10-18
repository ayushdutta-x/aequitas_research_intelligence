"""Equity Research Intelligence Agent: terminal research workflow."""
from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

import pandas as pd

from modules.ingestion import load_dataset, DataLoadError
from modules.validation import validate_dataset
from modules.entities import build_company_lookup, resolve_company, suggest_company_matches
from modules.retrieval import build_research_index, search_research
from modules.deduplication import find_near_duplicates
from modules.events import build_event_timeline
from modules.ranking import rank_recent_insights

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
SOURCE_ROOT = PROJECT_DIR / "data"
DISPLAY_LIMIT = 12


def _text(value, fallback="—"):
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def _date_key(value):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return parsed


def _print_record(record, number=None, show_content=True):
    prefix = f"{number}. " if number is not None else ""
    print(f"\n{prefix}{_text(record.get('headline'), 'Untitled research record')}")
    print(f"  ID: {_text(record.get('record_id'))} | Date: {_text(record.get('published_at'))}")
    print(f"  Company: {_text(record.get('company'))} ({_text(record.get('ticker'))})")
    print(f"  Sector: {_text(record.get('sector'))} | Source: {_text(record.get('source_name'))} [{_text(record.get('source_type'))}]")
    if "relevance_score" in record and pd.notna(record.get("relevance_score")):
        print(f"  Search relevance: {float(record['relevance_score']):.3f}")
    if show_content:
        content = _text(record.get("content"), "")
        if content:
            print(f"  Context: {content[:320]}{'…' if len(content) > 320 else ''}")
    if _text(record.get("source_path"), ""):
        print(f"  Source path: {_text(record.get('source_path'))}")


def _display_records(records, limit=DISPLAY_LIMIT):
    if records is None or records.empty:
        print("No matching research records found.")
        return
    shown = records.head(limit)
    for i, (_, record) in enumerate(shown.iterrows(), 1):
        _print_record(record, i)
    print(f"\nShowing {len(shown)} of {len(records)} matching record(s).")


def _resolve_source_path(source_path):
    """Resolve a feed path safely inside the project's data directory."""
    if not source_path or pd.isna(source_path):
        return None
    path = Path(str(source_path).strip()).expanduser()
    if not path.is_absolute():
        path = SOURCE_ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(SOURCE_ROOT.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def initialise_agent():
    dataset = load_dataset(DATA_DIR)
    report = validate_dataset(dataset, SOURCE_ROOT)
    lookup = build_company_lookup(dataset.companies)
    index = build_research_index(dataset)
    print("Building evidence-linked research histories (this may take a moment)...")
    candidate_pairs = find_near_duplicates(dataset.feed)
    events = build_event_timeline(dataset.feed, candidate_pairs)
    return dataset, report, lookup, index, candidate_pairs, events


def startup_summary(dataset, report, candidate_pairs, events):
    linked = sum(event.get("record_count", 0) > 1 for event in events)
    print("\n" + "=" * 68)
    print("          EQUITY RESEARCH INTELLIGENCE AGENT")
    print("=" * 68)
    print(f"Research records loaded:       {len(dataset.feed):>6}")
    print(f"Companies in master:           {len(dataset.companies):>6}")
    print(f"Watchlist entries:             {len(dataset.watchlist):>6}")
    print(f"Validation errors:             {report.error_count:>6}")
    print(f"Validation warnings:           {report.warning_count:>6}")
    print(f"Validation info notes:         {report.info_count:>6}")
    print(f"Candidate relationships:       {len(candidate_pairs):>6}")
    print(f"Linked research histories:     {linked:>6}")
    print(f"Total event groups:            {len(events):>6}")
    print("=" * 68)
    if report.error_count:
        print("Data-quality errors were found; review option 1 before relying on results.")
    elif report.warning_count:
        print("Ingestion completed. Some records have warnings; see option 1 for details.")
    elif report.info_count:
        print("Ingestion completed. Informational notes describe records intentionally left unassigned.")
    else:
        print("Ingestion and validation completed without reported issues.")


def data_quality_menu(dataset, report):
    print("\n--- Data Quality Report ---")
    print(f"Records: {len(dataset.feed)} | Errors: {report.error_count} | Warnings: {report.warning_count} | Info: {report.info_count}")
    if not report.issues:
        print("No validation issues found.")
        return
    print("\nIssue counts by check:")
    for check, counts in report.counts_by_check.items():
        total = sum(counts.values())
        print(f"  {check}: {total} (errors {counts['ERROR']}, warnings {counts['WARNING']}, info {counts['INFO']})")
    choice = input("\nShow [E]rrors, [W]arnings, [I]nfo, or [A]ll? [A]: ").strip().casefold() or "a"
    severity = {"e": "ERROR", "w": "WARNING", "i": "INFO"}.get(choice)
    issues = report.issues_for(severity=severity) if severity else report.issues
    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    issues = sorted(
        issues,
        key=lambda item: (severity_order.get(item.severity.upper(), 3), item.check, item.record_id or ""),
    )
    if not issues:
        print("No issues in that category.")
        return
    for issue in issues[:DISPLAY_LIMIT]:
        rec = f" | record {issue.record_id}" if issue.record_id else ""
        print(f"[{issue.severity}] {issue.check}: {issue.message}{rec}")
    if len(issues) > DISPLAY_LIMIT:
        print(f"… {len(issues) - DISPLAY_LIMIT} additional issue(s) not shown.")


def search_menu(index):
    print("\n--- Search Research ---")
    query = input("Keywords or phrase: ").strip()
    if not query:
        print("Enter a search query to continue.")
        return
    company = input("Company filter (optional): ").strip() or None
    ticker = input("Ticker filter (optional): ").strip() or None
    sector = input("Sector filter (optional): ").strip() or None
    source_type = input("Source type filter (optional): ").strip() or None
    start = input("Start date YYYY-MM-DD (optional): ").strip() or None
    end = input("End date YYYY-MM-DD (optional): ").strip() or None
    sort_by = "date" if input("Sort [1] relevance / [2] newest [1]: ").strip() == "2" else "relevance"
    try:
        limit_text = input("Maximum results [10]: ").strip()
        limit = int(limit_text) if limit_text else 10
        results = search_research(query, index, limit=limit, company=company, ticker=ticker,
                                  sector=sector, source_type=source_type,
                                  start_date=start, end_date=end, sort_by=sort_by)
        _display_records(results)
    except (ValueError, TypeError) as exc:
        print(f"Could not run search: {exc}")


def _company_records(dataset, lookup, query):
    match = resolve_company(query, lookup)
    feed = dataset.feed
    if match.match_type != "unresolved":
        ticker = str(match.ticker).strip().casefold()
        rows = feed[feed["ticker"].fillna("").astype(str).str.strip().str.casefold() == ticker].copy()
        return match.canonical_company, rows, match
    # Permit a direct partial company/ticker lookup if not in master.
    q = query.casefold()
    mask = feed["company"].fillna("").astype(str).str.casefold().str.contains(q, regex=False)
    mask |= feed["ticker"].fillna("").astype(str).str.casefold().eq(q)
    return query, feed[mask].copy(), match


def company_timeline_menu(dataset, lookup, events):
    print("\n--- Company Development Timeline ---")
    query = input("Company name, alias, or ticker: ").strip()
    if not query:
        return
    name, records, match = _company_records(dataset, lookup, query)
    if records.empty:
        print(f"No research records found for '{query}'.")
        if match.match_type == "unresolved":
            suggestions = suggest_company_matches(query, lookup)
            if suggestions:
                print("Possible matches:")
                for s in suggestions:
                    print(f"  {s.canonical_company} ({s.ticker}) — {s.confidence:.2f}")
        return
    records["_date_sort"] = pd.to_datetime(records["published_at"], errors="coerce", utc=True)
    records = records.sort_values("_date_sort", na_position="last", kind="stable")
    print(f"\n{name}: {len(records)} research record(s), chronological")
    for i, (_, row) in enumerate(records.iterrows(), 1):
        print(f"{i:>3}. {_text(row.get('published_at'))} | {_text(row.get('record_id'))} | {_text(row.get('headline'))}")
    event_map = {rid: event for event in events for rid in event.get("record_ids", [])}
    selection = input("\nEnter a record ID to inspect its linked history, or Enter to return: ").strip()
    if selection:
        event = event_map.get(selection)
        if not event:
            print("No linked event history found for that ID.")
        else:
            _display_event(dataset, event)


def _display_event(dataset, event):
    print(f"\nHistory {event.get('event_id')} | {event.get('relationship')} | status={event.get('resolution_status')} | confidence={event.get('confidence')}")
    print(f"Records: {event.get('record_count')} | range: {_text(event.get('event_start'))} → {_text(event.get('latest_update'))}")
    if event.get("transitive_link_risk"):
        print("Review note: some records are connected indirectly without every pair having direct evidence.")
    ids = set(event.get("record_ids", []))
    records = dataset.feed[dataset.feed["record_id"].isin(ids)].copy()
    records["_date_sort"] = pd.to_datetime(records["published_at"], errors="coerce", utc=True)
    records = records.sort_values("_date_sort", na_position="last", kind="stable")
    _display_records(records, limit=30)
    evidence = event.get("relationship_evidence", [])
    if evidence:
        print("Relationship evidence:")
        for item in evidence:
            print(f"  {item.get('record_id_1')} ↔ {item.get('record_id_2')}: {item.get('relationship')}, similarity={item.get('similarity')}, identical content={item.get('same_content')}")


def sector_topic_menu(dataset, index):
    print("\n--- Sector / Topic Tracker ---")
    print("Enter a sector name (e.g. Industrials) or topic keywords (e.g. capex, order execution).")
    query = input("Sector or topic: ").strip()
    if not query:
        return
    feed = dataset.feed
    sector_values = feed["sector"].fillna("").astype(str)
    exact_sector = sector_values.str.casefold().eq(query.casefold())
    if exact_sector.any():
        rows = feed[exact_sector].copy()
        label = f"sector '{query}'"
    else:
        try:
            rows = search_research(query, index, limit=len(feed), sort_by="date")
        except (ValueError, TypeError) as exc:
            print(f"Could not track topic: {exc}")
            return
        label = f"topic '{query}'"
    if rows.empty:
        print(f"No records found for {label}.")
        return
    rows["_date_sort"] = pd.to_datetime(rows["published_at"], errors="coerce", utc=True)
    rows = rows.sort_values("_date_sort", na_position="last", kind="stable")
    print(f"\nTimeline for {label}: {len(rows)} record(s)")
    for _, row in rows.iterrows():
        print(f"{_text(row.get('published_at'))} | {_text(row.get('company'))} ({_text(row.get('ticker'))}) | {_text(row.get('headline'))}")
    print("\nRecords are ordered chronologically; topic matches use the existing TF-IDF retrieval index.")


def follow_up_menu(dataset, index, events):
    print("\n--- Follow Up on Research ---")
    query = input("Enter a record ID, or search terms to find a record: ").strip()
    if not query:
        return
    feed = dataset.feed
    direct = feed[feed["record_id"].fillna("").astype(str).str.casefold() == query.casefold()]
    if direct.empty:
        try:
            direct = search_research(query, index, limit=8)
        except (ValueError, TypeError) as exc:
            print(f"Search failed: {exc}")
            return
        if direct.empty:
            print("No record matched that ID or query.")
            return
        _display_records(direct, limit=8)
        chosen = input("Enter the ID to follow up: ").strip()
    else:
        chosen = str(direct.iloc[0]["record_id"])
    record = feed[feed["record_id"].astype(str) == chosen]
    if record.empty:
        print("That record ID was not found.")
        return
    print("\nSelected record:")
    _print_record(record.iloc[0])
    event_map = {rid: event for event in events for rid in event.get("record_ids", [])}
    event = event_map.get(chosen)
    if event and event.get("record_count", 0) > 1:
        print("\nLinked research history:")
        _display_event(dataset, event)
    # Show related candidates, including rejected pairs, without presenting them as confirmed links.
    if event:
        rejected = event.get("excluded_relationships", [])
        if rejected:
            print("\nRelated candidates not automatically linked (review, not confirmed):")
            for item in rejected[:8]:
                print(f"  {item.get('record_id_1')} ↔ {item.get('record_id_2')} | {item.get('relationship')} | {item.get('reason_not_linked')}")
    current_date = _date_key(record.iloc[0].get("published_at"))
    ticker = _text(record.iloc[0].get("ticker"), "").casefold()
    if ticker:
        related = feed[(feed["ticker"].fillna("").astype(str).str.casefold() == ticker) & (feed["record_id"].astype(str) != chosen)].copy()
        if not related.empty:
            related["_date_sort"] = pd.to_datetime(related["published_at"], errors="coerce", utc=True)
            if pd.notna(current_date):
                related["_distance"] = (related["_date_sort"] - current_date).abs()
                related = related.sort_values("_distance", na_position="last")
            else:
                related = related.sort_values("_date_sort", ascending=False, na_position="last")
            print("\nOther research for the same ticker:")
            _display_records(related.drop(columns=[c for c in ["_distance", "_date_sort"] if c in related.columns]), limit=6,)


def recent_insights_menu(dataset, events):
    print("\\n--- Recent Insights ---")
    try:
        ranked = rank_recent_insights(dataset.feed, limit=6)
    except (ValueError, TypeError) as exc:
        print(f"Could not rank recent research: {exc}")
        return

    event_map = {rid: event for event in events for rid in event.get("record_ids", [])}
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        _print_record(row, i, show_content=False)
        print(f"  Priority score: {row['priority_score']:.1f}/100")
        print(f"  Why prioritized: {row['priority_reasons']}")
        event = event_map.get(row.get("record_id"))
        if event and event.get("record_count", 0) > 1:
            print(f"  Linked history: {event.get('event_id')} "
                  f"({event.get('record_count')} records, {event.get('resolution_status')})")

    print("\\nPriority is a transparent triage aid, not an investment recommendation or prediction of returns.")


def view_source_menu(dataset):
    print("\n--- View Source Document ---")
    record_id = input("Record ID (e.g. RF0043): ").strip()
    rows = dataset.feed[dataset.feed["record_id"].fillna("").astype(str).str.casefold() == record_id.casefold()]
    if rows.empty:
        print("Record ID not found.")
        return
    row = rows.iloc[0]
    path = _resolve_source_path(row.get("source_path"))
    if path is None:
        print(f"Source document is missing or its path is outside the project data directory: {_text(row.get('source_path'))}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"Could not read source document: {exc}")
        return
    print(f"\nSource: {path.relative_to(PROJECT_DIR)}")
    print("-" * 72)
    print(text[:12000])
    if len(text) > 12000:
        print("\n[Output truncated at 12,000 characters.] ")
    print("-" * 72)
    if input("Open this file in the system's default text viewer? [y/N]: ").strip().casefold() == "y":
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            elif os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                print("Automatic opening is not supported on this operating system.")
        except OSError as exc:
            print(f"Could not launch a viewer: {exc}")


def main():
    print("\nWelcome to the Equity Research Intelligence Agent.")
    print("Datasets queued for ingestion:")
    for name in ("research_feed.csv", "company_master.csv", "watchlist.csv"):
        print(f"  • {DATA_DIR / name}")
    if input("\nIngest and validate these datasets now? [Y/n]: ").strip().casefold() in {"n", "no"}:
        print("Ingestion cancelled. Goodbye.")
        return
    try:
        dataset, report, lookup, index, candidate_pairs, events = initialise_agent()
    except (DataLoadError, OSError, ValueError) as exc:
        print(f"\nCould not initialise the agent: {exc}")
        print("Check the data files and installed dependencies, then run again.")
        return
    startup_summary(dataset, report, candidate_pairs, events)
    while True:
        print("\nMAIN MENU")
        print("1. Data Quality Report")
        print("2. Search Research")
        print("3. Company Development Timeline")
        print("4. Sector / Topic Tracker")
        print("5. Follow Up on Research")
        print("6. Recent Insights")
        print("7. View Source Document")
        print("8. Exit")
        choice = input("\nSelect an option [1-8]: ").strip()
        if choice == "1":
            data_quality_menu(dataset, report)
        elif choice == "2":
            search_menu(index)
        elif choice == "3":
            company_timeline_menu(dataset, lookup, events)
        elif choice == "4":
            sector_topic_menu(dataset, index)
        elif choice == "5":
            follow_up_menu(dataset, index, events)
        elif choice == "6":
            recent_insights_menu(dataset, events)
        elif choice == "7":
            view_source_menu(dataset)
        elif choice == "8":
            print("Exiting Research Tool.")
            break
        else:
            print("Select a number from 1 to 8.")


if __name__ == "__main__":
    main()
