# Equity Research Intelligence Agent

A local, terminal-based research repository assistant built around the supplied fictional Aequitas research data pack. It provides searchable research records, company timelines, sector/topic timelines, source-document access, data-quality checks, and evidence-linked research histories.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Setup (Linux/macOS)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python agent.py
```

Confirm ingestion when prompted. The application loads the three CSV datasets in `data/raw/`, resolves document references from the `data/` directory (for example, `raw_documents/RF0043.txt`), validates metadata and source paths, builds the search index, ranks recent records, and builds evidence-linked research histories before showing the menu.

## Menu

1. **Data Quality Report** — review validation errors, warnings, and informational issues.
2. **Search Research** — TF-IDF keyword search with company, ticker, sector, source-type, and date filters.
3. **Company Development Timeline** — chronological company research and linked history inspection.
4. **Sector / Topic Tracker** — chronological records for a sector or retrieval-based topic query.
5. **Follow Up on Research** — inspect a record, its linked history, rejected candidate links, and other records for the same ticker.
6. **Recent Insights** — six recent/high-priority records with an explainable triage score.
7. **View Source Document** — display the source text associated with a record and optionally open it in the system viewer.
8. **Exit**

The recent-insights score is implemented in `modules/ranking.py` and is a transparent prioritization aid based on recency relative to the newest dated record in the dataset, source type, and metadata completeness. It is not an investment recommendation or return prediction.

## Tests

```bash
python -m pytest -v
```

The suite tests ingestion, data quality, entity/ticker conflicts, retrieval, ranking, deduplication, event-history linking, and the end-to-end pipeline.

## Project layout

```text
agent.py
modules/
  ingestion.py
  validation.py
  entities.py
  retrieval.py
  deduplication.py
  events.py
  ranking.py
tests/                 # automated regression and integration tests
DEMO_SCRIPT.md         # suggested walkthrough narration
data/
  raw/                 # CSV repository, company master, watchlist, workbook
  raw_documents/       # source context documents
 tests/
```

## Interpretation notes

- TF-IDF search measures lexical relevance, not factual correctness.
- Candidate duplicate relationships are heuristic evidence, not proof that records describe the same real-world event.
- The event-history builder links only approved relationship types with supporting identity evidence; uncertain/rejected candidates remain reviewable.
- Missing source files are reported rather than silently substituted.
- Macro/sector-level records with no company assignment are informational; company-specific-looking records missing both company and ticker are warnings. Suggested names are review aids and are never auto-assigned.
- Company names and tickers are checked together so a known company paired with another company's known ticker is reported as a conflict.
