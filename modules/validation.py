"""Rule-based data quality checks for the research dataset.

Validation reports issues; it does not silently repair or rewrite source data.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modules.entities import build_company_lookup, normalise_name, suggest_company_matches
from modules.ingestion import ResearchDataset


@dataclass
class ValidationIssue:
    severity: str
    check: str
    message: str
    record_id: str | None = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue]

    @property
    def error_count(self) -> int:
        return sum(issue.severity.upper() == "ERROR" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity.upper() == "WARNING" for issue in self.issues)

    @property
    def info_count(self) -> int:
        return sum(issue.severity.upper() == "INFO" for issue in self.issues)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def affected_record_count(self) -> int:
        return len({
            str(issue.record_id)
            for issue in self.issues
            if issue.record_id is not None
            and not pd.isna(issue.record_id)
            and str(issue.record_id).strip()
        })

    @property
    def counts_by_check(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for issue in self.issues:
            check_counts = counts.setdefault(
                issue.check, {"ERROR": 0, "WARNING": 0, "INFO": 0}
            )
            severity = issue.severity.upper()
            check_counts[severity] = check_counts.get(severity, 0) + 1
        return dict(sorted(counts.items()))

    def issues_for(
        self, *, severity: str | None = None, check: str | None = None
    ) -> list[ValidationIssue]:
        result = self.issues
        if severity:
            wanted = severity.upper()
            result = [issue for issue in result if issue.severity.upper() == wanted]
        if check:
            result = [issue for issue in result if issue.check == check]
        return result


def _is_blank(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return not value.strip()
    return not str(value).strip()


def _record_id(row) -> str | None:
    value = row.get("record_id")
    return None if _is_blank(value) else str(value).strip()


def validate_record_ids(feed: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    ids = feed["record_id"].astype("string").str.strip()
    missing_mask = ids.isna() | ids.eq("")
    for index in feed.index[missing_mask]:
        issues.append(ValidationIssue("ERROR", "record_identity", "Missing record ID."))
    duplicate_mask = ids.notna() & ids.ne("") & ids.duplicated(keep=False)
    for index, row in feed.loc[duplicate_mask].iterrows():
        issues.append(ValidationIssue(
            "ERROR", "record_identity", f"Duplicate record ID: {ids.at[index]}",
            _record_id(row),
        ))
    return issues


def validate_required_fields(feed: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    for field, severity in {"headline": "ERROR", "content": "WARNING"}.items():
        values = feed[field].astype("string").str.strip()
        for index in feed.index[values.isna() | values.eq("")]:
            issues.append(ValidationIssue(
                severity, "required_fields", f"Missing {field}.",
                _record_id(feed.loc[index]),
            ))
    return issues


def validate_publication_dates(feed: pd.DataFrame) -> list[ValidationIssue]:
    issues = []
    dates = pd.to_datetime(feed["published_at"], errors="coerce", utc=True)
    for index in feed.index[dates.isna()]:
        issues.append(ValidationIssue(
            "WARNING", "publication_date", "Missing or invalid publication date.",
            _record_id(feed.loc[index]),
        ))
    future_mask = dates.notna() & dates.gt(pd.Timestamp.now(tz="UTC"))
    for index in feed.index[future_mask]:
        issues.append(ValidationIssue(
            "WARNING", "publication_date", "Publication date is in the future.",
            _record_id(feed.loc[index]),
        ))
    return issues


def _master_identity_maps(companies: pd.DataFrame):
    """Return normalized name/alias -> company and ticker -> company maps."""
    name_to_company = {}
    ticker_to_company = {}
    issues = []
    for _, row in companies.iterrows():
        company = row.get("company")
        ticker = row.get("ticker")
        if _is_blank(company):
            issues.append(ValidationIssue(
                "ERROR", "company_master_integrity",
                "Company master row has no canonical company name.",
            ))
            continue
        canonical = str(company).strip()
        key = normalise_name(canonical)
        if key in name_to_company and name_to_company[key] != canonical:
            issues.append(ValidationIssue(
                "WARNING", "company_master_integrity",
                f"Canonical company name collision: {canonical}",
            ))
        name_to_company[key] = canonical
        if not _is_blank(ticker):
            ticker_key = str(ticker).strip().casefold()
            if ticker_key in ticker_to_company and ticker_to_company[ticker_key] != canonical:
                issues.append(ValidationIssue(
                    "WARNING", "company_master_integrity",
                    f"Ticker {ticker} is assigned to multiple companies.",
                ))
            ticker_to_company[ticker_key] = canonical
        aliases = row.get("aliases")
        if not _is_blank(aliases):
            for alias in str(aliases).split(";"):
                alias_key = normalise_name(alias)
                if not alias_key:
                    continue
                previous = name_to_company.get(alias_key)
                if previous is not None and previous != canonical:
                    issues.append(ValidationIssue(
                        "WARNING", "company_master_integrity",
                        f"Alias '{alias.strip()}' maps to both {previous} and {canonical}.",
                    ))
                else:
                    name_to_company[alias_key] = canonical
    return name_to_company, ticker_to_company, issues


def _is_intentionally_unassigned(row) -> bool:
    """Macro/sector-level records may correctly have no single company."""
    sector_value = row.get("sector", "")
    tags_value = row.get("tags", "")
    sector = "" if _is_blank(sector_value) else str(sector_value).casefold()
    tags = "" if _is_blank(tags_value) else str(tags_value).casefold()
    return (
        "macro" in sector
        or "macro" in tags
        or "sector development" in tags
        or "market-wide" in tags
    )


def validate_company_mappings(
    feed: pd.DataFrame, companies: pd.DataFrame
) -> list[ValidationIssue]:
    issues = []
    name_to_company, ticker_to_company, master_issues = _master_identity_maps(companies)
    issues.extend(master_issues)
    lookup = build_company_lookup(companies)

    for _, row in feed.iterrows():
        record_id = _record_id(row)
        company = row.get("company")
        ticker = row.get("ticker")
        company_blank = _is_blank(company)
        ticker_blank = _is_blank(ticker)

        if company_blank and ticker_blank:
            if _is_intentionally_unassigned(row):
                issues.append(ValidationIssue(
                    "INFO", "entity_identity",
                    "No company assigned; record appears to be macro/sector-level research.",
                    record_id,
                ))
            else:
                headline = row.get("headline")
                headline_text = "" if _is_blank(headline) else str(headline).casefold()
                mentioned = []
                # Suggest a company only when a sufficiently distinctive canonical
                # name or alias literally appears in the headline. Never auto-assign.
                for known_name, canonical in name_to_company.items():
                    if len(known_name) >= 8 and known_name in headline_text:
                        if canonical not in mentioned:
                            mentioned.append(canonical)
                suggestion = (
                    f" Headline mentions: {', '.join(mentioned[:2])};"
                    " review whether this company should be assigned."
                    if mentioned else
                    " Review whether the record should be assigned to a company."
                )
                issues.append(ValidationIssue(
                    "WARNING", "entity_identity",
                    "Company-specific-looking record has no company or ticker." + suggestion,
                    record_id,
                ))
            continue

        if company_blank != ticker_blank:
            missing_field = "company name" if company_blank else "ticker"
            issues.append(ValidationIssue(
                "WARNING", "entity_identity",
                f"Partial company identity: {missing_field} is missing.",
                record_id,
            ))

        canonical_from_name = None
        canonical_from_ticker = None

        if not company_blank:
            company_key = normalise_name(str(company))
            canonical_from_name = name_to_company.get(company_key)
            if canonical_from_name is None:
                suggestions = suggest_company_matches(str(company), lookup, limit=3, min_score=0.55)
                suggestion_text = ""
                if suggestions:
                    suggestion_text = "; possible matches: " + ", ".join(
                        f"{item.canonical_company} ({item.ticker}, {item.confidence:.2f})"
                        for item in suggestions
                    )
                issues.append(ValidationIssue(
                    "WARNING", "company_mapping",
                    f"Company name not found in canonical master or aliases: {company}{suggestion_text}",
                    record_id,
                ))

        if not ticker_blank:
            ticker_key = str(ticker).strip().casefold()
            canonical_from_ticker = ticker_to_company.get(ticker_key)
            if canonical_from_ticker is None:
                issues.append(ValidationIssue(
                    "WARNING", "company_mapping", f"Unknown ticker: {ticker}",
                    record_id,
                ))

        if (
            not company_blank and not ticker_blank
            and canonical_from_name is not None
            and canonical_from_ticker is not None
            and canonical_from_name != canonical_from_ticker
        ):
            issues.append(ValidationIssue(
                "WARNING", "company_mapping",
                f"Company–ticker conflict: '{company}' maps to {canonical_from_name}, "
                f"but ticker '{ticker}' maps to {canonical_from_ticker}.",
                record_id,
            ))

    return issues


def validate_source_paths(
    feed: pd.DataFrame, source_root: str | Path
) -> list[ValidationIssue]:
    issues = []
    source_root = Path(source_root).expanduser().resolve()
    for _, row in feed.iterrows():
        source_path = row.get("source_path")
        record_id = _record_id(row)
        if _is_blank(source_path):
            issues.append(ValidationIssue(
                "INFO", "source_integrity", "No source document path supplied.",
                record_id,
            ))
            continue
        path = Path(str(source_path).strip()).expanduser()
        if not path.is_absolute():
            path = source_root / path
        try:
            resolved = path.resolve()
            resolved.relative_to(source_root)
        except (OSError, ValueError):
            issues.append(ValidationIssue(
                "WARNING", "source_integrity",
                f"Source path is invalid or escapes the data directory: {source_path}",
                record_id,
            ))
            continue
        if not resolved.is_file():
            issues.append(ValidationIssue(
                "WARNING", "source_integrity", f"Source file not found: {source_path}",
                record_id,
            ))
    return issues


def validate_dataset(
    data: ResearchDataset, source_root: str | Path
) -> ValidationReport:
    feed = data.feed
    issues = []
    issues.extend(validate_record_ids(feed))
    issues.extend(validate_required_fields(feed))
    issues.extend(validate_publication_dates(feed))
    issues.extend(validate_company_mappings(feed, data.companies))
    issues.extend(validate_source_paths(feed, source_root))
    return ValidationReport(issues=issues)
