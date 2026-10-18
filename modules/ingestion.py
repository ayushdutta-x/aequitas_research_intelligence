
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEED_COLUMNS = {
    "record_id",
    "published_at",
    "source_type",
    "source_name",
    "headline",
    "content",
    "company",
    "ticker",
    "sector",
    "industry",
    "country",
    "tags",
    "source_path",
    "language",
}

COMPANY_COLUMNS = {
    "company",
    "ticker",
    "aliases",
    "sector",
    "industry",
    "geography",
    "description",
}

WATCHLIST_COLUMNS = {
    "ticker", 
    "company"
}


class DataLoadError(Exception):
    """Raised when a required research dataset cannot be loaded."""

@dataclass
class ResearchDataset:
    feed: pd.DataFrame
    companies: pd.DataFrame
    watchlist: pd.DataFrame


def read_csv_checked(
    path: str | Path,
    required_columns: set[str],
) -> pd.DataFrame:

    path = Path(path)

    if not path.is_file():
        raise DataLoadError(f"Required data file not found: {path}")

    try:
        dataframe = pd.read_csv(
            path,
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(
            f"Data file is empty or has no readable header: {path}"
        ) from exc
    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise DataLoadError(
            f"Could not read data file '{path}': {exc}"
        ) from exc

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise DataLoadError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    for column in result.columns:
        values = result[column].astype("string").str.strip()
        result[column] = values.replace("", pd.NA)

    return result


def parse_publication_dates(df: pd.DataFrame) -> pd.DataFrame:

    if "published_at" not in df.columns:
        raise DataLoadError(
            "Cannot parse publication dates: "
            "the 'published_at' column is missing."
        )

    result = df.copy()

    result["published_at"] = pd.to_datetime(
        result["published_at"],
        errors="coerce",
    )

    return result


def load_dataset(data_dir: str | Path) -> ResearchDataset:

    data_dir = Path(data_dir).expanduser()

    feed_path = data_dir / "research_feed.csv"
    company_path = data_dir / "company_master.csv"
    watchlist_path = data_dir / "watchlist.csv"

    feed = normalise_dataframe(
        read_csv_checked(feed_path, FEED_COLUMNS)
    )

    companies = normalise_dataframe(
        read_csv_checked(company_path, COMPANY_COLUMNS)
    )

    watchlist = normalise_dataframe(
        read_csv_checked(watchlist_path, WATCHLIST_COLUMNS)
    )

    feed = parse_publication_dates(feed)

    return ResearchDataset(
        feed=feed,
        companies=companies,
        watchlist=watchlist,
    )
