from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ingestion import load_dataset
from modules.validation import validate_dataset

data = load_dataset("data/raw")

report = validate_dataset(
    data,
    source_root="data",
)

print("Errors:", report.error_count)
print("Warnings:", report.warning_count)
print("Total issues:", len(report.issues))

for issue in report.issues[:20]:
    print(
        f"{issue.severity} | "
        f"{issue.check} | "
        f"{issue.record_id} | "
        f"{issue.message}"
    )