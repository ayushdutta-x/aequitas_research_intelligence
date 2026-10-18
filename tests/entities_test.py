from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ingestion import load_dataset
from modules.entities import build_company_lookup, normalise_name, resolve_company, suggest_company_matches

data = load_dataset("data/raw")
lookup = build_company_lookup(data.companies)

# print(normalise_name("  BLUEPEAK  "))
# print(normalise_name("Vardhan   Engg"))

# for query in ["Aranya Bank", "ARANYABANK", "Meridian Capital"]:
#     print(query, "->", lookup.get(normalise_name(query)))


# for query in [
#     "Aranya Bank",
#     "ARANYABANK",
#     "Meridian Capital",
#     "Vardhan Engg",
#     "",
# ]:
#     result = resolve_company(query, lookup)
#     print(result)



# test queries
query = "quantum LEAF"

exact = resolve_company(query, lookup)
print("Exact result:", exact.match_type)

candidates = suggest_company_matches(query, lookup)

for candidate in candidates:
    print(
        candidate.canonical_company,
        candidate.ticker,
        round(candidate.confidence, 3),
    )