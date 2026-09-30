"""List every source document for the full run: approved folders only, readable prose, no duplicates.

Writes data/doc_list.txt (one path per line) and prints a size and cost estimate.
"""
import hashlib
from pathlib import Path

from config import INCLUDE_FOLDERS as INCLUDE, SOURCE_DIR as SRC
ROOT = Path(__file__).resolve().parent.parent
MAX_DOC_CHARS = 60_000

SKIP_PARTS = ("99 Cleanup", "90 Source", "98 Needs Review")


def usable(text: str) -> bool:
    if len(text) < 1_500 or "base64," in text:
        return False
    return text.count("|") / len(text) < 0.02  # spreadsheet dumps teach nothing


seen, keep, input_chars = set(), [], 0
for folder in INCLUDE:
    for p in sorted((SRC / folder).rglob("*.md")):
        if any(part in str(p) for part in SKIP_PARTS):
            continue
        text = p.read_text(errors="ignore")
        digest = hashlib.md5(text.encode()).hexdigest()
        if digest in seen or not usable(text):
            continue
        seen.add(digest)
        keep.append(str(p))
        input_chars += min(len(text), MAX_DOC_CHARS)

(ROOT / "data" / "doc_list.txt").write_text("\n".join(keep) + "\n")
in_tok = input_chars / 4 + len(keep) * 700  # document text + instructions
out_tok = len(keep) * 5_200  # measured per document in the 18-doc test
cost = in_tok / 1e6 * 0.375 + out_tok / 1e6 * 1.875  # Gemini 3.8 Flash batch price
print(f"{len(keep)} documents, ~{in_tok / 1e6:.1f}M tokens in, ~{out_tok / 1e6:.1f}M out, ~{len(keep) * 8:,} examples")
print(f"estimated batch cost: ${cost:.2f}")
