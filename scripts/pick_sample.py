"""Pick a spread of 20 readable documents from the source archive for the generator test.

Reads only. Copies the chosen markdown files into data/sample_docs/.
"""
import hashlib
import random
import shutil
from pathlib import Path

from config import INCLUDE_FOLDERS, SOURCE_DIR as SRC
OUT = Path.home() / "BrandManagerAI" / "data" / "sample_docs"

INCLUDE = INCLUDE_FOLDERS[:13]
PER_FOLDER = {name: 1 for name in INCLUDE}
PER_FOLDER.update({INCLUDE[0]: 3, INCLUDE[1]: 2, INCLUDE[2]: 2, INCLUDE[3]: 2})  # 20 total


def looks_like_prose(text: str) -> bool:
    if not 3_000 <= len(text) <= 60_000:
        return False
    if "base64," in text:
        return False
    pipe_ratio = text.count("|") / len(text)
    return pipe_ratio < 0.02  # skip spreadsheet dumps


random.seed(7)
OUT.mkdir(parents=True, exist_ok=True)
seen = set()
picked = []
for folder, n in PER_FOLDER.items():
    candidates = []
    for p in sorted((SRC / folder).rglob("*.md")):
        if "99 Cleanup" in str(p) or "90 Source" in str(p):
            continue
        text = p.read_text(errors="ignore")
        digest = hashlib.md5(text.encode()).hexdigest()
        if digest in seen or not looks_like_prose(text):
            continue
        candidates.append((p, digest))
    for p, digest in random.sample(candidates, min(n, len(candidates))):
        seen.add(digest)
        dest = OUT / f"{len(picked) + 1:02d}_{p.name}"
        shutil.copy(p, dest)
        picked.append((folder, p.name, p.stat().st_size))

for i, (folder, name, size) in enumerate(picked, 1):
    print(f"{i:02d}  {size // 1000:>4} KB  {folder.split('/')[-1][:40]:<40}  {name[:60]}")
print(f"\n{len(picked)} documents copied to {OUT}")
