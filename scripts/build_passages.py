"""Chop the approved source documents into ~300-word passages for retrieval (RAG).

Reads data/doc_list.txt, writes data/passages.jsonl and copies it to the private Drive folder.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from config import DRIVE_DIR as DRIVE
WORDS, OVERLAP, MIN_WORDS = 300, 50, 60


def clean(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)  # image links
    text = re.sub(r"<[^>]+>", " ", text)  # stray html
    text = re.sub(r"\|[-:| ]+\|", " ", text)  # table rulers
    words = [w for w in text.split() if len(w) <= 40]  # binary garbage shows up as giant "words"
    return " ".join(words)


def readable(text: str) -> bool:
    letters = sum(c.isascii() and c.isalpha() for c in text)
    return letters / max(len(text), 1) > 0.6


seen, passages = set(), []
for path in (ROOT / "data" / "doc_list.txt").read_text().split("\n"):
    if not path:
        continue
    words = clean(Path(path).read_text(errors="ignore")).split()
    for start in range(0, len(words), WORDS - OVERLAP):
        chunk = words[start:start + WORDS]
        if len(chunk) < MIN_WORDS:
            continue
        text = " ".join(chunk)
        if not readable(text):
            continue
        digest = hashlib.md5(text.lower().encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        passages.append({"id": len(passages), "source": Path(path).name, "text": text})

out = ROOT / "data" / "passages.jsonl"
out.write_text("".join(json.dumps(p) + "\n" for p in passages))
shutil.copy(out, DRIVE / out.name)
words_total = sum(len(p["text"].split()) for p in passages)
print(f"{len(passages)} passages from {len({p['source'] for p in passages})} documents, {words_total / 1e6:.1f}M words")
print("copied to", DRIVE / out.name)
