"""Clean the generated flashcards and split them into train / validation / test chat files.

Test examples come from documents the model never trains on, so the head-to-head is fair.
Writes data/dataset/{train,val,test}.jsonl and copies them to the private Drive folder.

  python3 scripts/prepare_dataset.py                 # full run
  python3 scripts/prepare_dataset.py --smoke         # 140-card test set -> Drive BrandManagerAI/smoke
"""
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = "--smoke" in sys.argv
SRC = ROOT / "data" / "generated" / ("gemini.jsonl" if SMOKE else "full.jsonl")
OUT = ROOT / "data" / ("dataset_smoke" if SMOKE else "dataset")
from config import BLOCKED_TERMS, DRIVE_DIR as DRIVE
if SMOKE:
    DRIVE = DRIVE / "smoke"
HOLDOUT = 2 if SMOKE else 25  # whole documents held out for each of test and val

SYSTEM = (
    "You are an experienced consumer brand manager. You give founders and marketers direct, "
    "practical advice: a clear recommendation, the reason behind it, and a sharp question when "
    "something important is missing."
)
LEAK = re.compile(BLOCKED_TERMS)


def clean(text: str) -> str:
    text = text.replace("—", ", ").replace("–", "-")
    return re.sub(r"[ \t]+", " ", text).strip()


rows, dropped = [], defaultdict(int)
for line in SRC.open():
    r = json.loads(line)
    if "type" not in r:
        dropped["malformed"] += 1
        continue
    req, rep = clean(r.get("request", "")), clean(r.get("reply", ""))
    if not req or not rep:
        dropped["empty"] += 1
    elif LEAK.search(req + " " + rep):
        dropped["mentions source-company terms"] += 1
    elif not 80 <= len(rep.split()) <= 450:
        dropped["reply length"] += 1
    else:
        rows.append({"source": r["source"], "type": r["type"], "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": req},
            {"role": "assistant", "content": rep},
        ]})

by_doc = defaultdict(list)
for r in rows:
    by_doc[r["source"]].append(r)
docs = sorted(by_doc)
random.Random(42).shuffle(docs)
test_docs, val_docs = set(docs[:HOLDOUT]), set(docs[HOLDOUT:2 * HOLDOUT])
splits = {"train": [], "val": [], "test": []}
for d in docs:
    splits["test" if d in test_docs else "val" if d in val_docs else "train"].extend(by_doc[d])

OUT.mkdir(parents=True, exist_ok=True)
DRIVE.mkdir(parents=True, exist_ok=True)
for name, items in splits.items():
    random.Random(1).shuffle(items)
    path = OUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(i) + "\n" for i in items))
    shutil.copy(path, DRIVE / path.name)
    print(f"{name}: {len(items)} examples")
print("dropped:", dict(dropped))
print("copied to", DRIVE)
