"""Generate training examples from each sample doc with Gemini. Logs token usage for cost checks.

Run: GEMINI_API_KEY=... python3 scripts/gen_gemini.py
"""
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-3.8-flash"
DOCS = sorted((ROOT / "data" / "sample_docs").glob("*.md"))
OUT = ROOT / "data" / "generated" / "gemini.jsonl"
PROMPT = (ROOT / "scripts" / "prompt.md").read_text()
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_DOC_CHARS = 60_000

usage = {"input": 0, "output": 0}
with OUT.open("w") as out:
    for doc in DOCS:
        body = {
            "contents": [{"parts": [{"text": PROMPT + doc.read_text(errors="ignore")[:MAX_DOC_CHARS]}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp = json.load(r)
                break
            except Exception as e:
                print(f"  retry {attempt + 1} for {doc.name}: {e}")
                time.sleep(5)
        else:
            print(f"FAILED {doc.name}")
            continue
        meta = resp.get("usageMetadata", {})
        usage["input"] += meta.get("promptTokenCount", 0)
        usage["output"] += meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        text = resp["candidates"][0]["content"]["parts"][-1]["text"]
        try:
            examples = json.loads(text)["examples"]
        except (json.JSONDecodeError, KeyError):
            print(f"BAD JSON {doc.name}")
            continue
        for ex in examples:
            out.write(json.dumps({"source": doc.name, "model": MODEL, **ex}) + "\n")
        print(f"{doc.name[:50]:<50} {len(examples)} examples")

print(json.dumps(usage))
