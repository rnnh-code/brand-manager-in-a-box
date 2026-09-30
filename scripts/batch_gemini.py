"""Full run through the Gemini Batch API (half price, finishes within 24 hours).

  python3 scripts/batch_gemini.py submit N   # build part N of PARTS, upload, start the job
  python3 scripts/batch_gemini.py status N   # show job state for part N
  python3 scripts/batch_gemini.py fetch      # download finished parts into data/generated/full.jsonl

The run is split into PARTS jobs because the account tier caps queued batch tokens at 3M.

Needs GEMINI_API_KEY in the environment.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "gemini-3.8-flash"
API = "https://generativelanguage.googleapis.com"
KEY = os.environ["GEMINI_API_KEY"]
MAX_DOC_CHARS = 60_000
PARTS = 3
PART = int(sys.argv[2]) if len(sys.argv) > 2 else 0
JOB_FILE = ROOT / "data" / f"batch_job_{PART}.json"
REQUESTS = ROOT / "data" / f"batch_requests_{PART}.jsonl"
OUT = ROOT / "data" / "generated" / "full.jsonl"

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "examples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING", "enum": ["advice", "critique", "explain", "pushback"]},
                    "request": {"type": "STRING"},
                    "reply": {"type": "STRING"},
                },
                "required": ["type", "request", "reply"],
            },
        }
    },
    "required": ["examples"],
}


def call(url, data=None, headers=None, raw=False):
    req = urllib.request.Request(url, data=data, headers={"x-goog-api-key": KEY, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.read() if raw else json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='ignore')[:2000]}") from None


COST_LIMIT = 25.0  # dollars; Rai approved ~$20 and asked to be stopped above ~$25
DONE = ("JOB_STATE_SUCCEEDED", "BATCH_STATE_SUCCEEDED")


def part_results(part):
    """Download a finished part's results once, cache them, and return (rows, cost)."""
    raw = ROOT / "data" / "generated" / f"full_raw_{part}.jsonl"
    if not raw.exists():
        job = call(f"{API}/v1beta/{json.loads((ROOT / 'data' / f'batch_job_{part}.json').read_text())['name']}")
        if state(job) not in DONE:
            raise RuntimeError(f"part {part} not finished: {state(job)}")
        (ROOT / "data" / f"batch_job_{part}_final.json").write_text(json.dumps(job, indent=2))
        results_file = job.get("response", {}).get("responsesFile") or job.get("dest", {}).get("fileName") \
            or job.get("metadata", {}).get("output", {}).get("responsesFile")
        if not results_file:
            raise RuntimeError(f"no results file in job reply; see data/batch_job_{part}_final.json")
        raw.write_bytes(call(f"{API}/download/v1beta/{results_file}:download?alt=media", raw=True))
    rows = [json.loads(line) for line in raw.open()]
    tok_in = tok_out = 0
    for row in rows:
        meta = row.get("response", {}).get("usageMetadata", {})
        tok_in += meta.get("promptTokenCount", 0)
        tok_out += meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
    return rows, tok_in / 1e6 * 0.375 + tok_out / 1e6 * 1.875


def submit():
    if PART > 0:  # cost guard: project the total from the parts already done
        spent = sum(part_results(p)[1] for p in range(PART))
        projected = spent / PART * PARTS
        print(f"parts 0-{PART - 1} cost ${spent:.2f}; projected total ${projected:.2f}")
        if projected > COST_LIMIT:
            sys.exit(f"STOP: projected ${projected:.2f} is over the ${COST_LIMIT:.0f} limit Rai set")
    prompt = (ROOT / "scripts" / "prompt.md").read_text()
    docs = (ROOT / "data" / "doc_list.txt").read_text().split("\n")
    with REQUESTS.open("w") as f:
        docs = [p for p in docs if p]
        for i, path in enumerate(docs):
            if i % PARTS != PART:
                continue
            text = Path(path).read_text(errors="ignore")[:MAX_DOC_CHARS]
            f.write(json.dumps({
                "key": f"{i:05d}|{Path(path).name}",
                "request": {
                    "contents": [{"parts": [{"text": prompt + text}]}],
                    "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA},
                },
            }) + "\n")
    size = REQUESTS.stat().st_size
    # Resumable upload: start, then send the bytes.
    start = urllib.request.Request(
        f"{API}/upload/v1beta/files",
        data=json.dumps({"file": {"display_name": f"brand-manager-requests-{PART}"}}).encode(),
        headers={
            "x-goog-api-key": KEY,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(start) as r:
        upload_url = r.headers["X-Goog-Upload-URL"]
    uploaded = call(upload_url, data=REQUESTS.read_bytes(), headers={
        "Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    file_name = uploaded["file"]["name"]
    for attempt in range(6):  # the queue limit can still count the part that just finished
        try:
            job = call(f"{API}/v1beta/models/{MODEL}:batchGenerateContent",
                       data=json.dumps({"batch": {"display_name": f"brand-manager-part-{PART}",
                                                  "input_config": {"file_name": file_name}}}).encode(),
                       headers={"Content-Type": "application/json"})
            break
        except RuntimeError as e:
            if "429" not in str(e) or attempt == 5:
                raise
            print("queue busy, retrying in 5 minutes")
            time.sleep(300)
    JOB_FILE.write_text(json.dumps({"name": job["name"], "input_file": file_name}, indent=2))
    print(f"submitted {job['name']} with {sum(1 for _ in REQUESTS.open())} requests ({size / 1e6:.1f} MB)")


def state(job):
    return job.get("metadata", {}).get("state") or job.get("state")


def status():
    job = call(f"{API}/v1beta/{json.loads(JOB_FILE.read_text())['name']}")
    meta = job.get("metadata", job)
    print(state(job), json.dumps(meta.get("batchStats", {})))
    return job


def fetch():
    kept = failed = 0
    total = 0.0
    with OUT.open("w") as out:
        for part in range(PARTS):
            rows, cost = part_results(part)
            total += cost
            for row in rows:
                source = row.get("key", "").split("|", 1)[-1]
                try:
                    text = row["response"]["candidates"][0]["content"]["parts"][-1]["text"]
                    examples = json.loads(text)["examples"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    failed += 1
                    continue
                for ex in examples:
                    out.write(json.dumps({"source": source, "model": MODEL, **ex}) + "\n")
                    kept += 1
    print(f"{kept} examples, {failed} failed documents, batch cost ~${total:.2f}")


{"submit": submit, "status": status, "fetch": fetch}[sys.argv[1]]()
