"""Quick paid check of prompt + schema on two documents before the full batch."""
import json, re, sys
sys.argv = ["batch_gemini.py", "none"]
src = open(__file__.replace("trial_two_docs.py", "batch_gemini.py")).read().split('{"submit": submit')[0]
exec(compile(src, __file__.replace("trial_two_docs.py", "batch_gemini.py"), "exec"))
prompt = (ROOT / "scripts" / "prompt.md").read_text()
docs = [p for p in (ROOT / "data" / "doc_list.txt").read_text().split("\n") if p]
for path in [docs[5], docs[700]]:
    body = {"contents": [{"parts": [{"text": prompt + Path(path).read_text(errors="ignore")[:MAX_DOC_CHARS]}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA}}
    r = call(f"{API}/v1beta/models/{MODEL}:generateContent", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    ex = json.loads(r["candidates"][0]["content"]["parts"][-1]["text"])["examples"]
    print(Path(path).name[:50], len(ex), "examples |", [e["type"] for e in ex])
    print("  reply openers:", [" ".join(e["reply"].split()[:3]) for e in ex])
    print("  request openers:", [" ".join(e["request"].split()[:3]) for e in ex])
    print("  percents:", sum(len(re.findall(r"\d+%", e["reply"])) for e in ex), "| long dashes:", sum(e["reply"].count("—") for e in ex))
