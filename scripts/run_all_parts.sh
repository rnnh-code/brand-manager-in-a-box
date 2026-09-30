#!/bin/zsh -i
# Wait for each batch part to finish, then start the next one. Stops on any failure.
cd "$(dirname "$0")/.."
for part in 0 1 2; do
  [ -f "data/batch_job_$part.json" ] || python3 scripts/batch_gemini.py submit $part || exit 1
  while true; do
    s=$(python3 scripts/batch_gemini.py status $part)
    echo "$(date +%H:%M) part $part: $s"
    case "$s" in
      *SUCCEEDED*) break ;;
      *FAILED*|*CANCELLED*|*EXPIRED*) echo "part $part stopped: $s"; exit 1 ;;
    esac
    sleep 300
  done
done
python3 scripts/batch_gemini.py fetch
