"""Private settings live in config.local.json (never committed). See config.example.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_cfg = json.loads((ROOT / "config.local.json").read_text())
SOURCE_DIR = Path(_cfg["source_dir"])
DRIVE_DIR = Path(_cfg["drive_dir"])
INCLUDE_FOLDERS = _cfg["include_folders"]
BLOCKED_TERMS = _cfg["blocked_terms"]
COMPANY_NAME = _cfg["company_name"]
