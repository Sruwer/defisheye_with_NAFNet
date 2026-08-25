from __future__ import annotations
import json
from pathlib import Path

def load_config(path):
    try: import yaml
    except ImportError as exc: raise RuntimeError("PyYAML is required; install with: pip install -e .") from exc
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

