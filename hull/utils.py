"""Utility helpers used across modules.
This file is intentionally lightweight: add common IO and path helpers
so other modules can gradually import from here during refactor.
"""
import json
import os
from typing import Any

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

__all__ = [
    'ensure_dir',
    'read_json',
    'write_json',
    'safe_remove',
]
