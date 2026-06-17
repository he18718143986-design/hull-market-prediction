"""Thin trainer facade module.

The project currently implements the training logic in `train_adjusted_sharpe.py`.
This module provides a small facade so other parts of the codebase can import
`trainer.train_from_config` while we gradually migrate implementation here.
"""
from typing import Any, Callable
import json
import sys

def train_from_config(cfg: dict, progress_hook: Callable[[str], None] = None) -> Any:
    """Delegate to the existing training implementation.
    This keeps a stable import surface while allowing migration later.
    """
    # import lazily so we don't change import semantics of existing modules
    import train_adjusted_sharpe as tas
    return tas.train_from_config(cfg, progress_hook=progress_hook)

def train_main(cfg_path: str) -> int:
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(json.dumps({'error': f'failed to read cfg: {e}'}))
        return 2

    try:
        res = train_from_config(cfg)
        print(json.dumps({'result': res}, default=str, ensure_ascii=False))
        return 0
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(json.dumps({'error': str(e), 'traceback': tb}))
        return 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'missing config path'}))
        sys.exit(2)
    sys.exit(train_main(sys.argv[1]))

__all__ = ['train_from_config', 'train_main']
