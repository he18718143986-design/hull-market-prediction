#!/usr/bin/env python3
"""Small CLI wrapper to run baseline.train_lightgbm_from_config from a subprocess.
Usage: python lgb_runner.py /path/to/config.json
Writes a JSON result to stdout (dict returned by train_lightgbm_from_config).
"""
import sys
import json
import os

# Ensure parent directory (project root) is on sys.path so imports like
# `baseline` work when this script is executed from the `hull/` folder.
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'missing config path'}))
        sys.exit(2)
    cfg_path = sys.argv[1]
    if not os.path.exists(cfg_path):
        print(json.dumps({'error': f'config not found: {cfg_path}'}))
        sys.exit(2)
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(json.dumps({'error': f'failed to read cfg: {e}'}))
        sys.exit(2)

    # Import baseline lazily to avoid interfering with parent process
    try:
        import baseline as bsl
    except Exception as e:
        print(json.dumps({'error': f'failed to import baseline: {e}'}))
        sys.exit(2)

    # progress messages: print lines prefixed so parent can capture
    def progress_hook(msg):
        try:
            print('PROGRESS:' + str(msg), flush=True)
        except Exception:
            pass

    try:
        res = bsl.train_lightgbm_from_config(cfg, progress_hook=progress_hook)
        # ensure JSON-serializable
        try:
            json_res = json.dumps({'result': res}, default=str, ensure_ascii=False)
            print('RESULT:' + json_res, flush=True)
            sys.exit(0)
        except Exception:
            print(json.dumps({'error': 'result not JSON serializable'}))
            sys.exit(1)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(json.dumps({'error': str(e), 'traceback': tb}))
        sys.exit(1)
