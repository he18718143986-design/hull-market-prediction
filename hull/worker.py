#!/usr/bin/env python3
"""Worker that polls `hull/jobs/` and runs training jobs in subprocesses.

Usage: run this in background (or systemd / screen / tmux):
  python worker.py

Behavior:
 - atomically moves a job file from jobs/ -> processing/
 - creates a run dir under runs/<job_id>/
 - writes cfg.json, runs the appropriate runner (trainer or lgb_runner)
 - enforces sane env (OMP_NUM_THREADS=1 etc) for the subprocess
 - captures stdout/stderr into result.json
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict

ROOT = os.path.abspath(os.path.dirname(__file__))
JOBS = os.path.join(ROOT, 'jobs')
PROCESSING = os.path.join(ROOT, 'processing')
RUNS = os.path.join(ROOT, 'runs')

for d in (JOBS, PROCESSING, RUNS):
    os.makedirs(d, exist_ok=True)


DEFAULT_ENV_OVERRIDES = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_MAX_THREADS': '1',
}


def pick_job_file() -> str | None:
    files = [f for f in os.listdir(JOBS) if f.endswith('.json')]
    files.sort()
    return files[0] if files else None


def run_job(job_file: str, args: argparse.Namespace) -> None:
    queued = os.path.join(JOBS, job_file)
    job_id = os.path.splitext(job_file)[0]
    processing = os.path.join(PROCESSING, job_file)
    try:
        # move atomically
        os.replace(queued, processing)
    except Exception:
        return

    with open(processing, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    cfg = meta.get('cfg', {})
    run_dir = os.path.join(RUNS, job_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg_path = os.path.join(run_dir, 'cfg.json')
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Choose runner: if model_type == 'lightgbm' prefer lgb_runner.py, else trainer.py
    model_type = cfg.get('model_type', '').lower()
    if model_type == 'lightgbm':
        runner = os.path.join(ROOT, 'lgb_runner.py')
    else:
        runner = os.path.join(ROOT, 'trainer.py')

    cmd = [sys.executable, runner, cfg_path]
    env = os.environ.copy()
    env.update(DEFAULT_ENV_OVERRIDES)

    started_at = time.time()
    result = {
        'job_id': job_id,
        'started_at': started_at,
        'cmd': cmd,
    }

    # Stream stdout/stderr to progress.jsonl in run_dir while process runs
    progress_path = os.path.join(run_dir, 'progress.jsonl')
    result_path = os.path.join(run_dir, 'result.json')
    try:
        with open(progress_path, 'a', encoding='utf-8') as progress_f:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # readers for stdout and stderr
            def _drain_stream(stream, name):
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                    entry = {'ts': time.time(), 'stream': name, 'line': line.rstrip('\n')}
                    try:
                        progress_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        progress_f.flush()
                    except Exception:
                        pass
                try:
                    stream.close()
                except Exception:
                    pass

            import threading as _thr
            t_out = _thr.Thread(target=_drain_stream, args=(proc.stdout, 'stdout'), daemon=True)
            t_err = _thr.Thread(target=_drain_stream, args=(proc.stderr, 'stderr'), daemon=True)
            t_out.start()
            t_err.start()

            try:
                rc = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired as e:
                proc.kill()
                rc = None

            # ensure threads finish
            t_out.join(timeout=1.0)
            t_err.join(timeout=1.0)

            result['returncode'] = rc
    except Exception as e:
        result['returncode'] = None
        result['stderr'] = str(e)

    result['finished_at'] = time.time()
    # write result
    try:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # remove processing marker (job consumed)
    try:
        os.remove(processing)
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--poll-interval', type=float, default=1.0)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()

    print('Worker started, polling jobs in', JOBS)
    try:
        while True:
            job = pick_job_file()
            if job:
                print('Picked job', job)
                run_job(job, args)
            else:
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print('Worker interrupted')


if __name__ == '__main__':
    main()
