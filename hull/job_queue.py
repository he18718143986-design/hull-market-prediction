"""Simple file-based job queue for local worker processes.

Directory layout (all under repo `hull/`):
 - jobs/        : new job JSON files (queued)
 - processing/  : job files currently being processed
 - runs/        : output per-job directories with results

This module provides minimal helpers for submitting jobs and checking status.
"""
from __future__ import annotations
import json
import os
import time
import uuid
from typing import Any, Dict, Optional

_ROOT = os.path.abspath(os.path.dirname(__file__))
JOBS_DIR = os.path.join(_ROOT, 'jobs')
PROCESSING_DIR = os.path.join(_ROOT, 'processing')
RUNS_DIR = os.path.join(_ROOT, 'runs')

for d in (JOBS_DIR, PROCESSING_DIR, RUNS_DIR):
    os.makedirs(d, exist_ok=True)


def _timestamp() -> str:
    return time.strftime('%Y%m%dT%H%M%S')


def submit_job(cfg: Dict[str, Any]) -> str:
    """Write a job JSON to jobs/ and return job_id.

    `cfg` is an arbitrary serializable dict. We'll attach metadata.
    """
    job_id = f"job_{_timestamp()}_{uuid.uuid4().hex[:8]}"
    meta = {
        'job_id': job_id,
        'submitted_at': time.time(),
        'cfg': cfg,
    }
    path = os.path.join(JOBS_DIR, f'{job_id}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return job_id


def job_path(job_id: str) -> Dict[str, str]:
    return {
        'queued': os.path.join(JOBS_DIR, f'{job_id}.json'),
        'processing': os.path.join(PROCESSING_DIR, f'{job_id}.json'),
        'run_dir': os.path.join(RUNS_DIR, job_id),
    }


def get_job_status(job_id: str) -> Dict[str, Optional[str]]:
    """Return high-level job status and paths.
    status in ('queued','processing','finished','missing')
    """
    paths = job_path(job_id)
    if os.path.exists(paths['queued']):
        return {'status': 'queued', 'path': paths['queued']}
    if os.path.exists(paths['processing']):
        return {'status': 'processing', 'path': paths['processing']}
    if os.path.exists(os.path.join(paths['run_dir'], 'result.json')):
        return {'status': 'finished', 'path': os.path.join(paths['run_dir'], 'result.json')}
    return {'status': 'missing', 'path': None}


def list_queued_jobs() -> list:
    files = [f for f in os.listdir(JOBS_DIR) if f.endswith('.json')]
    return sorted(files)


__all__ = ['submit_job', 'get_job_status', 'list_queued_jobs', 'job_path']
