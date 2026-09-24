"""
Central path configuration for OpenMythos / Sentinel Grid.

All data (checkpoints, HF cache, corpus files) lives next to the repo, on
local disk. Every path here is overridable via an environment variable so a
future move doesn't silently break every script again -- this has already
happened twice: first the mount changed from /Volumes/Claude/openmythos to
/Volumes/home/Claude/openmythos, then the whole project moved off that
(repeatedly-disconnecting network share) to local disk at
~/Desktop/openmythos_data, and finally (2026-09-23) off the Desktop onto the
archive volume at /Users/Shared/mounts/archive/openmythos_data. The venv that
used to sit beside the data was NOT moved -- rebuild it from
requirements-lock.txt with pyenv.
"""
import os

DATA_ROOT = os.environ.get(
    "OPENMYTHOS_DATA_ROOT",
    "/Users/Shared/mounts/archive/openmythos_data",
)

CHECKPOINT_DIR = os.environ.get(
    "OPENMYTHOS_CHECKPOINT_DIR", f"{DATA_ROOT}/checkpoints"
)
HF_CACHE_DIR = os.environ.get(
    "OPENMYTHOS_HF_CACHE_DIR", f"{DATA_ROOT}/hf_cache"
)
CORPUS_DIR = os.environ.get(
    "OPENMYTHOS_CORPUS_DIR", f"{DATA_ROOT}/cyber_corpus"
)

# Import this before any `transformers` import so HF respects the cache dir.
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

if __name__ == "__main__":
    for name in ("DATA_ROOT", "CHECKPOINT_DIR", "HF_CACHE_DIR", "CORPUS_DIR"):
        path = globals()[name]
        exists = "OK" if os.path.exists(path) else "MISSING"
        print(f"{name:20s} {path}  [{exists}]")
