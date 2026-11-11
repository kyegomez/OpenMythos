"""
Regression test for the training script's data sharding.

On 4 GPUs with 4 DataLoader workers each, the script asked the streaming
dataset for 16 file-level shards. The subset had fewer files, and the workers
with the highest indices died with IndexError before the second step.
"""

import importlib.util
import json
from pathlib import Path

import pytest

datasets = pytest.importorskip("datasets")
pytest.importorskip("loguru")

SCRIPT = Path(__file__).resolve().parent.parent / "training" / "3b_fine_web_edu.py"
N_FILES = 3
DOCS_PER_FILE = 10


@pytest.fixture(scope="module")
def shard_stream():
    spec = importlib.util.spec_from_file_location("train_3b", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.shard_stream


@pytest.fixture()
def data_files(tmp_path):
    files = []
    for f in range(N_FILES):
        path = tmp_path / f"part{f}.jsonl"
        with open(path, "w") as fh:
            for d in range(DOCS_PER_FILE):
                fh.write(json.dumps({"text": f"doc-{f}-{d}"}) + "\n")
        files.append(str(path))
    return files


def load(data_files):
    return datasets.load_dataset("json", data_files=data_files, split="train", streaming=True)


@pytest.mark.parametrize("total_shards", [2, N_FILES, 8, 16])
def test_shards_are_disjoint_and_cover_everything(shard_stream, data_files, total_shards):
    seen = []
    for index in range(total_shards):
        seen.extend(s["text"] for s in shard_stream(load(data_files), total_shards, index))
    assert len(seen) == N_FILES * DOCS_PER_FILE
    assert len(set(seen)) == N_FILES * DOCS_PER_FILE


def test_plain_file_sharding_fails_with_more_consumers_than_files(data_files):
    """The failure the fallback exists for; if this stops raising, the
    fallback is no longer needed."""
    with pytest.raises(IndexError):
        list(load(data_files).shard(num_shards=16, index=15))
