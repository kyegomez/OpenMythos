"""
Streaming FineWeb-Edu dataloader.

Packs concatenated documents into fixed-length (input, target) pairs of
length seq_len, where target = input shifted by one. Each DataLoader worker
pulls a disjoint shard of the HuggingFace streaming dataset so workers never
overlap.
"""

from typing import Iterator

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset
from transformers import AutoTokenizer


class FineWebEduStream(IterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len: int,
        dataset_name: str,
        dataset_config: str,
        split: str = "train",
        skip_docs: int = 0,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.skip_docs = skip_docs
        self.eos_id = tokenizer.eos_token_id or 0

    def __iter__(self) -> Iterator[tuple]:
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.skip_docs > 0:
            ds = ds.skip(self.skip_docs)
        if num_workers > 1:
            ds = ds.shard(num_shards=num_workers, index=worker_id)

        buffer: list[int] = []
        need = self.seq_len + 1

        for doc in ds:
            text = doc.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.eos_id)
            buffer.extend(ids)

            while len(buffer) >= need:
                chunk = buffer[:need]
                buffer = buffer[need - 1 :]  # keep last token as start of next
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def build_loader(
    tokenizer,
    seq_len: int,
    batch_size: int,
    dataset_name: str,
    dataset_config: str,
    num_workers: int = 2,
    split: str = "train",
    skip_docs: int = 0,
) -> DataLoader:
    ds = FineWebEduStream(
        tokenizer=tokenizer,
        seq_len=seq_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        skip_docs=skip_docs,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def get_tokenizer(name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
