"""
Integration test: distillation pipeline modules compose correctly
without a GPU or real model downloads.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tempfile

from distill_config import DistillConfig
from train_student_distill import (
    distillation_loss,
    load_cache_shard,
    CachedLogitDataset,
    collate_fn,
)


def make_fake_shard(tmpdir: str, n: int = 8, seq: int = 16, vocab: int = 100) -> str:
    cache = {
        "input_ids": torch.randint(0, vocab, (n, seq)),
        "soft_labels": torch.softmax(torch.randn(n, seq, vocab), dim=-1),
        "temperature": 4.0,
        "model": "facebook/opt-1.3b",
        "vocab_size": vocab,
    }
    path = f"{tmpdir}/shard_0000_of_0001.pt"
    torch.save(cache, path)
    return path


def test_full_pipeline_loss_and_backward():
    """Load shard → build dataset → batch → loss → backward."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = make_fake_shard(tmpdir, n=8, seq=16, vocab=100)
        ds = CachedLogitDataset([path])
        assert len(ds) == 8

        batch = collate_fn([ds[0], ds[1]])
        B, S = batch["input_ids"].shape
        V = batch["soft_labels"].shape[-1]

        student_logits = torch.randn(B, S, V, requires_grad=True)
        labels = batch["input_ids"].clone()
        labels[:, :-1] = batch["input_ids"][:, 1:]
        labels[:, -1] = -100

        loss = distillation_loss(
            student_logits,
            batch["soft_labels"].float(),
            labels,
            temperature=4.0,
            alpha=0.5,
        )
        assert not torch.isnan(loss)
        assert loss.item() > 0

        loss.backward()
        assert student_logits.grad is not None
        assert student_logits.grad.norm().item() > 0


def test_config_shard_path_roundtrip():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/cache",
        shard_id=3,
        num_shards=10,
    )
    path = cfg.cache_path()
    assert "shard_0003_of_0010" in path
    start, end = cfg.shard_range()
    assert end - start == cfg.shard_size
