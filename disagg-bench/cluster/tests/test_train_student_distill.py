import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tempfile


def test_distillation_loss_pure_ce():
    """alpha=1.0 → loss equals cross-entropy."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=1.0)
    expected = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, V), labels.reshape(-1)
    )
    assert abs(loss.item() - expected.item()) < 1e-4


def test_distillation_loss_pure_kl():
    """alpha=0.0 → loss is non-negative KL divergence."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.0)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_distillation_loss_combined_not_nan():
    """Combined loss with alpha=0.5 is finite and positive."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 16, 100
    student_logits = torch.randn(B, S, V)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.5)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_distillation_loss_gradients_flow():
    """Backward pass produces non-zero gradients."""
    from train_student_distill import distillation_loss

    B, S, V = 2, 8, 50
    student_logits = torch.randn(B, S, V, requires_grad=True)
    soft_labels = torch.softmax(torch.randn(B, S, V), dim=-1)
    labels = torch.randint(0, V, (B, S))

    loss = distillation_loss(student_logits, soft_labels, labels, temperature=4.0, alpha=0.5)
    loss.backward()
    assert student_logits.grad is not None
    assert student_logits.grad.norm().item() > 0.0


def test_load_cache_shard_keys():
    """load_cache_shard returns dict with required keys."""
    from train_student_distill import load_cache_shard

    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {
            "input_ids": torch.randint(0, 1000, (10, 128)),
            "soft_labels": torch.randn(10, 128, 50272),
            "temperature": 4.0,
            "model": "facebook/opt-30b",
            "vocab_size": 50272,
        }
        path = f"{tmpdir}/shard_0000_of_0001.pt"
        torch.save(fake, path)

        data = load_cache_shard(path)
        assert "input_ids" in data
        assert "soft_labels" in data
        assert data["input_ids"].shape == (10, 128)


def test_cached_logit_dataset_length():
    """CachedLogitDataset returns correct length from shard."""
    from train_student_distill import CachedLogitDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {
            "input_ids": torch.randint(0, 100, (12, 32)),
            "soft_labels": torch.softmax(torch.randn(12, 32, 100), dim=-1),
            "temperature": 4.0,
            "model": "facebook/opt-1.3b",
            "vocab_size": 100,
        }
        path = f"{tmpdir}/shard_0000_of_0001.pt"
        torch.save(fake, path)

        ds = CachedLogitDataset([path])
        assert len(ds) == 12
        sample = ds[0]
        assert "input_ids" in sample
        assert "soft_labels" in sample
