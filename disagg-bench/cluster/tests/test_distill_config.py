import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill_config import DistillConfig


def test_distill_config_defaults():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=1000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=0,
        num_shards=1,
    )
    assert cfg.teacher_model == "facebook/opt-30b"
    assert cfg.temperature == 4.0
    assert cfg.alpha == 0.5
    assert cfg.shard_size == 1000


def test_distill_config_shard_size():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=2,
        num_shards=5,
    )
    assert cfg.shard_size == 2000


def test_distill_config_shard_range():
    cfg = DistillConfig(
        teacher_model="facebook/opt-30b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=512,
        num_samples=10000,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/logit_cache",
        shard_id=2,
        num_shards=5,
    )
    start, end = cfg.shard_range()
    assert start == 4000
    assert end == 6000


def test_cache_path_format():
    cfg = DistillConfig(
        teacher_model="facebook/opt-1.3b",
        student_model="facebook/opt-1.3b",
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_seq_length=128,
        num_samples=100,
        temperature=4.0,
        alpha=0.5,
        output_dir="/tmp/test_cache",
        shard_id=0,
        num_shards=4,
    )
    assert cfg.cache_path().endswith("shard_0000_of_0004.pt")
