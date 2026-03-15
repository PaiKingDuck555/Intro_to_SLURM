"""
Distillation configuration dataclass.
Shared by cache_teacher_logits.py and train_student_distill.py.
"""
import math
import os
from dataclasses import dataclass


@dataclass
class DistillConfig:
    # Models
    teacher_model: str
    student_model: str

    # Data
    dataset_name: str        # HuggingFace dataset, e.g. "wikitext"
    dataset_config: str      # dataset config, e.g. "wikitext-2-raw-v1"
    max_seq_length: int

    # Distillation hyperparameters
    temperature: float       # softmax temperature τ; sweet spot 2–8
    alpha: float             # loss = alpha*CE + (1-alpha)*KL

    # Sharding
    output_dir: str
    num_samples: int
    shard_id: int            # 0-indexed
    num_shards: int

    @property
    def shard_size(self) -> int:
        return math.ceil(self.num_samples / self.num_shards)

    def shard_range(self) -> tuple[int, int]:
        start = self.shard_id * self.shard_size
        end = min(start + self.shard_size, self.num_samples)
        return start, end

    def cache_path(self) -> str:
        return os.path.join(
            self.output_dir,
            f"shard_{self.shard_id:04d}_of_{self.num_shards:04d}.pt",
        )
