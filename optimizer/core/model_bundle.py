from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass
class ModelBundle:
    model: nn.Module
    provider: str

    original_checkpoint: Any = None

    ignored_pruning_layers: list[nn.Module] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )