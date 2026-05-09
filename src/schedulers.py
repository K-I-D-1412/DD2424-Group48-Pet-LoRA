"""
Gradual unfreezing scheduler for Strategy 2.
"""

from __future__ import annotations
from typing import List
from torch import nn
from src.models import RESNET_BLOCKS_OUTPUT_TO_INPUT, freeze_all_except_last_n_blocks


class GradualUnfreezingScheduler:
    """
    During the training process, the ResNet blocks are gradually unfrozen according to the schedule. 
    Schedule Example:
        [
            {"epoch": 0,  "unfreeze_blocks": 1},  # fc only
            {"epoch": 5,  "unfreeze_blocks": 2},  # + layer4
            {"epoch": 10, "unfreeze_blocks": 3},  # + layer3
            {"epoch": 15, "unfreeze_blocks": 4},  # + layer2
        ] 
    `step(epoch)` returns `True` if the thawing state has changed.
    At this point, the caller must reinitialize the optimizer; otherwise, the newly thawed parameters will not be updated.
    """

    def __init__(self, model: nn.Module, schedule: List[dict]):
        if not schedule:
            raise ValueError("schedule cannot be empty")
        for entry in schedule:
            if "epoch" not in entry or "unfreeze_blocks" not in entry:
                raise ValueError(f"Each item must have an epoch and unfreeze_blocks, got: {entry}")

        self.model = model
        self.schedule = sorted(schedule, key=lambda e: int(e["epoch"]))
        self.current_n = -1  # -1: Ensure that epoch 0 will definitely trigger once

    def step(self, epoch: int) -> bool:
        """
        Find the largest value of 'unfreeze_blocks' in all schedule entries where 'epoch' is less than or equal to the current epoch.
        If it is different from the current state, unfreeze and return True.
        """
        target_n = self.current_n
        for entry in self.schedule:
            if epoch >= int(entry["epoch"]):
                target_n = int(entry["unfreeze_blocks"])

        if target_n != self.current_n:
            freeze_all_except_last_n_blocks(self.model, n=target_n)
            self.current_n = target_n
            return True
        return False

    def describe(self) -> str:
        lines = []
        for entry in self.schedule:
            n = int(entry["unfreeze_blocks"])
            blocks = RESNET_BLOCKS_OUTPUT_TO_INPUT[:n]
            lines.append(f"  epoch {int(entry['epoch']):>3d}: unfreeze {blocks}")
        return "\n".join(lines)