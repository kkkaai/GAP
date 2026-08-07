"""Conditional flow matching models for prosthetic q_goal generation."""

from .dataset import QGoalDataset, QGoalDatasetConfig, build_task_vocab
from .flow import build_flow_matcher
from .model import QGoalCFMModel, QGoalCFMModelConfig
from .normalization import JointNormalizer

__all__ = [
    "JointNormalizer",
    "QGoalCFMModel",
    "QGoalCFMModelConfig",
    "QGoalDataset",
    "QGoalDatasetConfig",
    "build_flow_matcher",
    "build_task_vocab",
]

