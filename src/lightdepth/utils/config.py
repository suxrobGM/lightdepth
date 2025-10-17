# CS 7180 Advanced Perception
# Author: Sukhrobbek Ilyosbekov
# Date: 2025-10-16
# Description: Configuration management for LightDepth model

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    """Simple training configuration."""

    # Data
    data_root: str = "data/nyu"
    img_height: int = 480
    img_width: int = 640

    # Training
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4

    # System
    num_workers: int = 4
    device: str = "cuda"

    # Resuming
    resume_from: str | None = None  # Path to checkpoint to resume from

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object with loaded settings
        """

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls(**config_dict)

    def save_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML file should be saved
        """

        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
