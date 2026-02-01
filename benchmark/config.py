"""Configuration loading utilities for benchmark."""

from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Dictionary containing the parsed YAML configuration

    Raises:
        FileNotFoundError: If the config file does not exist
        yaml.YAMLError: If the YAML is malformed
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
