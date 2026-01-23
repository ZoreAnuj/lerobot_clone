from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


def _parse_feature_map(raw: dict[str, Any]) -> dict[str, PolicyFeature]:
    return {
        k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(int(x) for x in v["shape"]))
        for k, v in raw.items()
    }


def _parse_norm_map(raw: dict[str, Any]) -> dict[str, NormalizationMode]:
    return {k: NormalizationMode(v) for k, v in raw.items()}


def load_act_config_from_pretrained_compat(
    pretrained_model_dir: str | Path,
    *,
    device: str | None = None,
) -> ACTConfig:
    """Load `ACTConfig` from a checkpoint even if extra/renamed fields exist.

    Some checkpoints were produced with slightly different config schemas; this loader:
    - drops unknown fields (forward/backward compatibility)
    - handles a couple common renames
    - converts `input_features` / `output_features` into `PolicyFeature` objects
    - converts `normalization_mapping` values into `NormalizationMode` enums
    """
    pretrained_model_dir = Path(pretrained_model_dir)
    data = json.loads((pretrained_model_dir / "config.json").read_text(encoding="utf-8"))

    # Drop draccus choice tag
    data.pop("type", None)

    # Renames seen in some checkpoints
    if "use_separate_backbone_per_camera" in data and "separate_backbones_per_camera" not in data:
        data["separate_backbones_per_camera"] = data.pop("use_separate_backbone_per_camera")

    # Parse structured fields
    if "input_features" in data:
        data["input_features"] = _parse_feature_map(data["input_features"])
    if "output_features" in data:
        data["output_features"] = _parse_feature_map(data["output_features"])
    if "normalization_mapping" in data:
        data["normalization_mapping"] = _parse_norm_map(data["normalization_mapping"])

    allowed = {f.name for f in fields(ACTConfig)}
    filtered = {k: v for k, v in data.items() if k in allowed}

    if device is not None:
        filtered["device"] = device

    # Helpful for downstream code (e.g. logging/debugging)
    filtered["pretrained_path"] = Path(pretrained_model_dir)

    return ACTConfig(**filtered)


def load_diffusion_config_from_pretrained_compat(
    pretrained_model_dir: str | Path,
    *,
    device: str | None = None,
) -> DiffusionConfig:
    """Load `DiffusionConfig` from a checkpoint even if extra/renamed fields exist.

    This mirrors `load_act_config_from_pretrained_compat`:
    - drops unknown fields (forward/backward compatibility)
    - converts `input_features` / `output_features` into `PolicyFeature` objects
    - converts `normalization_mapping` values into `NormalizationMode` enums
    """
    pretrained_model_dir = Path(pretrained_model_dir)
    data = json.loads((pretrained_model_dir / "config.json").read_text(encoding="utf-8"))

    # Drop draccus choice tag
    data.pop("type", None)

    # Parse structured fields
    if "input_features" in data:
        data["input_features"] = _parse_feature_map(data["input_features"])
    if "output_features" in data:
        data["output_features"] = _parse_feature_map(data["output_features"])
    if "normalization_mapping" in data:
        data["normalization_mapping"] = _parse_norm_map(data["normalization_mapping"])

    # A few fields are tuples in the dataclass but may be serialized as lists.
    if "down_dims" in data and isinstance(data["down_dims"], list):
        data["down_dims"] = tuple(int(x) for x in data["down_dims"])
    if "crop_shape" in data and isinstance(data["crop_shape"], list) and len(data["crop_shape"]) == 2:
        data["crop_shape"] = (int(data["crop_shape"][0]), int(data["crop_shape"][1]))

    allowed = {f.name for f in fields(DiffusionConfig)}
    filtered = {k: v for k, v in data.items() if k in allowed}

    if device is not None:
        filtered["device"] = device

    # Helpful for downstream code (e.g. logging/debugging)
    filtered["pretrained_path"] = Path(pretrained_model_dir)

    return DiffusionConfig(**filtered)


