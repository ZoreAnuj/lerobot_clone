from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyContract:
    policy_type: str
    state_key: str
    state_dim: int
    action_key: str
    action_dim: int
    image_keys: tuple[str, ...]
    image_shapes_chw: dict[str, tuple[int, int, int]]

    @property
    def required_observation_keys(self) -> tuple[str, ...]:
        return (self.state_key, *self.image_keys)


def load_policy_contract_from_pretrained(pretrained_model_dir: str | Path) -> PolicyContract:
    """Load the expected observation/action contract from a LeRobot `pretrained_model/` folder."""
    pretrained_model_dir = Path(pretrained_model_dir)
    cfg_path = pretrained_model_dir / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    policy_type = data["type"]
    input_features: dict = data["input_features"]
    output_features: dict = data["output_features"]

    # State
    state_key = "observation.state"
    if state_key not in input_features:
        raise ValueError(f"Expected `{state_key}` in config input_features. Got keys: {list(input_features)}")
    state_shape = tuple(input_features[state_key]["shape"])
    if len(state_shape) != 1:
        raise ValueError(f"Expected `{state_key}` shape [D]. Got {state_shape}")
    state_dim = int(state_shape[0])

    # Action
    action_key = "action"
    if action_key not in output_features:
        raise ValueError(f"Expected `{action_key}` in config output_features. Got keys: {list(output_features)}")
    action_shape = tuple(output_features[action_key]["shape"])
    if len(action_shape) != 1:
        raise ValueError(f"Expected `{action_key}` shape [D]. Got {action_shape}")
    action_dim = int(action_shape[0])

    # Images
    image_keys: list[str] = []
    image_shapes: dict[str, tuple[int, int, int]] = {}
    for k, feat in input_features.items():
        if k.startswith("observation.images."):
            shape = tuple(feat["shape"])
            if len(shape) != 3:
                raise ValueError(f"Expected `{k}` shape [C,H,W]. Got {shape}")
            image_keys.append(k)
            image_shapes[k] = (int(shape[0]), int(shape[1]), int(shape[2]))

    if not image_keys:
        raise ValueError("Expected at least one `observation.images.*` key in config input_features.")

    return PolicyContract(
        policy_type=policy_type,
        state_key=state_key,
        state_dim=state_dim,
        action_key=action_key,
        action_dim=action_dim,
        # Preserve checkpoint order (ACT uses config.image_features ordering for stacking cameras).
        image_keys=tuple(image_keys),
        image_shapes_chw=image_shapes,
    )


def assert_expected_keys_and_shapes(
    contract: PolicyContract,
    *,
    state_dim: int,
    action_dim: int,
    images_chw: dict[str, tuple[int, int, int]],
) -> None:
    if state_dim != contract.state_dim:
        raise ValueError(f"State dim mismatch. Policy expects {contract.state_dim}, backend provides {state_dim}.")
    if action_dim != contract.action_dim:
        raise ValueError(f"Action dim mismatch. Policy expects {contract.action_dim}, backend provides {action_dim}.")

    missing = [k for k in contract.image_keys if k not in images_chw]
    if missing:
        raise ValueError(f"Missing required image keys: {missing}. Backend provides: {sorted(images_chw)}")

    for k in contract.image_keys:
        got = images_chw[k]
        exp = contract.image_shapes_chw[k]
        if got != exp:
            raise ValueError(f"Image shape mismatch for `{k}`. Policy expects {exp} (CHW), backend provides {got}.")


