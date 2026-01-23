#!/usr/bin/env python

from __future__ import annotations

import sys
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import packaging.version
import safetensors
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_model as load_model_as_safetensor

# Allow running as a plain script from repo root without having to set PYTHONPATH.
# File is: lerobot/examples/luckyengine/tune_act_piper_room_params.py
# We want to add: lerobot/src
_src_dir = Path(__file__).resolve().parents[2] / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lerobot.luckyengine.config_compat import load_act_config_from_pretrained_compat
from lerobot.luckyengine.contracts import (
    assert_expected_keys_and_shapes,
    load_policy_contract_from_pretrained,
)
from lerobot.luckyengine.hazel_backend import HazelGrpcBackend
from lerobot.luckyengine.mock_backend import MockPiperRoomBackend
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# Debug: set True to print `MAP | ...` lines and assert schema lengths once per episode start.
_ENABLE_MAPPING_DEBUG = False


def _hwc_uint8_to_chw_float01(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) uint8 RGB -> (3,H,W) float32 in [0,1]."""
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC with 3 channels, got {img.shape}")
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _percentile_ms(values_s: list[float], p: float) -> float:
    if not values_s:
        return float("nan")
    return float(np.percentile(np.asarray(values_s, dtype=np.float64) * 1000.0, p))


def _mean_ms(values_s: list[float]) -> float:
    if not values_s:
        return float("nan")
    return float(np.mean(np.asarray(values_s, dtype=np.float64) * 1000.0))


def _parse_csv_ints(s: str) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_floats_or_none(s: str) -> list[float | None]:
    if not s:
        return []
    out: list[float | None] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(float(part))
    return out


def _count_transitions(gripper: list[float], *, threshold: float = 0.0175) -> int:
    """Count open/close transitions by thresholding the gripper signal."""
    if not gripper:
        return 0
    prev = gripper[0] > threshold
    n = 0
    for v in gripper[1:]:
        cur = v > threshold
        if cur != prev:
            n += 1
            prev = cur
    return n


def _backend_image_hw(backend) -> tuple[int, int]:
    """Return (H,W) for whichever backend we are using."""
    if hasattr(backend, "height") and hasattr(backend, "width"):
        return int(backend.height), int(backend.width)
    if hasattr(backend, "image_height") and hasattr(backend, "image_width"):
        return int(backend.image_height), int(backend.image_width)
    raise AttributeError("Backend does not expose (height,width) or (image_height,image_width)")


def _backend_reset(backend, *, camera_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Return (state7, images_by_camera_hwc_uint8, timestamps_s)."""
    if isinstance(backend, MockPiperRoomBackend):
        step = backend.reset()
        return step.state, step.images_hwc_uint8, step.timestamps_s
    if isinstance(backend, HazelGrpcBackend):
        return backend.reset(camera_names=camera_names)
    raise TypeError(f"Unsupported backend type: {type(backend)}")


def _backend_step(
    backend, *, action7: np.ndarray, camera_names: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Return (state7, images_by_camera_hwc_uint8, timestamps_s)."""
    if isinstance(backend, MockPiperRoomBackend):
        step = backend.step(action7)
        return step.state, step.images_hwc_uint8, step.timestamps_s
    if isinstance(backend, HazelGrpcBackend):
        return backend.step(action7=action7, camera_names=camera_names)
    raise TypeError(f"Unsupported backend type: {type(backend)}")


def _stable_unique(items: list[str]) -> list[str]:
    """Return unique items preserving the first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _log_and_assert_stream_mappings(*, policy, contract, backend, cam_names: list[str]) -> None:
    """One-time, high-signal logging + assertions for camera + joint mapping."""
    # Cameras: contract -> Hazel streams
    backend_cams = backend.list_cameras()
    missing_cams = [c for c in cam_names if c not in backend_cams]
    if missing_cams:
        raise ValueError(
            f"Backend is missing required cameras {missing_cams}. "
            f"Requested={cam_names}, available={backend_cams}"
        )

    # Cameras: policy order (ACT uses config.image_features list)
    policy_cam_keys = list(getattr(getattr(policy, "config", None), "image_features", []) or [])
    contract_cam_keys = list(contract.image_keys)

    # Joints: schema names (Hazel)
    obs_names = backend.observation_names() if hasattr(backend, "observation_names") else []
    act_names = backend.action_names() if hasattr(backend, "action_names") else []

    # Derive gripper index from schema if possible (tolerate joint7 naming).
    gripper_obs_idx = None
    gripper_act_idx = None
    if obs_names:
        if "gripper" in obs_names:
            gripper_obs_idx = obs_names.index("gripper")
        elif "joint7" in obs_names:
            gripper_obs_idx = obs_names.index("joint7")
    if act_names:
        if "gripper" in act_names:
            gripper_act_idx = act_names.index("gripper")
        elif "joint7" in act_names:
            gripper_act_idx = act_names.index("joint7")

    # Keep logs one-line-ish but readable.
    if _ENABLE_MAPPING_DEBUG:
        print(
            f"{_now()} | MAP | cams_req={cam_names} | cams_avail={backend_cams} | "  # noqa: T201
            f"policy_img_keys={policy_cam_keys} | contract_img_keys={contract_cam_keys} | "
            f"obs_names={obs_names} | act_names={act_names} | gripper_idx(obs,act)={(gripper_obs_idx, gripper_act_idx)}"
        )

    # Assert the basic joint contract if schema is available.
    if obs_names:
        if len(obs_names) != int(backend.state_dim()):
            raise ValueError(
                f"Agent schema observation_names length mismatch: {len(obs_names)} vs state_dim={backend.state_dim()}"
            )
    if act_names:
        if len(act_names) != int(backend.action_dim()):
            raise ValueError(
                f"Agent schema action_names length mismatch: {len(act_names)} vs action_dim={backend.action_dim()}"
            )


def iter_checkpoints(checkpoints_root: Path) -> list[Path]:
    ckpts: list[Path] = []
    for p in checkpoints_root.iterdir():
        try:
            if not p.is_dir():
                continue
        except OSError:
            # On Windows, some entries (e.g. special symlinks like `last`) can fail stat().
            continue
        pretrained = p / "pretrained_model"
        if (pretrained / "config.json").exists():
            ckpts.append(pretrained)
    return sorted(ckpts, key=lambda x: x.parent.name)


def _pick_pretrained_dir(*, checkpoints_root: Path, checkpoint: str | None) -> Path:
    ckpts = iter_checkpoints(checkpoints_root)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {checkpoints_root}")

    if checkpoint is None or str(checkpoint).strip() == "":
        return ckpts[-1]  # latest (sorted by folder name)

    p = Path(checkpoint)
    if p.exists():
        # Accept either .../pretrained_model or .../<ckpt>/pretrained_model
        if p.is_dir() and (p / "config.json").exists():
            return p
        if p.is_dir() and (p / "pretrained_model" / "config.json").exists():
            return p / "pretrained_model"

    # Treat as checkpoint folder name under checkpoints_root
    candidate = checkpoints_root / str(checkpoint) / "pretrained_model"
    if (candidate / "config.json").exists():
        return candidate

    # Fallback: look for exact parent match
    for d in ckpts:
        if d.parent.name == str(checkpoint):
            return d
    raise FileNotFoundError(
        f"Could not resolve checkpoint={checkpoint}. "
        f"Expected a path to pretrained_model/ or a folder name under {checkpoints_root}."
    )


def _load_home_from_dataset(dataset_dir: Path, *, episode: int = 0, frame: int = 0) -> np.ndarray:
    """Loads dataset-home pose from a local LeRobot dataset directory as `observation.state`."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    ds = LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir, download_videos=False)
    from_idx = ds.meta.episodes["dataset_from_index"][episode]
    idx = int(from_idx) + int(frame)
    # IMPORTANT: avoid `ds[idx]` because it triggers video decoding (torchcodec/ffmpeg).
    row = ds.hf_dataset[int(idx)]
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
    if state.size != 7:
        raise ValueError(f"Expected dataset observation.state to be 7D, got {state.size} at {dataset_dir} ep={episode}")
    return state


def _read_dataset_fps(dataset_dir: Path) -> float | None:
    info = dataset_dir / "meta" / "info.json"
    if not info.exists():
        return None
    try:
        obj = json.loads(info.read_text(encoding="utf-8"))
        fps = obj.get("fps", None)
        return float(fps) if fps is not None else None
    except Exception:
        return None


def _drive_to_home(
    *,
    backend,
    camera_names: list[str],
    home_action7: np.ndarray,
    settle_steps: int,
    settle_sleep_s: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Repeatedly send absolute joint targets until the robot settles near dataset-home."""
    home_action7 = np.asarray(home_action7, dtype=np.float32).reshape(-1)
    if home_action7.size != backend.action_dim():
        raise ValueError(f"Home action dim mismatch: expected {backend.action_dim()}, got {home_action7.size}")

    state = None
    images = None
    for _ in range(int(settle_steps)):
        state, images, _ = _backend_step(backend, action7=home_action7, camera_names=camera_names)
        if settle_sleep_s > 0:
            time.sleep(float(settle_sleep_s))
    assert state is not None and images is not None
    return state, images


@dataclass
class WeightLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]


def load_local_policy_with_report(*, policy_class, cfg, pretrained_dir: Path):
    """Load policy weights from local `pretrained_model/` and return missing/unexpected keys."""
    model_file = pretrained_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.exists():
        raise FileNotFoundError(f"Missing {SAFETENSORS_SINGLE_FILE} under {pretrained_dir}")

    policy = policy_class(cfg)

    kwargs: dict[str, object] = {"strict": False}
    if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
        kwargs["device"] = cfg.device

    missing_keys, unexpected_keys = load_model_as_safetensor(policy, str(model_file), **kwargs)
    policy.to(cfg.device)
    policy.eval()

    report = WeightLoadReport(missing_keys=sorted(set(missing_keys)), unexpected_keys=sorted(set(unexpected_keys)))
    return policy, report


@dataclass(frozen=True)
class Variant:
    name: str
    n_action_steps: int
    temporal_ensemble_coeff: float | None
    torch_compile: bool


@dataclass
class EpisodeResult:
    checkpoint: str
    variant: str
    try_idx: int
    steps: int
    wall_s: float
    achieved_hz: float
    infer_mean_ms: float
    infer_p95_ms: float
    backend_mean_ms: float
    backend_p95_ms: float
    cam_skew_p95_ms: float
    gripper_transitions: int
    gripper_min: float
    gripper_max: float
    picked_step: int | None
    placed_step: int | None
    success_step: int | None
    failure_mode: str | None
    error: str | None


def _validate_variant(variant: Variant, *, chunk_size: int) -> None:
    if variant.temporal_ensemble_coeff is not None and variant.n_action_steps != 1:
        raise ValueError("temporal_ensemble_coeff requires n_action_steps=1")
    if variant.n_action_steps <= 0:
        raise ValueError("n_action_steps must be >0")
    if variant.n_action_steps > int(chunk_size):
        raise ValueError(f"n_action_steps={variant.n_action_steps} must be <= chunk_size={chunk_size}")


@dataclass(frozen=True)
class SuccessConfig:
    enabled: bool
    block_name_substr: str
    dropbox_name_substr: str
    block_exact_name: str | None
    dropbox_exact_name: str | None
    lift_dz_m: float
    drop_xy_tol_m: float
    drop_z_tol_m: float
    confirm_steps: int
    scene_poll_every: int


@dataclass(frozen=True)
class AssistConfig:
    """Best-effort assistance using SceneService ground truth.

    This does NOT improve the learned policy; it overrides the gripper action (and optionally freezes the arm)
    when the end-effector is close to the block / when the block is at the dropbox.
    """

    enabled: bool
    gripper_name_substr: str
    gripper_exact_name: str | None
    pick_xy_tol_m: float
    place_xy_tol_m: float
    close_value: float
    open_value: float
    hold_steps: int
    freeze_arm: bool


def _dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def run_episode(
    *,
    policy,
    preprocessor,
    postprocessor,
    contract,
    backend,
    episode_steps: int,
    task: str | None,
    home_action7: np.ndarray | None,
    home_settle_steps: int,
    home_settle_sleep_s: float,
    control_hz: float | None,
    success_cfg: SuccessConfig,
    assist_cfg: AssistConfig | None = None,
    trace_jsonl_path: Path | None = None,
) -> tuple[EpisodeResult, dict[str, object]]:
    cam_names = _stable_unique([k.split(".")[-1] for k in contract.image_keys])
    state0, images0, ts0 = _backend_reset(backend, camera_names=cam_names)

    if home_action7 is not None:
        state0, images0 = _drive_to_home(
            backend=backend,
            camera_names=cam_names,
            home_action7=home_action7,
            settle_steps=home_settle_steps,
            settle_sleep_s=home_settle_sleep_s,
        )

    policy.reset()

    # Contract checks (backend side, with concrete shapes).
    h, w = _backend_image_hw(backend)
    images_chw = {f"observation.images.{cam}": (3, h, w) for cam in cam_names}
    assert_expected_keys_and_shapes(
        contract,
        state_dim=backend.state_dim(),
        action_dim=backend.action_dim(),
        images_chw=images_chw,
    )

    # Optional mapping debug (helps catch swapped cameras / joint order issues).
    if _ENABLE_MAPPING_DEBUG:
        _log_and_assert_stream_mappings(policy=policy, contract=contract, backend=backend, cam_names=cam_names)

    dt = (1.0 / float(control_hz)) if control_hz is not None and control_hz > 0 else None

    trace_f = None
    if trace_jsonl_path is not None:
        trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        trace_f = trace_jsonl_path.open("w", encoding="utf-8")
        trace_f.write(
            json.dumps(
                {
                    "event": "meta",
                    "chunk_size": int(getattr(getattr(policy, "config", None), "chunk_size", -1) or -1),
                    "n_action_steps": int(getattr(getattr(policy, "config", None), "n_action_steps", -1) or -1),
                    "temporal_ensemble_coeff": getattr(getattr(policy, "config", None), "temporal_ensemble_coeff", None),
                    "cam_names": cam_names,
                    "state_key": contract.state_key,
                    "image_keys": list(contract.image_keys),
                }
            )
            + "\n"
        )

    gripper_trace: list[float] = [float(state0[-1])]
    infer_s: list[float] = []
    backend_s: list[float] = []
    cam_skew_ms: list[float] = []

    # ---- Success tracking (Hazel SceneService) --------------------------------
    block_id = None
    drop_id = None
    gripper_entity_id = None
    block_z0 = None
    drop_pos0 = None
    lift_count = 0
    place_count = 0
    picked_step: int | None = None
    placed_step: int | None = None
    success_step: int | None = None
    failure_mode: str | None = None

    if success_cfg.enabled and isinstance(backend, HazelGrpcBackend):
        block_id = backend.find_entity_id(
            name_substr=success_cfg.block_name_substr, prefer_exact=success_cfg.block_exact_name
        )
        drop_id = backend.find_entity_id(
            name_substr=success_cfg.dropbox_name_substr, prefer_exact=success_cfg.dropbox_exact_name
        )
        if block_id is not None:
            bp = backend.get_entity_pos(int(block_id))
            if bp is not None:
                block_z0 = float(bp[2])
        if drop_id is not None:
            dp = backend.get_entity_pos(int(drop_id))
            if dp is not None:
                drop_pos0 = (float(dp[0]), float(dp[1]), float(dp[2]))

        # Optional "assist" mode: use ground-truth transforms to help the gripper close/open.
        if assist_cfg is not None and bool(assist_cfg.enabled):
            gripper_entity_id = backend.find_entity_id(
                name_substr=str(assist_cfg.gripper_name_substr),
                prefer_exact=assist_cfg.gripper_exact_name,
            )
            if gripper_entity_id is None:
                # Heuristic fallback: try a few common substrings.
                for needle in ("gripper", "hand", "eef", "ee", "tcp", "tool", "palm"):
                    gripper_entity_id = backend.find_entity_id(name_substr=needle, prefer_exact=None)
                    if gripper_entity_id is not None:
                        break
            if gripper_entity_id is None:
                print(  # noqa: T201 (example script)
                    f"{_now()} | WARN | assist enabled but could not find a gripper entity in SceneService. "
                    f"Try adjusting --assist_gripper_name_substr / --assist_gripper_exact_name."
                )

    state = state0
    images = images0
    ts = ts0

    chunk_id = 0
    chunk_step = 0

    # Assist state (countdown holds to stabilize close/open).
    assist_close_left = 0
    assist_open_left = 0

    # Determine gripper action index from schema when available, else assume last dimension.
    gripper_act_idx = None
    if hasattr(backend, "action_names"):
        try:
            act_names = list(backend.action_names() or [])
        except Exception:  # noqa: BLE001
            act_names = []
        if act_names:
            if "gripper" in act_names:
                gripper_act_idx = int(act_names.index("gripper"))
            elif "joint7" in act_names:
                gripper_act_idx = int(act_names.index("joint7"))
    if gripper_act_idx is None:
        gripper_act_idx = int(backend.action_dim()) - 1

    t_wall0 = time.perf_counter()
    try:
        for step_i in range(int(episode_steps)):
            t_step0 = time.perf_counter()

            # If this policy uses action chunking, `select_action()` will internally pop from a queue for
            # n_action_steps-1 steps, and only recompute a new chunk when the queue is empty. This is the
            # source of "repeating" action chunks in logs when n_action_steps > 1.
            action_queue = getattr(policy, "_action_queue", None)
            queue_empty_before = bool(action_queue is not None and len(action_queue) == 0)
            if queue_empty_before:
                chunk_id += 1
                chunk_step = 0
            else:
                chunk_step += 1

            state_before = state

            batch: dict[str, object] = {}
            batch[contract.state_key] = torch.from_numpy(state).to(dtype=torch.float32)

            for k in contract.image_keys:
                cam_name = k.split(".")[-1]
                img_hwc = images[cam_name]
                img_chw = _hwc_uint8_to_chw_float01(img_hwc)
                batch[k] = torch.from_numpy(img_chw).to(dtype=torch.float32)

            if task is not None:
                batch["task"] = task

            obs_t = preprocessor(batch)
            t_inf0 = time.perf_counter()
            with torch.no_grad():
                action_norm = policy.select_action(obs_t)
                action = postprocessor(action_norm)
            t_inf1 = time.perf_counter()

            action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # ---- Assist overrides ----------------------------------------------------
            # A pragmatic "make it work" mode: use SceneService transforms to trigger gripper close/open.
            if (
                assist_cfg is not None
                and bool(assist_cfg.enabled)
                and isinstance(backend, HazelGrpcBackend)
                and block_id is not None
            ):
                # Trigger close when end-effector is near the block.
                if picked_step is None and gripper_entity_id is not None:
                    gp = backend.get_entity_pos(int(gripper_entity_id))
                    bp = backend.get_entity_pos(int(block_id))
                    if gp is not None and bp is not None:
                        if _dist_xy(gp, bp) <= float(assist_cfg.pick_xy_tol_m):
                            assist_close_left = max(int(assist_cfg.hold_steps), assist_close_left)

                # Trigger open when block reaches dropbox vicinity (use block pos vs dropbox pos).
                if picked_step is not None and drop_pos0 is not None:
                    bp = backend.get_entity_pos(int(block_id))
                    if bp is not None:
                        xy_ok = _dist_xy(bp, drop_pos0) <= float(assist_cfg.place_xy_tol_m)
                        z_ok = float(bp[2]) <= float(drop_pos0[2]) + float(success_cfg.drop_z_tol_m)
                        if xy_ok and z_ok:
                            assist_open_left = max(int(assist_cfg.hold_steps), assist_open_left)

                # Apply holds.
                if assist_close_left > 0:
                    action_np[int(gripper_act_idx)] = float(assist_cfg.close_value)
                    if bool(assist_cfg.freeze_arm) and state_before is not None and state_before.shape == action_np.shape:
                        for j in range(int(action_np.size)):
                            if j != int(gripper_act_idx):
                                action_np[j] = float(state_before[j])
                    assist_close_left -= 1
                elif assist_open_left > 0:
                    action_np[int(gripper_act_idx)] = float(assist_cfg.open_value)
                    if bool(assist_cfg.freeze_arm) and state_before is not None and state_before.shape == action_np.shape:
                        for j in range(int(action_np.size)):
                            if j != int(gripper_act_idx):
                                action_np[j] = float(state_before[j])
                    assist_open_left -= 1

            # If we're at a chunk boundary, dump the whole queued chunk in action-space (action7 units).
            # This makes it easy to spot repeated chunks and subtle joint drift (e.g. Joint 1).
            if trace_f is not None and queue_empty_before and int(getattr(policy.config, "n_action_steps", 1)) > 1:
                planned_action7: list[list[float]] = [action_np.tolist()]
                for a_norm in list(getattr(policy, "_action_queue", [])):
                    a = postprocessor(a_norm)
                    a_np = a.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    planned_action7.append(a_np.tolist())
                trace_f.write(
                    json.dumps(
                        {
                            "event": "chunk_plan",
                            "step": int(step_i),
                            "chunk_id": int(chunk_id),
                            "n_action_steps": int(getattr(policy.config, "n_action_steps", len(planned_action7))),
                            "actions_action7": planned_action7,
                        }
                    )
                    + "\n"
                )

            t_b0 = time.perf_counter()
            state, images, ts = _backend_step(backend, action7=action_np, camera_names=cam_names)
            t_b1 = time.perf_counter()

            infer_s.append(t_inf1 - t_inf0)
            backend_s.append(t_b1 - t_b0)

            cam_ts = [ts.get(cam) for cam in cam_names if cam in ts]
            if len(cam_ts) >= 2:
                cam_skew_ms.append(float(max(cam_ts) - min(cam_ts)) * 1000.0)

            gripper_trace.append(float(state[-1]))

            if trace_f is not None:
                trace_f.write(
                    json.dumps(
                        {
                            "event": "step",
                            "step": int(step_i),
                            "chunk_id": int(chunk_id),
                            "chunk_step": int(chunk_step),
                            "queue_empty_before": bool(queue_empty_before),
                            "state": np.asarray(state_before, dtype=np.float32).reshape(-1).tolist(),
                            "action_action7": action_np.tolist(),
                            "next_state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
                            "timestamps_s": {k: float(v) for k, v in (ts or {}).items()},
                        }
                    )
                    + "\n"
                )

            # Success checks (poll scene at a lower rate)
            if (
                success_cfg.enabled
                and isinstance(backend, HazelGrpcBackend)
                and block_id is not None
                and drop_pos0 is not None
                and block_z0 is not None
                and (step_i % max(1, int(success_cfg.scene_poll_every)) == 0)
            ):
                bp = backend.get_entity_pos(int(block_id))
                if bp is not None:
                    if float(bp[2]) > float(block_z0) + float(success_cfg.lift_dz_m):
                        lift_count += 1
                    else:
                        lift_count = 0
                    if picked_step is None and lift_count >= int(success_cfg.confirm_steps):
                        picked_step = int(step_i)

                    if picked_step is not None:
                        xy_ok = _dist_xy(bp, drop_pos0) <= float(success_cfg.drop_xy_tol_m)
                        z_ok = float(bp[2]) <= float(drop_pos0[2]) + float(success_cfg.drop_z_tol_m)
                        if xy_ok and z_ok:
                            place_count += 1
                        else:
                            place_count = 0
                        if placed_step is None and place_count >= int(success_cfg.confirm_steps):
                            placed_step = int(step_i)
                            success_step = int(step_i)

            if success_step is not None:
                # Early stop on success to maximize throughput across checkpoints.
                break

            if dt is not None:
                elapsed = time.perf_counter() - t_step0
                to_sleep = dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
    finally:
        if trace_f is not None:
            trace_f.close()

    wall_s = time.perf_counter() - t_wall0
    steps_executed = (int(success_step) + 1) if success_step is not None else int(episode_steps)
    achieved_hz = float(steps_executed) / wall_s if wall_s > 0 else float("nan")

    if success_cfg.enabled and isinstance(backend, HazelGrpcBackend):
        if success_step is not None:
            failure_mode = "success"
        elif block_id is None or drop_pos0 is None or block_z0 is None:
            failure_mode = "no_entities"
        elif picked_step is None:
            failure_mode = "no_pick"
        else:
            failure_mode = "no_place"

    summary = {
        "infer_times_s": infer_s,
        "backend_times_s": backend_s,
        "cam_skew_ms": cam_skew_ms,
        "last_timestamps_s": ts,
    }

    dummy = EpisodeResult(
        checkpoint="",
        variant="",
        try_idx=-1,
        steps=int(steps_executed),
        wall_s=float(wall_s),
        achieved_hz=float(achieved_hz),
        infer_mean_ms=_mean_ms(infer_s),
        infer_p95_ms=_percentile_ms(infer_s, 95),
        backend_mean_ms=_mean_ms(backend_s),
        backend_p95_ms=_percentile_ms(backend_s, 95),
        cam_skew_p95_ms=float(np.percentile(np.asarray(cam_skew_ms), 95)) if cam_skew_ms else float("nan"),
        gripper_transitions=_count_transitions(gripper_trace),
        gripper_min=float(np.min(gripper_trace)),
        gripper_max=float(np.max(gripper_trace)),
        picked_step=picked_step,
        placed_step=placed_step,
        success_step=success_step,
        failure_mode=failure_mode,
        error=None,
    )
    return dummy, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ACT inference knobs against LuckyEngine/Hazel (n_action_steps, temporal_ensemble_coeff, torch.compile, pacing)."
    )
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("lerobot/outputs/train/act_piper_01192026/checkpoints"),
        help="Folder containing numbered checkpoint subfolders (each has pretrained_model/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help='Checkpoint folder name under checkpoints_root (e.g. 074750) OR a path to a checkpoint folder or pretrained_model/. Use "all" to sweep all checkpoints.',
    )
    parser.add_argument("--limit_checkpoints", type=int, default=0, help="If >0 and --checkpoint all, only run the first N checkpoints.")
    parser.add_argument(
        "--min_checkpoint",
        type=int,
        default=None,
        help='If set and --checkpoint all, only run checkpoints with numeric folder name >= this value (e.g. 3000 runs 003000 and above).',
    )
    parser.add_argument("--backend", choices=["mock", "hazel"], default="hazel", help="Backend implementation to use.")
    parser.add_argument("--hazel_address", type=str, default="127.0.0.1:50051", help="Hazel gRPC host:port.")
    parser.add_argument(
        "--hazel_stubs_dir",
        type=Path,
        default=None,
        help="Directory containing generated Hazel python stubs (e.g. *_pb2.py, *_pb2_grpc.py).",
    )
    parser.add_argument("--hazel_agent_name", type=str, default="PiperAgent", help="Agent name in Hazel.")
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--device", type=str, default=None, help="Torch device for policy inference (cuda/cpu).")
    parser.add_argument("--task", type=str, default=None, help="Optional task string (if policy expects it).")
    parser.add_argument("--episode_steps", type=int, default=400, help="Steps per episode.")
    parser.add_argument("--tries_per_variant", type=int, default=1, help="Episodes per variant.")
    parser.add_argument(
        "--n_action_steps_grid",
        type=str,
        default="30,10,5",
        help="Comma-separated n_action_steps values to test (replan frequency).",
    )
    parser.add_argument(
        "--temporal_ensemble_coeff_grid",
        type=str,
        default="0.01",
        help='Comma-separated temporal_ensemble_coeff values to test. Use "none" to disable. '
        "NOTE: any non-none value forces n_action_steps=1 for that variant.",
    )
    parser.add_argument(
        "--control_hz",
        type=float,
        default=None,
        help="If set, enforce a max control loop frequency by sleeping to match this Hz.",
    )
    parser.add_argument(
        "--sync_control_hz_to_dataset",
        action="store_true",
        help="If set and --home_from_dataset is provided, auto-set --control_hz to dataset fps (unless --control_hz is already set).",
    )
    parser.add_argument(
        "--home_from_dataset",
        type=Path,
        default=None,
        help="Optional local LeRobot dataset directory (e.g. lerobot/dataset/session_20260119_135603) to read home from episode-0 frame-0 observation.state.",
    )
    parser.add_argument("--home_dataset_episode", type=int, default=0)
    parser.add_argument("--home_dataset_frame", type=int, default=0)
    parser.add_argument("--home_settle_steps", type=int, default=50, help="How many control steps to hold home pose.")
    parser.add_argument("--home_settle_sleep_s", type=float, default=0.02, help="Sleep between settle steps.")
    parser.add_argument("--out_dir", type=Path, default=Path("lerobot/outputs/tuning/act_piper_room_params"))
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile on the ACT model for faster inference.")
    parser.add_argument(
        "--torch_compile_backend",
        type=str,
        default="inductor",
        help="Backend for torch.compile (inductor, aot_eager, cudagraphs).",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="default",
        help="Compilation mode (default, reduce-overhead, max-autotune).",
    )
    parser.add_argument(
        "--skip_startup_checks",
        action="store_true",
        help="Skip startup contract checks (not recommended).",
    )
    parser.add_argument(
        "--debug_trace",
        action="store_true",
        help="Write per-step trace JSONL (state/action/timestamps) and per-chunk planned actions for debugging repeated chunks.",
    )
    parser.add_argument(
        "--debug_trace_only_checkpoint",
        type=str,
        default=None,
        help="If set, only write traces for this checkpoint name (e.g. 005000).",
    )
    parser.add_argument(
        "--debug_trace_only_variant",
        type=str,
        default=None,
        help="If set, only write traces for this variant (e.g. replan_n30 or ensemble_c0.01).",
    )

    # Success detection (red block -> dropbox) via SceneService
    parser.add_argument("--success_enable", action="store_true", help="Enable success detection via SceneService entity transforms.")
    parser.add_argument("--success_block_name_substr", type=str, default="red", help="Substring used to find the red block entity.")
    parser.add_argument("--success_dropbox_name_substr", type=str, default="drop", help="Substring used to find the dropbox entity.")
    parser.add_argument("--success_block_exact_name", type=str, default=None, help="Exact red block entity name (preferred over substring match).")
    parser.add_argument("--success_dropbox_exact_name", type=str, default=None, help="Exact dropbox entity name (preferred over substring match).")
    parser.add_argument("--success_lift_dz_m", type=float, default=0.02, help="Pick threshold: block z must increase by this many meters vs start.")
    parser.add_argument("--success_drop_xy_tol_m", type=float, default=0.07, help="Place threshold: XY distance to dropbox center (meters).")
    parser.add_argument("--success_drop_z_tol_m", type=float, default=0.03, help="Place threshold: block z must be <= dropbox_z + tol (meters).")
    parser.add_argument("--success_confirm_steps", type=int, default=5, help="How many consecutive polls are needed to confirm pick/place.")
    parser.add_argument("--success_scene_poll_every", type=int, default=3, help="Poll SceneService every N sim steps (reduce overhead).")

    # Assisted pick/place (uses SceneService ground truth; helps "make it work" even if BC policy is imperfect)
    parser.add_argument("--assist_enable", action="store_true", help="Enable assisted gripper close/open using SceneService transforms.")
    parser.add_argument(
        "--assist_gripper_name_substr",
        type=str,
        default="gripper",
        help='Substring to find the robot gripper/end-effector entity in SceneService (e.g. "gripper" / "tcp").',
    )
    parser.add_argument("--assist_gripper_exact_name", type=str, default=None, help="Exact gripper entity name (preferred over substring).")
    parser.add_argument("--assist_pick_xy_tol_m", type=float, default=0.05, help="When gripper is within this XY distance of the block, force close.")
    parser.add_argument(
        "--assist_place_xy_tol_m",
        type=float,
        default=0.07,
        help="When block is within this XY distance of dropbox AND below z tol, force open.",
    )
    parser.add_argument("--assist_close_value", type=float, default=0.0, help="Gripper command value to use for forced close (in action space units).")
    parser.add_argument("--assist_open_value", type=float, default=0.035, help="Gripper command value to use for forced open (in action space units).")
    parser.add_argument("--assist_hold_steps", type=int, default=15, help="How many control steps to hold the forced gripper command once triggered.")
    parser.add_argument("--assist_freeze_arm", action="store_true", help="If set, freeze arm joints while forcing the gripper close/open (helps stabilize grasps).")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_arg = (args.checkpoint or "").strip().lower()
    if ckpt_arg == "all":
        ckpt_dirs = iter_checkpoints(args.checkpoints_root)
        if args.min_checkpoint is not None:
            min_ck = int(args.min_checkpoint)
            # Keep only numeric checkpoint folder names (e.g. "003000"), ignore others (e.g. "last").
            filtered: list[Path] = []
            for d in ckpt_dirs:
                try:
                    n = int(d.parent.name)
                except Exception:
                    continue
                if n >= min_ck:
                    filtered.append(d)
            ckpt_dirs = filtered
        if args.limit_checkpoints and args.limit_checkpoints > 0:
            ckpt_dirs = ckpt_dirs[: int(args.limit_checkpoints)]
        if not ckpt_dirs:
            raise SystemExit(f"No checkpoints found under {args.checkpoints_root}")
    else:
        ckpt_dirs = [_pick_pretrained_dir(checkpoints_root=args.checkpoints_root, checkpoint=args.checkpoint)]

    home_action7 = None
    dataset_fps = None
    if args.home_from_dataset is not None:
        home_action7 = _load_home_from_dataset(
            args.home_from_dataset, episode=args.home_dataset_episode, frame=args.home_dataset_frame
        )
        dataset_fps = _read_dataset_fps(args.home_from_dataset)

    control_hz = args.control_hz
    if control_hz is None and args.sync_control_hz_to_dataset and dataset_fps is not None:
        control_hz = float(dataset_fps)

    # Build variants
    n_steps_list = _parse_csv_ints(args.n_action_steps_grid)
    coeffs_list = _parse_csv_floats_or_none(args.temporal_ensemble_coeff_grid)

    # De-dup while preserving order
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    n_steps_list = _dedup(n_steps_list)
    coeffs_list = _dedup(coeffs_list)

    variants: list[Variant] = []
    for n in n_steps_list:
        variants.append(Variant(name=f"replan_n{n}", n_action_steps=int(n), temporal_ensemble_coeff=None, torch_compile=bool(args.torch_compile)))
    for c in coeffs_list:
        if c is None:
            continue
        variants.append(
            Variant(
                name=f"ensemble_c{c:g}",
                n_action_steps=1,
                temporal_ensemble_coeff=float(c),
                torch_compile=bool(args.torch_compile),
            )
        )

    # Output dir
    run_tag = "ALL" if ckpt_arg == "all" else ckpt_dirs[0].parent.name
    run_dir = args.out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "results.csv"
    out_jsonl = run_dir / "results.jsonl"
    # "Sweep-style" summary files (mirrors `sweep_act_piper_room_checkpoints.py` schema,
    # plus `variant` to disambiguate rows).
    summary_csv = run_dir / "summary.csv"
    summary_jsonl = run_dir / "summary.jsonl"

    backend = None
    try:
        if args.backend == "mock":
            backend = MockPiperRoomBackend(width=args.image_width, height=args.image_height, seed=0)
        elif args.backend == "hazel":
            if args.hazel_stubs_dir is None:
                raise SystemExit("--hazel_stubs_dir is required when --backend hazel")
            backend = HazelGrpcBackend(
                address=args.hazel_address,
                stubs_dir=args.hazel_stubs_dir,
                agent_name=args.hazel_agent_name,
                image_width=args.image_width,
                image_height=args.image_height,
            )
            backend.connect()
        else:
            raise SystemExit(f"Unsupported backend: {args.backend}")

        success_cfg = SuccessConfig(
            enabled=bool(args.success_enable),
            block_name_substr=str(args.success_block_name_substr),
            dropbox_name_substr=str(args.success_dropbox_name_substr),
            block_exact_name=args.success_block_exact_name,
            dropbox_exact_name=args.success_dropbox_exact_name,
            lift_dz_m=float(args.success_lift_dz_m),
            drop_xy_tol_m=float(args.success_drop_xy_tol_m),
            drop_z_tol_m=float(args.success_drop_z_tol_m),
            confirm_steps=int(args.success_confirm_steps),
            scene_poll_every=int(args.success_scene_poll_every),
        )
        assist_cfg = AssistConfig(
            enabled=bool(args.assist_enable),
            gripper_name_substr=str(args.assist_gripper_name_substr),
            gripper_exact_name=args.assist_gripper_exact_name,
            pick_xy_tol_m=float(args.assist_pick_xy_tol_m),
            place_xy_tol_m=float(args.assist_place_xy_tol_m),
            close_value=float(args.assist_close_value),
            open_value=float(args.assist_open_value),
            hold_steps=int(args.assist_hold_steps),
            freeze_arm=bool(args.assist_freeze_arm),
        )
        if home_action7 is not None:
            print(
                f"{_now()} | HOME | from_dataset={args.home_from_dataset} ep={args.home_dataset_episode} frame={args.home_dataset_frame} | settle_steps={args.home_settle_steps} sleep={args.home_settle_sleep_s}",
                flush=True,
            )
        if control_hz is not None:
            print(f"{_now()} | PACE | control_hz={control_hz:g}", flush=True)
        if args.torch_compile:
            print(
                f"{_now()} | COMPILE | on | backend={args.torch_compile_backend} mode={args.torch_compile_mode}",
                flush=True,
            )

        policy_class = get_policy_class("act")

        with (
            out_csv.open("w", newline="", encoding="utf-8") as f_csv,
            out_jsonl.open("w", encoding="utf-8") as f_jsonl,
            summary_csv.open("w", newline="", encoding="utf-8") as f_sum_csv,
            summary_jsonl.open("w", encoding="utf-8") as f_sum_jsonl,
        ):
            writer = csv.DictWriter(
                f_csv,
                fieldnames=[
                    "checkpoint",
                    "variant",
                    "try_idx",
                    "steps",
                    "wall_s",
                    "achieved_hz",
                    "infer_mean_ms",
                    "infer_p95_ms",
                    "backend_mean_ms",
                    "backend_p95_ms",
                    "cam_skew_p95_ms",
                    "gripper_transitions",
                    "gripper_min",
                    "gripper_max",
                    "picked_step",
                    "placed_step",
                    "success_step",
                    "failure_mode",
                    "error",
                ],
            )
            writer.writeheader()

            sum_writer = csv.DictWriter(
                f_sum_csv,
                fieldnames=[
                    "checkpoint",
                    "variant",
                    "try_idx",
                    "steps",
                    "gripper_transitions",
                    "gripper_min",
                    "gripper_max",
                    "missing_keys_n",
                    "unexpected_keys_n",
                    "missing_keys_head",
                    "unexpected_keys_head",
                    "error",
                    "elapsed_s",
                ],
            )
            sum_writer.writeheader()

            for ckpt_i, pretrained_dir in enumerate(ckpt_dirs, start=1):
                checkpoint_name = pretrained_dir.parent.name
                contract = load_policy_contract_from_pretrained(pretrained_dir)
                if contract.policy_type != "act":
                    raise ValueError(f"Expected ACT checkpoint, got policy type `{contract.policy_type}` at {pretrained_dir}")

                if not args.skip_startup_checks:
                    cams = backend.list_cameras()
                    required_cams = {k.split(".")[-1] for k in contract.image_keys}
                    missing = sorted(required_cams - set(cams))
                    if missing:
                        raise ValueError(f"Backend missing required cameras {missing}. Backend has: {sorted(cams)}")
                    h, w = _backend_image_hw(backend)
                    images_chw = {f"observation.images.{cam}": (3, h, w) for cam in cams}
                    assert_expected_keys_and_shapes(
                        contract,
                        state_dim=backend.state_dim(),
                        action_dim=backend.action_dim(),
                        images_chw=images_chw,
                    )

                print(f"{_now()} | CKPT | {ckpt_i}/{len(ckpt_dirs)} | {checkpoint_name} | pretrained={pretrained_dir}", flush=True)

                for v in variants:
                    cfg = load_act_config_from_pretrained_compat(pretrained_dir, device=args.device)
                    _validate_variant(v, chunk_size=int(cfg.chunk_size))
                    cfg.n_action_steps = int(v.n_action_steps)
                    cfg.temporal_ensemble_coeff = v.temporal_ensemble_coeff

                    policy, report = load_local_policy_with_report(
                        policy_class=policy_class, cfg=cfg, pretrained_dir=pretrained_dir
                    )

                    if v.torch_compile and hasattr(torch, "compile"):
                        try:
                            policy.model = torch.compile(
                                policy.model, backend=args.torch_compile_backend, mode=args.torch_compile_mode
                            )
                        except Exception as e:  # noqa: BLE001 (example script)
                            print(f"{_now()} | WARN | torch.compile failed: {type(e).__name__}: {e}", flush=True)

                    preprocessor, postprocessor = make_pre_post_processors(
                        policy_cfg=policy.config,
                        pretrained_path=str(pretrained_dir),
                        preprocessor_overrides={"device_processor": {"device": args.device}},
                        postprocessor_overrides={"device_processor": {"device": "cpu"}},
                    )

                    print(
                        f"{_now()} | VAR | {checkpoint_name} | {v.name} | n_action_steps={v.n_action_steps} temporal_ensemble_coeff={v.temporal_ensemble_coeff} | weights_miss={len(report.missing_keys)} unexp={len(report.unexpected_keys)}",
                        flush=True,
                    )

                    for try_idx in range(int(args.tries_per_variant)):
                        t0 = time.perf_counter()
                        error = None
                        res = None
                        try:
                            trace_path = None
                            if bool(args.debug_trace):
                                if args.debug_trace_only_checkpoint and checkpoint_name != str(args.debug_trace_only_checkpoint):
                                    trace_path = None
                                elif args.debug_trace_only_variant and v.name != str(args.debug_trace_only_variant):
                                    trace_path = None
                                else:
                                    trace_dir = run_dir / "traces"
                                    trace_dir.mkdir(parents=True, exist_ok=True)
                                    trace_path = trace_dir / f"trace_{checkpoint_name}_{v.name}_try{int(try_idx)}.jsonl"

                            res, _debug = run_episode(
                                policy=policy,
                                preprocessor=preprocessor,
                                postprocessor=postprocessor,
                                contract=contract,
                                backend=backend,
                                episode_steps=int(args.episode_steps),
                                task=args.task,
                                home_action7=home_action7,
                                home_settle_steps=int(args.home_settle_steps),
                                home_settle_sleep_s=float(args.home_settle_sleep_s),
                                control_hz=control_hz,
                                success_cfg=success_cfg,
                                assist_cfg=assist_cfg,
                                trace_jsonl_path=trace_path,
                            )
                        except Exception as e:  # noqa: BLE001 (example script)
                            error = f"{type(e).__name__}: {e}"
                        elapsed = time.perf_counter() - t0

                        if res is None:
                            res = EpisodeResult(
                                checkpoint=checkpoint_name,
                                variant=v.name,
                                try_idx=int(try_idx),
                                steps=int(args.episode_steps),
                                wall_s=float("nan"),
                                achieved_hz=float("nan"),
                                infer_mean_ms=float("nan"),
                                infer_p95_ms=float("nan"),
                                backend_mean_ms=float("nan"),
                                backend_p95_ms=float("nan"),
                                cam_skew_p95_ms=float("nan"),
                                gripper_transitions=0,
                                gripper_min=float("nan"),
                                gripper_max=float("nan"),
                                picked_step=None,
                                placed_step=None,
                                success_step=None,
                                failure_mode=None,
                                error=error,
                            )
                        else:
                            res.checkpoint = checkpoint_name
                            res.variant = v.name
                            res.try_idx = int(try_idx)
                            res.error = error

                        row = asdict(res)
                        writer.writerow(row)
                        f_csv.flush()
                        f_jsonl.write(json.dumps(row) + "\n")
                        f_jsonl.flush()

                        # Also emit "sweep-style" summary row for quick metadata inspection.
                        sum_row = {
                            "checkpoint": checkpoint_name,
                            "variant": v.name,
                            "try_idx": int(try_idx),
                            "steps": int(getattr(res, "steps", args.episode_steps) or args.episode_steps),
                            "gripper_transitions": int(getattr(res, "gripper_transitions", 0) or 0),
                            "gripper_min": float(getattr(res, "gripper_min", float("nan"))),
                            "gripper_max": float(getattr(res, "gripper_max", float("nan"))),
                            "missing_keys_n": len(report.missing_keys),
                            "unexpected_keys_n": len(report.unexpected_keys),
                            "missing_keys_head": ";".join(report.missing_keys[:10]),
                            "unexpected_keys_head": ";".join(report.unexpected_keys[:10]),
                            "error": error,
                            "elapsed_s": round(float(elapsed), 6),
                        }
                        sum_writer.writerow(sum_row)
                        f_sum_csv.flush()
                        f_sum_jsonl.write(json.dumps(sum_row) + "\n")
                        f_sum_jsonl.flush()

                        status = "OK" if error is None else "ERR"
                        print(
                            f"{_now()} | {status} | {checkpoint_name} | {v.name} | try {try_idx+1}/{args.tries_per_variant} | elapsed={elapsed:.2f}s | hz={res.achieved_hz:.1f} | inf_p95={res.infer_p95_ms:.1f}ms | back_p95={res.backend_p95_ms:.1f}ms | success={res.success_step is not None} | fail={res.failure_mode}",
                            flush=True,
                        )

        print(
            f"{_now()} | DONE | out={run_dir} | results={out_csv.name},{out_jsonl.name} | summary={summary_csv.name},{summary_jsonl.name}",
            flush=True,
        )
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()


