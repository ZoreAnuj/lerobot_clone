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
# File is: lerobot/examples/luckyengine/tune_diffusion_piper_room_params.py
# We want to add: lerobot/src
_src_dir = Path(__file__).resolve().parents[2] / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lerobot.luckyengine.config_compat import load_diffusion_config_from_pretrained_compat
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


def _hwc_uint8_to_chw_float01(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) uint8 RGB -> (3,H,W) float32 in [0,1]."""
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC with 3 channels, got {img.shape}")
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _apply_roi_crop_resize(
    img_hwc: np.ndarray,
    crop_params: tuple[int, int, int, int],
    resize_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Apply ROI crop and resize to an HWC uint8 image.
    
    Args:
        img_hwc: Input image in HWC uint8 format.
        crop_params: (top, left, height, width) crop parameters.
        resize_size: Target (height, width) after resize.
    
    Returns:
        Cropped and resized HWC uint8 image.
    """
    import cv2
    top, left, height, width = crop_params
    cropped = img_hwc[top:top+height, left:left+width]
    resized = cv2.resize(cropped, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized


def _load_crop_params(crop_params_path: Path | None) -> dict[str, tuple[int, int, int, int]]:
    """Load crop params from JSON file, or return empty dict if None."""
    if crop_params_path is None:
        return {}
    with open(crop_params_path) as f:
        return json.load(f)


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


def _parse_csv_optional_ints(s: str) -> list[int | None]:
    if not s:
        return []
    out: list[int | None] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(int(part))
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
    num_inference_steps: int | None
    torch_compile: bool


@dataclass
class EpisodeResult:
    checkpoint: str
    variant: str
    try_idx: int
    steps: int
    gripper_transitions: int
    gripper_min: float
    gripper_max: float
    wall_s: float
    achieved_hz: float
    infer_mean_ms: float
    infer_p95_ms: float
    backend_mean_ms: float
    backend_p95_ms: float
    cam_skew_p95_ms: float
    error: str | None


def _validate_variant(*, base_cfg, v: Variant) -> None:
    n_obs = int(getattr(base_cfg, "n_obs_steps", 1) or 1)
    horizon = int(getattr(base_cfg, "horizon", 1) or 1)
    if v.n_action_steps <= 0:
        raise ValueError("n_action_steps must be >0")
    # DiffusionPolicy requires: n_action_steps <= horizon - n_obs_steps + 1
    max_n = horizon - n_obs + 1
    if v.n_action_steps > int(max_n):
        raise ValueError(
            f"n_action_steps={v.n_action_steps} must be <= horizon-n_obs_steps+1 = {horizon}-{n_obs}+1 = {max_n}"
        )
    if v.num_inference_steps is not None and int(v.num_inference_steps) <= 0:
        raise ValueError("num_inference_steps must be >0 or None")


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
    trace_jsonl_path: Path | None = None,
    crop_params: dict[str, tuple[int, int, int, int]] | None = None,
    crop_resize_size: tuple[int, int] = (128, 128),
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
    # If crop_params is provided, use cropped/resized dimensions.
    if crop_params:
        h, w = crop_resize_size
    else:
        h, w = _backend_image_hw(backend)
    images_chw = {f"observation.images.{cam}": (3, h, w) for cam in cam_names}
    assert_expected_keys_and_shapes(
        contract,
        state_dim=backend.state_dim(),
        action_dim=backend.action_dim(),
        images_chw=images_chw,
    )

    dt = (1.0 / float(control_hz)) if control_hz is not None and control_hz > 0 else None

    trace_f = None
    if trace_jsonl_path is not None:
        trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        trace_f = trace_jsonl_path.open("w", encoding="utf-8")
        trace_f.write(
            json.dumps(
                {
                    "event": "meta",
                    "n_obs_steps": int(getattr(getattr(policy, "config", None), "n_obs_steps", -1) or -1),
                    "horizon": int(getattr(getattr(policy, "config", None), "horizon", -1) or -1),
                    "n_action_steps": int(getattr(getattr(policy, "config", None), "n_action_steps", -1) or -1),
                    "num_inference_steps": getattr(getattr(policy, "config", None), "num_inference_steps", None),
                    "cam_names": cam_names,
                    "state_key": contract.state_key,
                    "image_keys": list(contract.image_keys),
                }
            )
            + "\n"
        )

    infer_s: list[float] = []
    backend_s: list[float] = []
    cam_skew_ms: list[float] = []
    gripper_trace: list[float] = [float(state0[-1])]

    state = state0
    images = images0
    ts = ts0

    # Diffusion chunk tracking: queue holds normalized actions; empty => new chunk is sampled.
    chunk_id = 0
    chunk_step = 0

    t_wall0 = time.perf_counter()
    try:
        for step_i in range(int(episode_steps)):
            t_step0 = time.perf_counter()

            action_q = getattr(policy, "_queues", {}).get("action", None)
            queue_empty_before = bool(action_q is not None and len(action_q) == 0)
            if queue_empty_before:
                chunk_id += 1
                chunk_step = 0
            else:
                chunk_step += 1

            batch: dict[str, object] = {}
            batch[contract.state_key] = torch.from_numpy(state).to(dtype=torch.float32)

            for k in contract.image_keys:
                cam_name = k.split(".")[-1]
                img_hwc = images[cam_name]
                # Apply ROI crop and resize if crop_params is provided
                if crop_params and k in crop_params:
                    img_hwc = _apply_roi_crop_resize(img_hwc, crop_params[k], crop_resize_size)
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

            # If we just generated a new action chunk, optionally dump the planned chunk in action-space.
            if trace_f is not None and queue_empty_before and int(getattr(policy.config, "n_action_steps", 1)) > 1:
                planned_action7: list[list[float]] = [action_np.tolist()]
                # Remaining actions in the queue are still in normalized space; postprocess them.
                queues = getattr(policy, "_queues", {}) or {}
                for a_norm in list(queues.get("action", []) or []):
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
                            "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
                            "action_action7": action_np.tolist(),
                            "timestamps_s": {k: float(v) for k, v in (ts or {}).items()},
                        }
                    )
                    + "\n"
                )

            if dt is not None:
                elapsed = time.perf_counter() - t_step0
                to_sleep = dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)
    finally:
        if trace_f is not None:
            trace_f.close()

    wall_s = time.perf_counter() - t_wall0
    achieved_hz = float(int(episode_steps)) / wall_s if wall_s > 0 else float("nan")

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
        steps=int(episode_steps),
        gripper_transitions=_count_transitions(gripper_trace),
        gripper_min=float(np.min(np.asarray(gripper_trace, dtype=np.float32))),
        gripper_max=float(np.max(np.asarray(gripper_trace, dtype=np.float32))),
        wall_s=float(wall_s),
        achieved_hz=float(achieved_hz),
        infer_mean_ms=_mean_ms(infer_s),
        infer_p95_ms=_percentile_ms(infer_s, 95),
        backend_mean_ms=_mean_ms(backend_s),
        backend_p95_ms=_percentile_ms(backend_s, 95),
        cam_skew_p95_ms=float(np.percentile(np.asarray(cam_skew_ms), 95)) if cam_skew_ms else float("nan"),
        error=None,
    )
    return dummy, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune DiffusionPolicy inference knobs against LuckyEngine/Hazel (n_action_steps, num_inference_steps, torch.compile, pacing)."
    )
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("lerobot/outputs/train/diffusion_piper_0120/checkpoints"),
        help="Folder containing numbered checkpoint subfolders (each has pretrained_model/).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help='Checkpoint folder name under checkpoints_root (e.g. 024750) OR a path to a checkpoint folder or pretrained_model/. Use "all" to sweep all checkpoints.',
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
    parser.add_argument(
        "--hazel_pb2_module",
        type=str,
        default="hazel_rpc_pb2",
        help="Python module name for Hazel pb2 stubs (expects `<name>.py` in --hazel_stubs_dir).",
    )
    parser.add_argument(
        "--hazel_pb2_grpc_module",
        type=str,
        default="hazel_rpc_pb2_grpc",
        help="Python module name for Hazel pb2_grpc stubs (expects `<name>.py` in --hazel_stubs_dir).",
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
        default="8,4,2,1",
        help="Comma-separated n_action_steps values to test (new chunk only when action queue is empty).",
    )
    parser.add_argument(
        "--num_inference_steps_grid",
        type=str,
        default="10,5,2",
        help='Comma-separated num_inference_steps values to test. Use "none" to keep checkpoint default.',
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
        default=Path("lerobot/dataset/session_20260119_135603"),
        help="Optional local LeRobot dataset directory to read home from episode-0 frame-0 observation.state.",
    )
    parser.add_argument("--home_dataset_episode", type=int, default=0)
    parser.add_argument("--home_dataset_frame", type=int, default=0)
    parser.add_argument("--home_settle_steps", type=int, default=50, help="How many control steps to hold home pose.")
    parser.add_argument("--home_settle_sleep_s", type=float, default=0.02, help="Sleep between settle steps.")
    parser.add_argument(
        "--crop_params",
        type=Path,
        default=None,
        help="Path to crop_params.json for ROI cropping. If provided, images will be cropped and resized to 128x128 before feeding to the policy.",
    )
    parser.add_argument(
        "--crop_resize_size",
        type=int,
        nargs=2,
        default=[128, 128],
        help="Target (height, width) after ROI crop resize. Default: 128 128",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("lerobot/outputs/sweeps/diffusion_piper_room_full_3x_home"))
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile on the diffusion model for faster inference (best-effort).")
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
        help="Write per-step trace JSONL (action/timestamps) and per-chunk planned actions for debugging.",
    )
    parser.add_argument(
        "--debug_trace_only_checkpoint",
        type=str,
        default=None,
        help="If set, only write traces for this checkpoint name (e.g. 024750).",
    )
    parser.add_argument(
        "--debug_trace_only_variant",
        type=str,
        default=None,
        help="If set, only write traces for this variant (e.g. n8_steps10).",
    )
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_arg = (args.checkpoint or "").strip().lower()
    if ckpt_arg == "all":
        ckpt_dirs = iter_checkpoints(args.checkpoints_root)
        if args.min_checkpoint is not None:
            min_ck = int(args.min_checkpoint)
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
        home_action7 = _load_home_from_dataset(args.home_from_dataset, episode=args.home_dataset_episode, frame=args.home_dataset_frame)
        dataset_fps = _read_dataset_fps(args.home_from_dataset)

    # Load ROI crop params if provided
    crop_params = _load_crop_params(args.crop_params)
    if crop_params:
        print(f"[{_now()}] Loaded ROI crop params from {args.crop_params}: {list(crop_params.keys())}")
        print(f"[{_now()}] Images will be cropped and resized to {tuple(args.crop_resize_size)}")

    control_hz = args.control_hz
    if control_hz is None and args.sync_control_hz_to_dataset and dataset_fps is not None:
        control_hz = float(dataset_fps)

    # Build variants (dedup while preserving order)
    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    n_steps_list = _dedup(_parse_csv_ints(args.n_action_steps_grid))
    infer_steps_list = _dedup(_parse_csv_optional_ints(args.num_inference_steps_grid))

    variants: list[Variant] = []
    for n in n_steps_list:
        for s in infer_steps_list:
            name = f"n{int(n)}_steps{('none' if s is None else int(s))}"
            variants.append(
                Variant(
                    name=name,
                    n_action_steps=int(n),
                    num_inference_steps=(None if s is None else int(s)),
                    torch_compile=bool(args.torch_compile),
                )
            )

    # Output dir
    # For `--checkpoint all` we mirror the sweep scripts: write directly into `--out_dir` (overwriting files).
    if ckpt_arg == "all":
        run_dir = args.out_dir
    else:
        run_dir = args.out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ckpt_dirs[0].parent.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "results.csv"
    out_jsonl = run_dir / "results.jsonl"
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
                pb2_module=str(args.hazel_pb2_module),
                pb2_grpc_module=str(args.hazel_pb2_grpc_module),
                agent_name=args.hazel_agent_name,
                image_width=args.image_width,
                image_height=args.image_height,
            )
            backend.connect()
        else:
            raise SystemExit(f"Unsupported backend: {args.backend}")

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

        policy_class = get_policy_class("diffusion")

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
                    "gripper_transitions",
                    "gripper_min",
                    "gripper_max",
                    "wall_s",
                    "achieved_hz",
                    "infer_mean_ms",
                    "infer_p95_ms",
                    "backend_mean_ms",
                    "backend_p95_ms",
                    "cam_skew_p95_ms",
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
                if contract.policy_type != "diffusion":
                    raise ValueError(
                        f"Expected diffusion checkpoint, got policy type `{contract.policy_type}` at {pretrained_dir}"
                    )

                if not args.skip_startup_checks:
                    cams = backend.list_cameras()
                    required_cams = {k.split(".")[-1] for k in contract.image_keys}
                    missing = sorted(required_cams - set(cams))
                    if missing:
                        raise ValueError(f"Backend missing required cameras {missing}. Backend has: {sorted(cams)}")
                    # Use cropped dimensions if crop_params is provided
                    if crop_params:
                        h, w = tuple(args.crop_resize_size)
                    else:
                        h, w = _backend_image_hw(backend)
                    images_chw = {f"observation.images.{cam}": (3, h, w) for cam in cams}
                    assert_expected_keys_and_shapes(
                        contract,
                        state_dim=backend.state_dim(),
                        action_dim=backend.action_dim(),
                        images_chw=images_chw,
                    )

                print(f"{_now()} | CKPT | {ckpt_i}/{len(ckpt_dirs)} | {checkpoint_name} | pretrained={pretrained_dir}", flush=True)

                # Load a base config once (used for validation); variant will override a few knobs.
                base_cfg = load_diffusion_config_from_pretrained_compat(pretrained_dir, device=args.device)

                for v in variants:
                    _validate_variant(base_cfg=base_cfg, v=v)
                    cfg = load_diffusion_config_from_pretrained_compat(pretrained_dir, device=args.device)
                    cfg.n_action_steps = int(v.n_action_steps)
                    if v.num_inference_steps is not None:
                        cfg.num_inference_steps = int(v.num_inference_steps)

                    policy, report = load_local_policy_with_report(policy_class=policy_class, cfg=cfg, pretrained_dir=pretrained_dir)

                    if v.torch_compile and hasattr(torch, "compile"):
                        try:
                            if hasattr(policy, "diffusion"):
                                policy.diffusion = torch.compile(
                                    policy.diffusion, backend=args.torch_compile_backend, mode=args.torch_compile_mode
                                )
                        except Exception as e:  # noqa: BLE001 (example script)
                            print(f"{_now()} | WARN | torch.compile failed: {type(e).__name__}: {e}", flush=True)

                    # Load the saved processors from the checkpoint; override device placement.
                    preprocessor, postprocessor = make_pre_post_processors(
                        policy_cfg=policy.config,
                        pretrained_path=str(pretrained_dir),
                        preprocessor_overrides={"device_processor": {"device": args.device}},
                        postprocessor_overrides={"device_processor": {"device": "cpu"}},
                    )

                    print(
                        f"{_now()} | VAR | {checkpoint_name} | {v.name} | n_action_steps={v.n_action_steps} num_inference_steps={cfg.num_inference_steps} | weights_miss={len(report.missing_keys)} unexp={len(report.unexpected_keys)}",
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
                                trace_jsonl_path=trace_path,
                                crop_params=crop_params,
                                crop_resize_size=tuple(args.crop_resize_size),
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
                                gripper_transitions=0,
                                gripper_min=float("nan"),
                                gripper_max=float("nan"),
                                wall_s=float("nan"),
                                achieved_hz=float("nan"),
                                infer_mean_ms=float("nan"),
                                infer_p95_ms=float("nan"),
                                backend_mean_ms=float("nan"),
                                backend_p95_ms=float("nan"),
                                cam_skew_p95_ms=float("nan"),
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

                        # Sweep-style summary (mirrors ACT summary schema but includes `variant`).
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
                            f"{_now()} | {status} | {checkpoint_name} | {v.name} | try {try_idx+1}/{args.tries_per_variant} | elapsed={elapsed:.2f}s | hz={res.achieved_hz:.1f} | inf_p95={res.infer_p95_ms:.1f}ms | back_p95={res.backend_p95_ms:.1f}ms",
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


