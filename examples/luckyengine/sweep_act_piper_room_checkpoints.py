#!/usr/bin/env python

from __future__ import annotations

import sys
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import packaging.version
import safetensors
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_model as load_model_as_safetensor

# Allow running as a plain script from repo root without having to set PYTHONPATH.
# File is: lerobot/examples/luckyengine/sweep_act_piper_room_checkpoints.py
# We want to add: lerobot/src
_src_dir = Path(__file__).resolve().parents[2] / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from lerobot.luckyengine.contracts import (
    assert_expected_keys_and_shapes,
    load_policy_contract_from_pretrained,
)
from lerobot.luckyengine.config_compat import load_act_config_from_pretrained_compat
from lerobot.luckyengine.hazel_backend import HazelGrpcBackend
from lerobot.luckyengine.mock_backend import MockPiperRoomBackend
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _hwc_uint8_to_chw_float01(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,3) uint8 RGB -> (3,H,W) float32 in [0,1]."""
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HWC with 3 channels, got {img.shape}")
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


@dataclass
class EpisodeMetrics:
    steps: int
    gripper_transitions: int
    gripper_min: float
    gripper_max: float


@dataclass
class WeightLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]


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


def iter_checkpoints(checkpoints_root: Path) -> list[Path]:
    ckpts = []
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


def _parse_action7(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.replace("[", "").replace("]", "").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if len(vals) != 7:
        raise ValueError(f"--home_action must have 7 floats, got {len(vals)}: {s}")
    return np.asarray(vals, dtype=np.float32)


def _load_home_from_dataset(dataset_dir: Path, *, episode: int = 0, frame: int = 0) -> np.ndarray:
    """Loads dataset-home pose from a local LeRobot dataset directory as `observation.state`."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    # LeRobotDataset expects `root` to be the dataset directory itself when loading locally.
    ds = LeRobotDataset(repo_id=dataset_dir.name, root=dataset_dir, download_videos=False)
    from_idx = ds.meta.episodes["dataset_from_index"][episode]
    idx = int(from_idx) + int(frame)
    # IMPORTANT: avoid `ds[idx]` because it triggers video decoding (torchcodec/ffmpeg).
    # We only need state, which is stored in the parquet-backed hf_dataset.
    row = ds.hf_dataset[int(idx)]
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
    if state.size != 7:
        raise ValueError(f"Expected dataset observation.state to be 7D, got {state.size} at {dataset_dir} ep={episode}")
    return state


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


def load_local_policy_with_report(*, policy_class, cfg, pretrained_dir: Path):
    """Load policy weights from local `pretrained_model/` and return missing/unexpected keys."""
    model_file = pretrained_dir / SAFETENSORS_SINGLE_FILE
    if not model_file.exists():
        raise FileNotFoundError(f"Missing {SAFETENSORS_SINGLE_FILE} under {pretrained_dir}")

    # Instantiate on CPU first; load directly to device when supported.
    policy = policy_class(cfg)

    kwargs: dict[str, object] = {"strict": False}
    if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
        kwargs["device"] = cfg.device

    missing_keys, unexpected_keys = load_model_as_safetensor(policy, str(model_file), **kwargs)
    policy.to(cfg.device)
    policy.eval()

    report = WeightLoadReport(missing_keys=sorted(set(missing_keys)), unexpected_keys=sorted(set(unexpected_keys)))
    return policy, report


def run_startup_checks(*, contract, backend) -> None:
    """Fail fast on contract mismatches + basic gripper motion sanity."""
    # Cameras
    cams = backend.list_cameras()
    required_cams = {k.split(".")[-1] for k in contract.image_keys}  # ...CameraLeft -> CameraLeft
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

    # One-time agent schema sanity: verify joint/gripper name ordering (Hazel).
    if not getattr(backend, "_lerobot_schema_checked", False) and hasattr(backend, "observation_names"):
        obs_names = list(backend.observation_names() or [])
        act_names = list(backend.action_names() or []) if hasattr(backend, "action_names") else []

        def _norm(n: str) -> str:
            return "".join(ch for ch in n.lower() if ch.isalnum())

        def _expected_names() -> list[str]:
            return [f"joint{i}" for i in range(1, 7)] + ["gripper"]

        exp = [_norm(x) for x in _expected_names()]
        obs_n = [_norm(x) for x in obs_names]
        act_n = [_norm(x) for x in act_names]

        def _matches(names_norm: list[str]) -> bool:
            if len(names_norm) != 7:
                return False
            # Accept common gripper naming variants.
            gr_ok = names_norm[6] in {"gripper", "joint7", "joint_7", "finger", "fingers"}
            return names_norm[:6] == exp[:6] and gr_ok

        schema_ok = True
        if obs_names and not _matches(obs_n):
            schema_ok = False
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | SCHEMA | observation_names mismatch: {obs_names}",
                flush=True,
            )
        if act_names and not _matches(act_n):
            schema_ok = False
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | SCHEMA | action_names mismatch: {act_names}",
                flush=True,
            )

        if schema_ok and (obs_names or act_names):
            print(
                f"{datetime.now().strftime('%H:%M:%S')} | SCHEMA | OK | obs={obs_names} | act={act_names}",
                flush=True,
            )

        setattr(backend, "_lerobot_schema_checked", True)

    # Gripper sanity: open then close should change the observed gripper value directionally.
    required_cam_names = sorted({k.split(".")[-1] for k in contract.image_keys})
    state0, _, _ = _backend_reset(backend, camera_names=required_cam_names)
    # Try to identify gripper index by schema (Hazel) or fall back to last dim.
    obs_names = backend.observation_names() if hasattr(backend, "observation_names") else []
    act_names = backend.action_names() if hasattr(backend, "action_names") else []

    def _find_gripper_idx(names: list[str]) -> int | None:
        for i, n in enumerate(names):
            nl = n.lower()
            if "gripper" in nl or nl in {"joint7", "joint_7", "finger", "fingers"}:
                return i
        return None

    obs_grip_idx = _find_gripper_idx(obs_names) if obs_names else None
    act_grip_idx = _find_gripper_idx(act_names) if act_names else None

    if obs_grip_idx is None:
        obs_grip_idx = len(state0) - 1
    if act_grip_idx is None:
        act_grip_idx = backend.action_dim() - 1

    g0 = float(state0[obs_grip_idx])

    open_action = state0.copy()
    open_action[act_grip_idx] = 0.035
    # For Hazel, the agent observation may not update immediately; accept either:
    # - observation gripper value moves, OR
    # - action echo shows the new target.
    state_open, acts_open, _ = (
        backend.capture_agent_frame(timeout_s=2.0)
        if hasattr(backend, "capture_agent_frame")
        else (None, None, None)
    )
    _ = _backend_step(backend, action7=open_action, camera_names=required_cam_names)
    g_open = None
    if hasattr(backend, "capture_agent_frame"):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            obs, acts, _, = backend.capture_agent_frame(timeout_s=0.5)
            if acts is not None and acts.size > act_grip_idx and abs(float(acts[act_grip_idx]) - 0.035) > 1e-6:
                # actions echo is present but doesn't match; keep waiting
                pass
            if acts is not None and acts.size > act_grip_idx and abs(float(acts[act_grip_idx]) - 0.035) <= 5e-3:
                break
        obs, acts_open, _, = backend.capture_agent_frame(timeout_s=0.5)
        g_open = float(obs[obs_grip_idx])
    else:
        g_open = float(_backend_step(backend, action7=open_action, camera_names=required_cam_names)[0][obs_grip_idx])

    close_action = state0.copy()
    close_action[act_grip_idx] = 0.0
    g_close = None
    acts_close = None
    if hasattr(backend, "capture_agent_frame"):
        _ = _backend_step(backend, action7=close_action, camera_names=required_cam_names)
        deadline = time.time() + 1.0
        while time.time() < deadline:
            obs, acts, _, = backend.capture_agent_frame(timeout_s=0.5)
            if acts is not None and acts.size > act_grip_idx and abs(float(acts[act_grip_idx]) - 0.0) <= 5e-3:
                break
        obs, acts_close, _, = backend.capture_agent_frame(timeout_s=0.5)
        g_close = float(obs[obs_grip_idx])
    else:
        g_close = float(_backend_step(backend, action7=close_action, camera_names=required_cam_names)[0][obs_grip_idx])

    # We keep the check directional when possible, but do not hard-fail Hazel if the observation
    # doesn't move even though action echo updated (common in paused/slow sim).
    if g_open is None or g_close is None:
        return
    acts_ok_open = (
        acts_open is not None
        and acts_open.size > act_grip_idx
        and abs(float(acts_open[act_grip_idx]) - 0.035) <= 5e-3
    )
    acts_ok_close = (
        acts_close is not None
        and acts_close.size > act_grip_idx
        and abs(float(acts_close[act_grip_idx]) - 0.0) <= 5e-3
    )

    if not (g_open > g0 or g_close < g0):
        if acts_ok_open and acts_ok_close:
            return
        raise ValueError(
            "Gripper sanity check failed (no directional change). "
            f"g0={g0:.6f} g_open={g_open:.6f} g_close={g_close:.6f}"
        )


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
) -> EpisodeMetrics:
    cam_names = sorted({k.split(".")[-1] for k in contract.image_keys})
    state0, images0, _ = _backend_reset(backend, camera_names=cam_names)

    # Per-try reset: always move to dataset-home pose before ACT reset, so every try starts the same.
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
    images_chw = {
        f"observation.images.{cam}": (3, h, w) for cam in backend.list_cameras()
    }
    assert_expected_keys_and_shapes(
        contract,
        state_dim=backend.state_dim(),
        action_dim=backend.action_dim(),
        images_chw=images_chw,
    )

    gripper_trace: list[float] = [float(state0[-1])]

    state = state0
    images = images0
    for _ in range(int(episode_steps)):
        batch: dict[str, object] = {}
        batch[contract.state_key] = torch.from_numpy(state).to(dtype=torch.float32)

        for k in contract.image_keys:
            cam_name = k.split(".")[-1]  # observation.images.CameraLeft -> CameraLeft
            img_hwc = images[cam_name]
            img_chw = _hwc_uint8_to_chw_float01(img_hwc)
            batch[k] = torch.from_numpy(img_chw).to(dtype=torch.float32)

        if task is not None:
            batch["task"] = task

        obs_t = preprocessor(batch)
        with torch.no_grad():
            action_norm = policy.select_action(obs_t)
            action = postprocessor(action_norm)

        # Convert (B,7) -> (7,)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)

        state, images, _ = _backend_step(backend, action7=action_np, camera_names=cam_names)
        gripper_trace.append(float(state[-1]))

    return EpisodeMetrics(
        steps=int(episode_steps),
        gripper_transitions=_count_transitions(gripper_trace),
        gripper_min=float(np.min(gripper_trace)),
        gripper_max=float(np.max(gripper_trace)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ACT checkpoints against a LuckyEngine backend.")
    parser.add_argument(
        "--checkpoints_root",
        type=Path,
        default=Path("lerobot/outputs/train/act_piper_01192026/checkpoints"),
        help="Folder containing numbered checkpoint subfolders (each has pretrained_model/).",
    )
    parser.add_argument("--backend", choices=["mock", "hazel"], default="mock", help="Backend implementation to use.")
    parser.add_argument("--hazel_address", type=str, default="127.0.0.1:50051", help="Hazel gRPC host:port.")
    parser.add_argument(
        "--hazel_stubs_dir",
        type=Path,
        default=None,
        help="Directory containing generated Hazel python stubs (e.g. *_pb2.py, *_pb2_grpc.py).",
    )
    parser.add_argument("--hazel_agent_name", type=str, default="PiperAgent", help="Agent name in Hazel.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--task", type=str, default=None, help="Optional task string (if policy expects it).")
    parser.add_argument("--episode_steps", type=int, default=200, help="Steps per episode.")
    parser.add_argument("--tries_per_ckpt", type=int, default=1, help="Number of episodes per checkpoint.")
    parser.add_argument("--out_dir", type=Path, default=Path("lerobot/outputs/sweeps/act_piper_room"), help="Output dir.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only run the first N checkpoints.")
    parser.add_argument(
        "--home_action",
        type=str,
        default=None,
        help='Optional 7-float joint+gripper home target, e.g. "0,0,0,0,0,0,0.035".',
    )
    parser.add_argument(
        "--home_from_dataset",
        type=Path,
        default=None,
        help="Optional local LeRobot dataset directory (e.g. lerobot/dataset/session_256) to read home from episode-0 frame-0 observation.state.",
    )
    parser.add_argument("--home_dataset_episode", type=int, default=0)
    parser.add_argument("--home_dataset_frame", type=int, default=0)
    parser.add_argument("--home_settle_steps", type=int, default=50, help="How many control steps to hold home pose.")
    parser.add_argument("--home_settle_sleep_s", type=float, default=0.02, help="Sleep between settle steps.")
    parser.add_argument(
        "--skip_startup_checks",
        action="store_true",
        help="Skip startup contract + gripper sanity checks (not recommended).",
    )
    args = parser.parse_args()

    home_action7 = None
    if args.home_action is not None:
        home_action7 = _parse_action7(args.home_action)
    elif args.home_from_dataset is not None:
        home_action7 = _load_home_from_dataset(
            args.home_from_dataset,
            episode=args.home_dataset_episode,
            frame=args.home_dataset_frame,
        )

    ckpt_dirs = iter_checkpoints(args.checkpoints_root)
    if args.limit and args.limit > 0:
        ckpt_dirs = ckpt_dirs[: args.limit]
    if not ckpt_dirs:
        raise SystemExit(f"No checkpoints found under {args.checkpoints_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "summary.csv"
    summary_jsonl = args.out_dir / "summary.jsonl"

    total_tries = int(len(ckpt_dirs) * int(args.tries_per_ckpt))
    tries_done = 0
    t_global0 = time.perf_counter()

    def _fmt_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h{m:02d}m"
        return f"{m:d}m{s:02d}s"

    def log_line(checkpoint: str, try_idx: int | None, stage: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        if try_idx is None:
            prefix = f"{ts} | ckpt {checkpoint} | {stage}"
        else:
            prefix = f"{ts} | ckpt {checkpoint} | try {try_idx+1}/{int(args.tries_per_ckpt)} | {stage}"
        print(f"{prefix} | {msg}", flush=True)

    backend = None
    try:
        if args.backend == "mock":
            backend = MockPiperRoomBackend(width=320, height=240, seed=0)
        elif args.backend == "hazel":
            if args.hazel_stubs_dir is None:
                raise SystemExit("--hazel_stubs_dir is required when --backend hazel")
            backend = HazelGrpcBackend(
                address=args.hazel_address,
                stubs_dir=args.hazel_stubs_dir,
                agent_name=args.hazel_agent_name,
                image_width=320,
                image_height=240,
            )
            backend.connect()
        else:
            raise SystemExit(f"Unsupported backend: {args.backend}")

        with summary_csv.open("w", newline="", encoding="utf-8") as f_csv, summary_jsonl.open(
            "w", encoding="utf-8"
        ) as f_jsonl:
            writer = csv.DictWriter(
                f_csv,
                fieldnames=[
                    "checkpoint",
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
            writer.writeheader()

            for ckpt_i, pretrained_dir in enumerate(ckpt_dirs, start=1):
                checkpoint_name = pretrained_dir.parent.name
                contract = load_policy_contract_from_pretrained(pretrained_dir)
                log_line(
                    checkpoint_name,
                    None,
                    "LOAD",
                    f"{ckpt_i}/{len(ckpt_dirs)} checkpoints | out={args.out_dir}",
                )

                if contract.policy_type != "act":
                    raise ValueError(
                        f"Expected ACT checkpoints, got policy type `{contract.policy_type}` at {pretrained_dir}"
                    )

                if not args.skip_startup_checks:
                    log_line(checkpoint_name, None, "CHECK", "startup contract + gripper sanity")
                    run_startup_checks(contract=contract, backend=backend)

                policy_class = get_policy_class(contract.policy_type)
                cfg = load_act_config_from_pretrained_compat(pretrained_dir, device=args.device)
                policy, weight_report = load_local_policy_with_report(
                    policy_class=policy_class, cfg=cfg, pretrained_dir=pretrained_dir
                )

                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy.config,
                    pretrained_path=str(pretrained_dir),
                    preprocessor_overrides={"device_processor": {"device": args.device}},
                    postprocessor_overrides={"device_processor": {"device": "cpu"}},
                )
                log_line(
                    checkpoint_name,
                    None,
                    "READY",
                    f"weights mismatch missing={len(weight_report.missing_keys)} unexpected={len(weight_report.unexpected_keys)} | policy_device={args.device}",
                )

                for try_idx in range(int(args.tries_per_ckpt)):
                    tries_done += 1
                    avg_s = (time.perf_counter() - t_global0) / max(tries_done - 1, 1)
                    eta = _fmt_eta((total_tries - (tries_done - 1)) * avg_s)
                    home_mode = (
                        "home=on"
                        if (args.home_action is not None or args.home_from_dataset is not None)
                        else "home=off"
                    )
                    log_line(
                        checkpoint_name,
                        try_idx,
                        "RUN",
                        f"progress {tries_done-1}/{total_tries} | ETA~{eta} | steps={args.episode_steps} | {home_mode}",
                    )
                    t0 = time.perf_counter()
                    error = None
                    metrics = None
                    try:
                        metrics = run_episode(
                            policy=policy,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            contract=contract,
                            backend=backend,
                            episode_steps=args.episode_steps,
                            task=args.task,
                            home_action7=home_action7,
                            home_settle_steps=args.home_settle_steps,
                            home_settle_sleep_s=args.home_settle_sleep_s,
                        )
                    except Exception as e:  # noqa: BLE001 (example script)
                        error = f"{type(e).__name__}: {e}"
                    elapsed = time.perf_counter() - t0
                    ok = "OK" if error is None else "ERR"
                    if metrics is not None:
                        log_line(
                            checkpoint_name,
                            try_idx,
                            ok,
                            f"elapsed={elapsed:.2f}s | grip_transitions={metrics.gripper_transitions} | grip_range=[{metrics.gripper_min:.4f},{metrics.gripper_max:.4f}]",
                        )
                    else:
                        log_line(checkpoint_name, try_idx, ok, f"elapsed={elapsed:.2f}s | error={error}")

                    row = {
                        "checkpoint": checkpoint_name,
                        "try_idx": try_idx,
                        "steps": getattr(metrics, "steps", None),
                        "gripper_transitions": getattr(metrics, "gripper_transitions", None),
                        "gripper_min": getattr(metrics, "gripper_min", None),
                        "gripper_max": getattr(metrics, "gripper_max", None),
                        "missing_keys_n": len(weight_report.missing_keys),
                        "unexpected_keys_n": len(weight_report.unexpected_keys),
                        "missing_keys_head": ";".join(weight_report.missing_keys[:10]),
                        "unexpected_keys_head": ";".join(weight_report.unexpected_keys[:10]),
                        "error": error,
                        "elapsed_s": round(elapsed, 6),
                    }
                    writer.writerow(row)
                    f_jsonl.write(json.dumps(row) + "\n")
                    f_jsonl.flush()
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()


