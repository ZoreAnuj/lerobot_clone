from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import grpc
import numpy as np


@dataclass(frozen=True)
class HazelModules:
    pb2: ModuleType
    pb2_grpc: ModuleType


def _module_to_relpath(module_name: str) -> Path:
    """Convert a python module name to a relative file path.

    Examples:
      - "hazel_rpc_pb2" -> "hazel_rpc_pb2.py"
      - "luckyrobots.hazel_rpc_pb2" -> "luckyrobots/hazel_rpc_pb2.py"
    """
    parts = [p for p in str(module_name).split(".") if p]
    if not parts:
        raise ValueError("Empty module name")
    return Path(*parts).with_suffix(".py")


def _normalize_stubs_root(stubs_dir: Path, *, module_name: str) -> Path:
    """Ensure sys.path can import `module_name` (including package modules).

    If stubs_dir points at the package directory itself (e.g. .../src/luckyrobots) and
    module_name is "luckyrobots.*", we must add the parent directory to sys.path.
    """
    stubs_dir = Path(stubs_dir)
    if "." in str(module_name):
        top_pkg = str(module_name).split(".", 1)[0]
        if stubs_dir.name == top_pkg:
            return stubs_dir.parent
    return stubs_dir


def _resolve_stubs_dir(stubs_dir: Path, *, pb2_module: str, pb2_grpc_module: str) -> Path:
    """Best-effort resolution of where the generated python stubs actually live.

    Users often point this at a repo root (or at an empty folder). We try to find a directory that
    contains both `<pb2_module>.py` and `<pb2_grpc_module>.py`.
    """
    stubs_dir = Path(stubs_dir)
    pb2_rel = _module_to_relpath(pb2_module)
    pb2_grpc_rel = _module_to_relpath(pb2_grpc_module)

    # Fast path: exact directory contains the files.
    if stubs_dir.is_dir() and (stubs_dir / pb2_rel).exists() and (stubs_dir / pb2_grpc_rel).exists():
        return stubs_dir

    # Common layout: <something>/luckyrobots or <something>/luckyrobots/src/luckyrobots
    candidates = [
        stubs_dir,
        stubs_dir / "luckyrobots",
        stubs_dir / "src" / "luckyrobots",
        stubs_dir.parent / "luckyrobots",
        stubs_dir.parent / "src" / "luckyrobots",
        stubs_dir.parent.parent / "luckyrobots",
        stubs_dir.parent.parent / "src" / "luckyrobots",
    ]
    for c in candidates:
        try:
            if c.is_dir() and (c / pb2_rel).exists() and (c / pb2_grpc_rel).exists():
                return c
        except OSError:
            continue

    # Slow path: recursive search under the provided directory.
    try:
        matches = list(stubs_dir.rglob(pb2_rel.name)) if stubs_dir.is_dir() else []
    except OSError:
        matches = []
    for m in matches:
        parent = m.parent
        # Candidate roots: either the folder containing the file, or its parent (if the file is inside a package dir).
        for root in (parent, parent.parent):
            try:
                if root.is_dir() and (root / pb2_rel).exists() and (root / pb2_grpc_rel).exists():
                    return root
            except OSError:
                continue

    # If we can find a sibling at the repo root (common in your setup), mention it in the error.
    hint = ""
    sibling = stubs_dir.parent.parent / "luckyrobots"
    if sibling.is_dir() and (sibling / pb2_rel).exists() and (sibling / pb2_grpc_rel).exists():
        hint = f" (hint: your stubs appear to be in {sibling})"

    raise FileNotFoundError(
        "Could not locate Hazel python stubs. Expected to find both "
        f"`{pb2_rel.as_posix()}` and `{pb2_grpc_rel.as_posix()}` under `{stubs_dir}` (or common subfolders).{hint}"
    )


def _import_from_dir(stubs_dir: Path, module_name: str) -> ModuleType:
    stubs_dir = _normalize_stubs_root(Path(stubs_dir), module_name=module_name)
    if not stubs_dir.exists():
        raise FileNotFoundError(f"gRPC stubs dir not found: {stubs_dir}")
    if str(stubs_dir) not in sys.path:
        sys.path.insert(0, str(stubs_dir))
    return importlib.import_module(module_name)


def _safe_set_fields(msg: Any, **kwargs: Any) -> Any:
    """Set fields that exist on a protobuf message; ignore unknown ones."""
    desc = getattr(msg, "DESCRIPTOR", None)
    fields_by_name = getattr(desc, "fields_by_name", {}) if desc is not None else {}
    for k, v in kwargs.items():
        if k in fields_by_name:
            field = fields_by_name[k]
            # Protobuf Python disallows direct assignment to repeated fields; use extend.
            if getattr(field, "label", None) == field.LABEL_REPEATED:
                container = getattr(msg, k)
                # Clear then extend for deterministic behavior.
                try:
                    del container[:]  # RepeatedScalarContainer supports slice deletion
                except Exception:  # noqa: BLE001
                    try:
                        container.clear()
                    except Exception:  # noqa: BLE001
                        pass
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    container.extend(v)
                else:
                    container.append(v)
            else:
                setattr(msg, k, v)
    return msg


def _make_request(pb2: ModuleType, method_name: str, **kwargs: Any) -> Any:
    """Best-effort request constructor for common proto naming conventions."""
    candidates = [
        f"{method_name}Request",
        f"{method_name}Req",
        "Request",
        "Empty",
    ]
    for cls_name in candidates:
        cls = getattr(pb2, cls_name, None)
        if cls is None:
            continue
        req = cls()
        return _safe_set_fields(req, **kwargs)
    raise AttributeError(
        f"Could not build request for `{method_name}`. Looked for {candidates} in module {pb2.__name__}."
    )


class _LatestFrame:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rgb_hwc_uint8: np.ndarray | None = None
        self.timestamp_s: float | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.error: str | None = None


class HazelCameraStream:
    """Background reader for CameraService.StreamCamera.

    This implementation is intentionally defensive and will require minor tweaks
    once we see the exact Hazel proto field names in your stubs.
    """

    def __init__(
        self,
        *,
        stub: Any,
        pb2: ModuleType,
        camera_name: str,
        width: int,
        height: int,
        fps: int | None = None,
    ) -> None:
        self._stub = stub
        self._pb2 = pb2
        self._camera_name = camera_name
        self._width = int(width)
        self._height = int(height)
        self._fps = fps

        self._latest = _LatestFrame()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"HazelCameraStream[{camera_name}]")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self) -> tuple[np.ndarray | None, float | None, str | None]:
        with self._latest.lock:
            return self._latest.rgb_hwc_uint8, self._latest.timestamp_s, self._latest.error

    def _run(self) -> None:
        try:
            req = _make_request(
                self._pb2,
                "StreamCamera",
                name=self._camera_name,
                width=self._width,
                height=self._height,
                target_fps=self._fps or 0,
                format="RGBA",
            )
            stream = self._stub.StreamCamera(req)
            for msg in stream:
                if self._stop.is_set():
                    break

                # Hazel ImageFrame:
                # - data: bytes
                # - width/height/channels
                # - timestamp_ms
                w = getattr(msg, "width", None)
                h = getattr(msg, "height", None)
                c = getattr(msg, "channels", None)
                data = getattr(msg, "data", None)
                ts_ms = getattr(msg, "timestamp_ms", None)

                if w is None or h is None or data is None:
                    continue

                buf = np.frombuffer(data, dtype=np.uint8)
                if c == 4 or buf.size == int(w) * int(h) * 4:
                    rgba = buf.reshape((int(h), int(w), 4))
                    rgb = rgba[:, :, :3].copy()
                elif c == 3 or buf.size == int(w) * int(h) * 3:
                    rgb = buf.reshape((int(h), int(w), 3)).copy()
                else:
                    continue

                with self._latest.lock:
                    self._latest.rgb_hwc_uint8 = rgb
                    self._latest.timestamp_s = (
                        float(ts_ms) / 1000.0 if ts_ms is not None else time.time()
                    )
                    self._latest.width = int(w)
                    self._latest.height = int(h)
        except Exception as e:  # noqa: BLE001
            with self._latest.lock:
                self._latest.error = f"{type(e).__name__}: {e}"


class HazelAgentStream:
    """Background reader for AgentService.StreamAgent -> latest AgentFrame.observations."""

    def __init__(self, *, stub: Any, pb2: ModuleType, agent_name: str, fps: int = 60) -> None:
        self._stub = stub
        self._pb2 = pb2
        self._agent_name = agent_name
        self._fps = int(fps)

        self._lock = threading.Lock()
        self._latest_obs: np.ndarray | None = None
        self._latest_actions: np.ndarray | None = None
        self._latest_ts_s: float | None = None
        self._error: str | None = None

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"HazelAgentStream[{agent_name}]")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str | None]:
        with self._lock:
            return self._latest_obs, self._latest_actions, self._latest_ts_s, self._error

    def _run(self) -> None:
        try:
            req = _make_request(self._pb2, "StreamAgent", agent_name=self._agent_name, target_fps=self._fps)
            stream = self._stub.StreamAgent(req)
            for msg in stream:
                if self._stop.is_set():
                    break
                obs = getattr(msg, "observations", None)
                acts = getattr(msg, "actions", None)
                ts_ms = getattr(msg, "timestamp_ms", None)
                if obs is None:
                    continue
                arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                with self._lock:
                    self._latest_obs = arr
                    self._latest_actions = (
                        np.asarray(acts, dtype=np.float32).reshape(-1) if acts is not None else None
                    )
                    self._latest_ts_s = float(ts_ms) / 1000.0 if ts_ms is not None else time.time()
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._error = f"{type(e).__name__}: {e}"


class HazelGrpcBackend:
    """LuckyEngine/Hazel gRPC backend.

    You must provide a directory containing generated python stubs (pb2 / pb2_grpc).
    Default module names are best-effort and may need flags adjusted to match your Hazel repo.
    """

    def __init__(
        self,
        *,
        address: str,
        stubs_dir: str | Path,
        pb2_module: str = "hazel_rpc_pb2",
        pb2_grpc_module: str = "hazel_rpc_pb2_grpc",
        agent_name: str = "PiperAgent",
        image_width: int = 320,
        image_height: int = 240,
    ) -> None:
        self.address = address
        self.stubs_dir = Path(stubs_dir)
        self.agent_name = agent_name
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self._pb2_module = pb2_module
        self._pb2_grpc_module = pb2_grpc_module

        self._channel: grpc.Channel | None = None
        self._mods: HazelModules | None = None
        self._agent_stub: Any = None
        self._camera_stub: Any = None
        self._mujoco_stub: Any = None
        self._scene_stub: Any = None
        self._streams: dict[str, HazelCameraStream] = {}
        self._agent_stream: HazelAgentStream | None = None
        self._agent_schema: Any = None

    def connect(self) -> None:
        self._channel = grpc.insecure_channel(self.address)
        # Auto-resolve stubs dir in case a repo root (or empty folder) was provided.
        self.stubs_dir = _resolve_stubs_dir(
            self.stubs_dir, pb2_module=self._pb2_module, pb2_grpc_module=self._pb2_grpc_module
        )
        # Ensure we import package stubs correctly (relative imports inside *_pb2_grpc.py).
        self.stubs_dir = _normalize_stubs_root(self.stubs_dir, module_name=self._pb2_grpc_module)
        mods = HazelModules(
            pb2=_import_from_dir(self.stubs_dir, self._pb2_module),
            pb2_grpc=_import_from_dir(self.stubs_dir, self._pb2_grpc_module),
        )
        self._mods = mods

        # Stubs: conventional naming `<ServiceName>Stub`.
        self._agent_stub = getattr(mods.pb2_grpc, "AgentServiceStub")(self._channel)
        self._camera_stub = getattr(mods.pb2_grpc, "CameraServiceStub")(self._channel)
        self._mujoco_stub = getattr(mods.pb2_grpc, "MujocoServiceStub")(self._channel)
        self._scene_stub = getattr(mods.pb2_grpc, "SceneServiceStub")(self._channel)

        # Cache agent schema (names + dims)
        req = _make_request(mods.pb2, "GetAgentSchema", agent_name=self.agent_name)
        resp = self._agent_stub.GetAgentSchema(req)
        self._agent_schema = getattr(resp, "schema", None)

    # ---- Scene helpers ---------------------------------------------------------

    def list_entities(self) -> list[dict[str, object]]:
        """Return a list of entities with id/name/position if SceneService supports it."""
        if self._mods is None:
            raise RuntimeError("Call connect() first.")
        if self._scene_stub is None:
            return []
        try:
            req = _make_request(self._mods.pb2, "ListEntities")
            resp = self._scene_stub.ListEntities(req)
        except Exception:  # noqa: BLE001
            return []

        entities = getattr(resp, "entities", None) or getattr(resp, "entity_infos", None)
        if entities is None:
            return []

        out: list[dict[str, object]] = []
        for e in list(entities):
            name = getattr(e, "name", None)
            eid = getattr(e, "id", None)
            # id may be nested EntityId { id: uint64 }
            if eid is not None and hasattr(eid, "id"):
                eid_val = int(getattr(eid, "id"))
            else:
                try:
                    eid_val = int(eid) if eid is not None else None
                except Exception:  # noqa: BLE001
                    eid_val = None

            tr = getattr(e, "transform", None)
            pos = getattr(tr, "position", None) if tr is not None else None
            px = float(getattr(pos, "x")) if pos is not None and hasattr(pos, "x") else None
            py = float(getattr(pos, "y")) if pos is not None and hasattr(pos, "y") else None
            pz = float(getattr(pos, "z")) if pos is not None and hasattr(pos, "z") else None

            out.append({"id": eid_val, "name": name, "pos": (px, py, pz)})
        return out

    def find_entity_id(self, *, name_substr: str, prefer_exact: str | None = None) -> int | None:
        """Find the first entity whose name matches (exact preferred, else substring match)."""
        ents = self.list_entities()
        if not ents:
            return None
        if prefer_exact:
            for e in ents:
                if str(e.get("name", "")).lower() == str(prefer_exact).lower():
                    return int(e["id"]) if e.get("id") is not None else None
        needle = str(name_substr).lower()
        for e in ents:
            if needle in str(e.get("name", "")).lower():
                return int(e["id"]) if e.get("id") is not None else None
        return None

    def get_entity_pos(self, entity_id: int) -> tuple[float, float, float] | None:
        """Get entity world position via GetEntity if available; fall back to ListEntities snapshot."""
        if self._mods is None:
            raise RuntimeError("Call connect() first.")
        if self._scene_stub is None:
            return None

        # Try GetEntity first (more direct)
        try:
            req = _make_request(self._mods.pb2, "GetEntity", id=int(entity_id))
            resp = self._scene_stub.GetEntity(req)
            e = getattr(resp, "entity", None) or getattr(resp, "info", None) or resp
            tr = getattr(e, "transform", None)
            pos = getattr(tr, "position", None) if tr is not None else None
            if pos is not None and hasattr(pos, "x"):
                return float(pos.x), float(pos.y), float(pos.z)
        except Exception:  # noqa: BLE001
            pass

        # Fallback: scan ListEntities output
        for e in self.list_entities():
            if e.get("id") == int(entity_id):
                px, py, pz = e.get("pos", (None, None, None))
                if px is None or py is None or pz is None:
                    return None
                return float(px), float(py), float(pz)
        return None

    def close(self) -> None:
        for s in self._streams.values():
            s.stop()
        self._streams.clear()
        if self._agent_stream is not None:
            self._agent_stream.stop()
            self._agent_stream = None
        if self._channel is not None:
            self._channel.close()
        self._channel = None

    # ---- Contract-ish metadata -------------------------------------------------

    def list_cameras(self) -> list[str]:
        if self._mods is None:
            raise RuntimeError("Call connect() first.")
        req = _make_request(self._mods.pb2, "ListCameras")
        resp = self._camera_stub.ListCameras(req)
        cameras = getattr(resp, "cameras", None) or getattr(resp, "names", None)
        if cameras is None:
            raise AttributeError("ListCameras response has no `cameras`/`names` field.")
        # hazel_rpc.proto: cameras is repeated CameraInfo { id, name }
        if len(cameras) > 0 and hasattr(cameras[0], "name"):
            return [c.name for c in cameras]
        return list(cameras)

    def state_dim(self) -> int:
        if self._agent_schema is not None and hasattr(self._agent_schema, "observation_size"):
            return int(self._agent_schema.observation_size)
        return 7

    def action_dim(self) -> int:
        if self._agent_schema is not None and hasattr(self._agent_schema, "action_size"):
            return int(self._agent_schema.action_size)
        return 7

    def observation_names(self) -> list[str]:
        if self._agent_schema is not None and hasattr(self._agent_schema, "observation_names"):
            return list(self._agent_schema.observation_names)
        return []

    def action_names(self) -> list[str]:
        if self._agent_schema is not None and hasattr(self._agent_schema, "action_names"):
            return list(self._agent_schema.action_names)
        return []

    # ---- Streams / frames ------------------------------------------------------

    def start_camera_streams(self, camera_names: list[str]) -> None:
        if self._mods is None:
            raise RuntimeError("Call connect() first.")
        for name in camera_names:
            if name in self._streams:
                continue
            s = HazelCameraStream(
                stub=self._camera_stub,
                pb2=self._mods.pb2,
                camera_name=name,
                width=self.image_width,
                height=self.image_height,
            )
            s.start()
            self._streams[name] = s

        if self._agent_stream is None:
            self._agent_stream = HazelAgentStream(
                stub=self._agent_stub, pb2=self._mods.pb2, agent_name=self.agent_name, fps=60
            )
            self._agent_stream.start()

    def capture_agent_frame(
        self, *, timeout_s: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray | None, float]:
        """Return (observations, actions_echo_or_none, timestamp_s) from AgentService.StreamAgent."""
        if self._agent_stream is None:
            raise RuntimeError("Agent stream not started. Call start_camera_streams() first.")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            obs, acts, ts, err = self._agent_stream.latest()
            if err:
                raise RuntimeError(f"Agent stream error: {err}")
            if obs is not None:
                if obs.size != self.state_dim():
                    raise ValueError(f"Expected {self.state_dim()}D state, got {obs.size}")
                return obs.astype(np.float32, copy=False), acts, float(ts) if ts is not None else time.time()
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for first agent observation frame.")

    def capture_state(self, *, timeout_s: float = 2.0) -> np.ndarray:
        obs, _, _ = self.capture_agent_frame(timeout_s=timeout_s)
        return obs

    def set_action(self, action7: np.ndarray) -> None:
        if self._mods is None:
            raise RuntimeError("Call connect() first.")
        action7 = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action7.size != self.action_dim():
            raise ValueError(f"Expected {self.action_dim()}D action, got {action7.size}")

        req = _make_request(
            self._mods.pb2,
            "SetAgentActions",
            agent_name=self.agent_name,
            actions=list(float(x) for x in action7.tolist()),
        )
        _ = self._agent_stub.SetAgentActions(req)

    # ---- Reset -----------------------------------------------------------------

    def reset_scene(self) -> None:
        # Best-effort: some Hazel builds might expose a reset method; otherwise reset is engine-side.
        if self._mods is None:
            return
        for stub in (self._mujoco_stub, self._scene_stub):
            if stub is None:
                continue
            for method_name in ("ResetScene", "Reset", "ResetSimulation"):
                fn = getattr(stub, method_name, None)
                if fn is None:
                    continue
                try:
                    req = _make_request(self._mods.pb2, method_name)
                    _ = fn(req)
                    return
                except Exception:  # noqa: BLE001
                    pass
        return

    # ---- Public step API -------------------------------------------------------

    def reset(self, *, camera_names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        self.reset_scene()
        self.start_camera_streams(camera_names)
        obs, _acts, agent_ts_s = self.capture_agent_frame()
        state = obs
        imgs, ts = self._get_images(camera_names)
        ts["agent"] = float(agent_ts_s)
        ts["state"] = time.time()
        return state, imgs, ts

    def step(
        self, *, action7: np.ndarray, camera_names: list[str]
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
        self.set_action(action7)
        obs, _acts, agent_ts_s = self.capture_agent_frame()
        state = obs
        imgs, ts = self._get_images(camera_names)
        ts["agent"] = float(agent_ts_s)
        ts["state"] = time.time()
        return state, imgs, ts

    def _get_images(
        self, camera_names: list[str], *, timeout_s: float = 2.0
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Get latest frames for the requested cameras, waiting briefly for the first frame."""
        imgs: dict[str, np.ndarray] = {}
        ts: dict[str, float] = {}

        deadline = time.time() + float(timeout_s)
        pending = set(camera_names)
        while pending and time.time() < deadline:
            for cam in list(pending):
                s = self._streams.get(cam)
                if s is None:
                    pending.remove(cam)
                    continue
                img, t, err = s.latest()
                if err:
                    raise RuntimeError(f"Camera stream error for {cam}: {err}")
                if img is None:
                    continue
                imgs[cam] = img
                ts[cam] = float(t) if t is not None else time.time()
                pending.remove(cam)
            if pending:
                time.sleep(0.01)

        if pending:
            raise TimeoutError(f"Timed out waiting for first frame(s) from: {sorted(pending)}")

        return imgs, ts


