from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray  # (7,), float32
    images_hwc_uint8: dict[str, np.ndarray]  # camera_name -> (H,W,3) uint8
    timestamps_s: dict[str, float]  # "state" + per camera timestamps


class MockPiperRoomBackend:
    """A tiny stand-in for LuckyEngine/Hazel to validate end-to-end wiring.

    - Observation: 7 floats (joint1..joint6, gripper)
    - Action: 7 floats (absolute targets)
    - Cameras: CameraLeft, CameraGripper, returned as RGB uint8 HWC
    """

    def __init__(self, *, width: int = 320, height: int = 240, seed: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.rng = np.random.default_rng(int(seed))

        self._state = np.zeros((7,), dtype=np.float32)
        self._t = 0

    # ---- Contract-ish metadata -------------------------------------------------

    def list_cameras(self) -> list[str]:
        return ["CameraLeft", "CameraGripper"]

    def state_dim(self) -> int:
        return 7

    def action_dim(self) -> int:
        return 7

    # ---- Simulation-ish API ----------------------------------------------------

    def reset(self, *, seed: int | None = None) -> StepResult:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self._state[...] = 0.0
        self._t = 0
        return self._make_step_result()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {action.shape}")

        # Smooth towards target.
        alpha = 0.35
        self._state = (1 - alpha) * self._state + alpha * action
        self._t += 1
        return self._make_step_result()

    # ---- Internals -------------------------------------------------------------

    def _make_step_result(self) -> StepResult:
        now = time.time()
        images = {
            "CameraLeft": self._random_rgb(),
            "CameraGripper": self._random_rgb(),
        }
        ts = {"state": now, "CameraLeft": now, "CameraGripper": now}
        return StepResult(state=self._state.copy(), images_hwc_uint8=images, timestamps_s=ts)

    def _random_rgb(self) -> np.ndarray:
        # (H,W,3) uint8
        return self.rng.integers(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)


