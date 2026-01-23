# LuckyEngine / Hazel sweep runner

This folder contains a checkpoint sweep runner that can evaluate LeRobot ACT checkpoints against:

- a **mock backend** (no engine required, used for wiring sanity), or
- a **Hazel/LuckyEngine gRPC backend** (real engine).

## 1) Mock sanity check (works in this repo)

From repo root:

```bash
python lerobot/examples/luckyengine/sweep_act_piper_room_checkpoints.py --backend mock --device cpu --limit 1 --episode_steps 5
```

Outputs are written to:

- `lerobot/outputs/sweeps/act_piper_room/summary.csv`
- `lerobot/outputs/sweeps/act_piper_room/summary.jsonl`

## 2) Real LuckyEngine run (Hazel gRPC)

### Prerequisites

- LuckyEngine running with Hazel gRPC services available.
- Python gRPC stubs generated from Hazel `.proto` files (you likely already have these in your LuckyEngine repo).

### Generate python stubs (example)

In the repo that contains Hazel `.proto` files:

```bash
python -m grpc_tools.protoc -I . --python_out . --grpc_python_out . path/to/agent_service.proto path/to/camera_service.proto path/to/mujoco_service.proto path/to/scene_service.proto
```

This should create files like:

- `agent_service_pb2.py`, `agent_service_pb2_grpc.py`
- `camera_service_pb2.py`, `camera_service_pb2_grpc.py`
- `mujoco_service_pb2.py`, `mujoco_service_pb2_grpc.py`
- `scene_service_pb2.py`, `scene_service_pb2_grpc.py`

### Run the sweep

```bash
python lerobot/examples/luckyengine/sweep_act_piper_room_checkpoints.py ^
  --backend hazel ^
  --hazel_address 127.0.0.1:50051 ^
  --hazel_stubs_dir D:\path\to\generated_stubs ^
  --hazel_agent_name PiperAgent ^
  --device cuda ^
  --limit 1 ^
  --episode_steps 200
```

## Notes

- The Hazel backend (`lerobot/src/lerobot/luckyengine/hazel_backend.py`) is intentionally **defensive** and may require
  small adjustments once we see the exact field names in your Hazel proto (frame bytes field names, request message names, etc.).
- The script runs **startup contract checks** (policy expects obs/action dims + camera keys + image shapes) and a basic
  **gripper motion sanity test** to fail fast before sweeping many checkpoints.


