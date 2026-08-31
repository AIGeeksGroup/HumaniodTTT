# HumaniodTTT: Test-Time Capability Reuse for Efficient Humanoid Control

HumanoidTTT reuses qualified motion references when the current robot state lies
inside a motion's applicability region. New candidates are adapted in a
low-dimensional latent space and checked before admission to a finite-capacity
memory. A store-conditioned value network selects retention actions and updates
from subsequent reuse feedback.

## Installation

Python 3.10 or later, NumPy, and PyTorch are required.

```bash
pip install -e .
```

## Method modules

| Module | Role |
| --- | --- |
| `types.py` | Motion reference and memory-entry interfaces. |
| `motion_features.py` | The 16-dimensional motion descriptor used by the admission policy. |
| `entry_features.py` | The 45-dimensional state representation for motion applicability. |
| `signatures.py` | Request, trajectory, controller-rate, and history compatibility signatures. |
| `certificates.py` | Local-cell certificate construction, qualification checks, and membership decisions. |
| `memory.py` | Skip/replace actions, probe construction, and immediate reuse utility. |
| `consolidation.py` | The 71–128–64–1 action scorer and online Double-DQN updates. |
| `latent.py` | Low-rank noise steering, adaptation state, and performance-preservation thresholds. |
| `sac.py` | Actor, twin critics, replay buffer, SAC updates, and conservative action gating. |
| `hashing.py` | Deterministic hashes for motion arrays, signatures, and certificates. |

## Data interfaces

Motion references use `float32[60, 36]` at 30 Hz: root position, root quaternion
in `wxyz` order, and 29 G1 joint positions. Stored motion archives expose a
`qpos_36` array with this layout. Entry features use a 32-step tracker history;
`FEATURE_NAMES` defines their order and units are encoded in field names.

The caller provides the motion generator, HoloMotion execution runtime, motion
archives, qualification observations, and policy weights. The upstream
[OMG](https://github.com/Tsinghua-MARS-Lab/OMG) and
[HoloMotion](https://github.com/HorizonRobotics/HoloMotion) projects provide the
generator and tracking stack. This package contains algorithm components;
application launchers and model assets are distributed separately upstream.

`capture_entry_features` accepts a runtime exposing `current_qpos`,
`current_qvel`, `recoverability_snapshot`, `audit_state`, `_contacts`,
`joint_position_limits`, and an active `sequence` containing MuJoCo state,
body IDs, and the previous action.

## Applicability and qualification

`build_certificate(package, reference, motion_build, geometry)` constructs a
union of local cells from successful replay endpoints. `motion_build` supplies
endpoint features and replay outcomes; `geometry` supplies shared feature scales,
the normalized RMS radius, and the minimum safety margin.

`request_from_entry` combines entry features with a request signature and physical
checks. `evaluate_membership` accepts a request when its signature matches, its
physical checks pass, and its feature vector belongs to a local cell.
`apply_frozen_compatibility` additionally checks semantic eligibility, entry-state
thresholds, replay coverage, and certificate geometry.

Certificate construction consumes qualification observations. The caller runs
the physical replays and completes independent verification before activating
a memory entry. Array, signature, and certificate hashes bind the stored payload
to the qualified reference.

## Online consolidation

Load a scorer state dictionary supplied by the application:

```python
import torch
from humanoid_ttt import ConsolidationPolicy

weights = torch.load("consolidation_policy.pt", map_location="cpu", weights_only=True)
policy = ConsolidationPolicy(weights)
```

For each request, call `observe_request(capability_id, reused=...)`. After a
qualified miss, call `select(capability_id, entries, candidate)`, apply the
returned `INSERT`, `DROP`, or `REPLACE` decision, and call `update()` once.

The policy keeps a pending transition until the next qualified miss. Its reward
is the realized reuse-hit rate between those misses. The scorer uses ten memory
slots, sixteen recent requests, and eleven admission actions. `reset()` restores
the supplied weights and clears online state; `state_dict()` exports current
scorer weights.

`memory.py` also exposes the local-probe utility calculation used for immediate
writeback feedback. This utility and the delayed reuse reward are separate APIs.

## Latent adaptation

`LatentBasis` constructs an RMS-normalized low-rank noise basis. `LatentSteering`
applies a bounded latent action to the initial diffusion noise. `CompactState`
encodes adaptation feedback as 24 features for an eight-dimensional action.

`SACAgent` provides action sampling, twin-critic updates, entropy tuning, and
parameter snapshots. `ConservativeGate` compares proposed actions with zero
steering using critic disagreement and value tolerance. `PerformanceBand`
compares candidate and reference performance summaries. The caller applies its
guard schedule and restores parameter snapshots when an update is rejected.

SAC snapshots contain network parameters and the entropy coefficient; optimizer
and replay state remain with the caller. Deserialize checkpoints from a trusted
source.
