# HumanoidTTT: Test-Time Capability Reuse for Efficient Humanoid Control

[Model weights](https://huggingface.co/AIGeeksGroup/HumanoidTTT) · [Files](https://huggingface.co/AIGeeksGroup/HumanoidTTT/tree/main)

HumanoidTTT reuses qualified full-motion capabilities when the current robot state
lies inside their applicability regions. On a miss, a frozen motion generator
produces a new candidate for qualification. A Double-DQN consolidation policy
learns which capabilities to retain in a finite-capacity store from subsequent
successful reuse.

## Method

- **Entry applicability (A2):** a 45-dimensional entry-state descriptor, local-cell certificates, and physical and signature checks determine whether a stored motion can be reused.
- **Consolidation:** a shared `71 → 128 → 64 → 1` network scores `SKIP` and `REPLACE(j)` actions when the ten-slot store is full.
- **Online feedback:** the reward is the fraction of subsequent requests with successful reuse between consecutive full-store qualified-miss decisions. The policy updates with Double-DQN.
- **Generation and tracking:** OMG and HoloMotion remain frozen.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/AIGeeksGroup/HumaniodTTT.git
cd HumaniodTTT
pip install -e .
pip install huggingface_hub
```

## Model weights

The initial consolidation checkpoint is hosted on
[Hugging Face](https://huggingface.co/AIGeeksGroup/HumanoidTTT).

| Checkpoint | Architecture | Parameters | Initialization seed |
| --- | --- | ---: | ---: |
| [consolidation_policy.pt](https://huggingface.co/AIGeeksGroup/HumanoidTTT/blob/main/consolidation_policy.pt) | 71 → 128 → 64 → 1 | 17,537 | 83001 |

The release also includes `config.json` and `SHA256SUMS`. The checkpoint contains
the initial FP32 scorer parameters before online adaptation. Both the online and
target networks are initialized from these weights.

Download and load the policy:

```python
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from humanoid_ttt import ConsolidationPolicy

repo_id = "AIGeeksGroup/HumanoidTTT"
config_path = hf_hub_download(repo_id, "config.json")
config = json.loads(Path(config_path).read_text())
weights_path = hf_hub_download(repo_id, config["checkpoint"])
assert hashlib.sha256(Path(weights_path).read_bytes()).hexdigest() == config["checkpoint_sha256"]
weights = torch.load(weights_path, map_location="cpu", weights_only=True)
policy = ConsolidationPolicy(
    weights,
    capacity=config["capacity"],
    online_learning_rate=config["online_learning_rate"],
    gamma=config["gamma"],
)
```

For reproducible runs, pin the Hugging Face commit with `revision=` in both
download calls and record the code commit and configuration used.

Download the frozen generator and tracker from the official
[OMG](https://github.com/Tsinghua-MARS-Lab/OMG) and
[HoloMotion](https://github.com/HorizonRobotics/HoloMotion) projects. Use the
checkpoint versions required by your execution configuration; the consolidation
checkpoint does not replace these upstream models.

## Using the components

This repository provides the method components. An application supplies the
generator, HoloMotion runtime, motion archives, qualification observations, and
the execution loop.

### Entry applicability

Motion references use `float32[60, 36]` at 30 Hz: root position, root quaternion
in `wxyz` order, and 29 G1 joint positions. Stored motion archives expose a
`qpos_36` array. `capture_entry_features()` produces the 45-dimensional descriptor;
`FEATURE_NAMES` specifies its order. History checks require 32 tracker steps.

`build_certificate(package, reference, motion_build, geometry)` constructs local
cells from successful qualification replay endpoints. The application supplies
the replay evidence and geometry, including feature scales and radius.
`request_from_entry()` builds the current request; `evaluate_membership()` checks
its signature, physical conditions, and certificate membership. Certificates must
refer to the corresponding motion and runtime configuration.

### Online consolidation

For each request, call `observe_request(capability_id, reused=successful_reuse)`
with the observed execution outcome. After a qualified miss, call
`select(capability_id, entries, candidate)`, apply the returned decision, and call
`update()` once.

The API returns `INSERT` while a slot is free, `DROP` for the paper's `SKIP`
action, or `REPLACE` with the selected entry ID. Learned decisions use eleven
action rows of 71 features when the store is full. The policy keeps a pending
transition until the next such decision, then computes the intervening successful
reuse fraction and performs the update.

`reset()` restores the initial online and target weights and clears the policy's
optimizer, replay, and request history. Store initialization is managed by the
application. `state_dict()` exports current online scorer weights; it is not a
complete checkpoint for resuming an interrupted online run.

## Code organization

| Module | Purpose |
| --- | --- |
| `entry_features.py` | 45-dimensional entry-state features. |
| `certificates.py` | Certificate construction and applicability checks. |
| `consolidation.py` | 71-dimensional action scorer and Double-DQN updates. |
| `motion_features.py` | 16-dimensional motion descriptors used in consolidation. |
| `memory.py` | Store actions and local-probe utility helpers. |
| `signatures.py` | Request and runtime compatibility signatures. |
| `types.py` | Motion-reference and store-entry interfaces. |
| `hashing.py` | Deterministic artifact hashes. |

The local-probe utility helpers in `memory.py` are separate from the delayed
successful-reuse reward used by `ConsolidationPolicy`.

## Acknowledgements

The generation and tracking stack builds on [OMG](https://github.com/Tsinghua-MARS-Lab/OMG)
and [HoloMotion](https://github.com/HorizonRobotics/HoloMotion).
