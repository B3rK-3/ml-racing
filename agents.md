# Ghost Racing: Competition and Agent Rules

This file governs participants and coding assistants working on this project.
MUST and MUST NOT are mandatory. An official race uses a jointly agreed repository
commit, track configuration, environment, and model artifacts. No participant may
change the rules or official environment unilaterally. Disputes or discovered
exploits MUST be resolved before accepting a result, not exploited silently.

## 1. Competition objective

Two independently trained policies drive identical cars on the same course.
They are **ghost cars**: each runs in a separate single-car simulation. Their
positions are combined only for display. They MUST NOT collide with, block,
draft behind, or sense one another. This is a driving time trial, not a
head-to-head interaction or combat competition.

A valid finisher MUST complete every required lap. Lowest total elapsed
**simulation time**, including the standing-start first lap, wins. The fastest
individual lap and the training reward do not determine the winner.

## 2. What participants MAY do

- Use any training algorithm, framework, architecture, optimizer, reward shaping,
  imitation learning, reinforcement learning, evolution, or combination thereof.
- Use any training compute budget; there is no training hardware or time cap.
- Train on the shared course, other courses, or generated courses. Known-track
  specialization is allowed; this initial competition is not a hidden-track test.
- Use simulator ground truth, coordinates, racing lines, teachers, and privileged
  critics **during training only**. The final inference graph MUST consume only
  the official observation tensor.
- Choose different training rewards and curricula. Local training wrappers MAY
  change rewards, logging, resets, and episode sampling without changing the
  official vehicle dynamics, sensing, or action meaning.
- Keep training code private and share only the final self-contained ONNX file.
- Learn internal representations of the known track through model weights.
  Removing coordinates from observations does not guarantee generalization.
- Use temporal processing within the provided four-frame input window.

## 3. What participants MUST NOT do

- Change the official physics, car dimensions, friction, speed limits, steering,
  timestep, sensors, noise, normalization, input history, start pose, track image,
  checkpoints, timing, collision/off-track checks, timeout, or winner selection.
- Feed a competing model coordinates, heading, track ID, racing-line or checkpoint
  coordinates, centerline offset, lap progress, race tick, or opponent information
  through additional inputs, side channels, or a modified observation wrapper.
- Use participant-supplied Python, callbacks, native libraries, custom ONNX
  operators, plugins, file reads, network access, processes, or external services
  to decide race actions. Only the submitted ONNX graph may control the car.
- Read another entrant's observations, actions, weights, timing, or private
  training data without permission. Publicly shared training data is allowed.
- Use persistent external memory, hidden-state tensors, mutable model state,
  randomness during inference, or wall-clock time to determine actions. Export
  deterministic evaluation behavior; stochastic training is allowed.
- Replace a failed inference with another controller or use the waypoint demo
  as a fallback. A failed policy attempt is DNF.
- Submit or change a model after inspecting official results to retroactively
  improve the same match. Both artifacts MUST be frozen before either is scored.
- Exploit checkpoint skipping, teleporting, resets, track cuts, timing bugs,
  nonfinite values, or modifications to the official runner. Report bugs instead.
- Represent a hand-coded privileged waypoint follower as a trained submission.
  The provided Pure Pursuit controllers are demonstrations only.

## 4. Exact submission and inference contract

Each entrant MUST provide exactly one self-contained `.onnx` file of at most
**104,857,600 bytes (100 MiB)**. Weights and preprocessing MUST be embedded in the
graph or supplied by the unchanged common observation wrapper. External tensor
files and extra graph inputs/outputs are forbidden.

| Direction | Exact tensor name | Type | Exact shape |
|---|---|---|---|
| Input | `obs` | `float32` | `[1, 156]` |
| Output | `action` | `float32` | `[1, 2]` |

The model MUST run with the pinned ONNX Runtime CPU provider. There is one
intra-op thread and one inter-op thread. Training framework dependencies are
not loaded by the race runner. Export graphs compatible with the pinned runtime;
an export that a training framework accepts but this runtime rejects is not an
eligible submission.

### Observations

`obs` contains four contiguous 39-value frames, oldest first, flattened in row
order. At reset, the first observation is repeated into all four history slots.
A new frame is appended every four physics steps (40 ms).

| Index within each frame | Feature | Exact normalization |
|---|---|---|
| 0–35 | Wall-only ray distances | `clip(distance_metres / 30, 0, 1)` |
| 36 | Signed forward speed | `clip(speed_metres_per_second / 8, -1, 1)` |
| 37 | Signed yaw rate | `clip(yaw_rate_radians_per_second / 10, -1, 1)` |
| 38 | Actual steering angle | `clip(steering_radians / 0.4189, -1, 1)` |

Ray indices are `numpy.rint(numpy.linspace(0, 1079, 36)).astype(int)` from the
native 1,080-beam scan. A sampled beam's angle relative to the car is
`-2.35 + index * 4.7 / 1079` radians. Positive angle, yaw rate, and steering mean
counterclockwise/left. Range noise remains enabled with the same seed for both
entrants. Maximum/no-hit normalized distance is 1. Ghost positions NEVER enter
these rays.

### Actions

| Output index | Interpretation | Shared conversion |
|---|---|---|
| `action[0,0]` | Steering request | `clip(value, -1, 1) * 0.4189` radians |
| `action[0,1]` | Target speed, NOT throttle | `clip(value, 0, 1) * 8` metres/second |

All output values MUST be finite. Finite out-of-range values are clipped by the
common wrapper, not treated as cheating or DNF. Zero target speed requests
stopping; negative values cannot request reverse. The underlying speed controller
and braking behavior remain the shared simulator implementation.

The policy runs at **25 Hz**; the action is held over four **100 Hz** physics
steps. Policies MUST NOT change these rates. There are no recurrent-state inputs
or outputs; all inference-time temporal context comes from the fixed window.

## 5. Official race procedure

1. Both participants MUST approve and record the exact official Git commit,
   dependency environment, course/configuration, lap count, timeout, seed, and
   evaluator machine before submissions are frozen. Changing any of them requires
   agreement and a new match for both participants.
2. Each participant supplies one frozen ONNX artifact. Record each file's SHA-256
   checksum. Filenames alone are not artifact identities. Do not commit large
   weights into Git; exchange files or use GitHub release assets.
3. Validate both submissions before the official attempt. An invalid size,
   external-data reference, tensor signature, or unloadable graph is an ineligible
   entry; the CLI aborts admission rather than producing a race winner.
4. Evaluate both files in the unchanged official runner on the same machine.
   Each gets its own single-car world, identical vehicle parameters, the same
   sensor seed, and the exact same standing start. There is no grid advantage.
5. Use the defaults below unless both participants froze another configuration
   in step 1. Demo races NEVER count as official results.
6. Preserve the command, commit, model checksums, runtime versions, and JSON
   results together. A rendered video is optional evidence, not the timekeeper.

| Default match setting | Value |
|---|---|
| Course | `examples/config_example_map.yaml` and referenced supplied files |
| Start | First racing-line point; heading along its forward tangent |
| Race length | 3 complete laps |
| Timeout | 180 simulated seconds per entrant |
| Seed | 12345 |
| Vehicle parameters | Upstream defaults, maximum speed set to 8 m/s |
| Physics / control | 100 Hz / 25 Hz |
| Starting velocity / steering | Zero / zero |

The configuration's legacy `sx/sy/stheta` do not set this runner's start pose.
No respawns, mid-race resets, flying starts, or selective reruns are allowed.
Rendering may be enabled or disabled; both use the same simulation-time scoring.
A timeout is resolved on physics ticks (a non-tick-aligned value is rounded down).

### Laps and timing

The runner derives 24 gates around the supplied closed racing line. Gate 0 is
start/finish; gates 1–23 must be crossed in order, followed by gate 0. Each gate
requires a forward crossing within its road-bounded endpoints. A completed
sequence counts one lap. Repeated start-line crossings alone do not count.

The timer starts from the stationary reset state. Finish time is interpolated
inside the finishing 10 ms physics step. Upstream ego-only termination and
start-zone lap counters are not used as the judge. The runner samples the vehicle
footprint against the road bitmap in addition to upstream wall detection.
Custom courses MUST be reviewed for correctly placed gates before official use.

### Failure and ranking

- Wall collision or an off-track footprint: that entrant is DNF.
- Inference error, malformed action, or NaN/infinite action: that entrant is DNF.
- Simulated timeout before finishing: that entrant is DNF.
- A DNF entrant stops advancing; the other continues unaffected.
- One finisher beats one DNF, regardless of the DNF's partial progress.
- Two finishers: lower total finish time wins. Differences of at most `1e-9`
  seconds are a tie, matching the runner's comparison tolerance.
- Two DNFs: no winner. No fastest-lap bonus or partial-progress points are awarded.
- Closing the viewer cancels any active attempt as DNF. An operator mistake or
  infrastructure failure MUST NOT be used to award a win: agree to void the
  incomplete match and rerun BOTH unchanged artifacts under the same conditions.

## 6. Enforcement boundaries

The loader automatically enforces file size, self-contained tensors, exact graph
signatures, CPU provider/thread configuration, and finite normalized actions.
The runner enforces independent worlds, checkpoint order, simulation timing,
timeout, and collision/off-track failure. Graph determinism, prohibited operator
behavior, and tampering require participant review; they are not fully audited
by the current loader.

There is **no hard per-inference wall-clock deadline, process-memory quota, or OS
sandbox**. The simulation timeout does not stop a hung inference call. Do not
claim a 10 ms inference cap or malware isolation. Both participants MUST trust
and review submissions; run unfamiliar artifacts in an appropriately isolated
machine. No unilateral time penalty or DNF may be invented after seeing a slow
model. Any additional resource limits require agreement and implementation
before a new match.

## 7. Rules for coding assistants and repository changes

- MUST read this file and the README before changing participant training or
  competition code. The strict rules here govern official matches; if docs and
  implementation disagree, stop and report the discrepancy before scoring.
- MUST keep participant training/experiments separate, for example under
  `training/<participant>/`. These are suggested locations, not existing APIs.
- MUST NOT modify the protected evaluator to improve one entrant's result.
  Protected files include `examples/ghost_race.py`, `examples/ghost_policy.py`,
  `examples/ghost_viewer.py`, `examples/requirements-ghost.txt`, the shared track
  YAML/PNG/waypoint files, `gym/f110_gym/`, and the evaluation regression tests.
- Shared bug fixes or rule changes MAY touch protected files only with both
  participants' agreement. Update this rulebook and README as needed, verify the
  affected behavior, then freeze a new official commit. Reevaluate both entrants.
- MUST NOT suppress wall failures, weaken checkpoints/tests, inject coordinates
  into policy inputs, special-case a model name/hash, or silently change defaults.
- MUST keep demos clearly labeled. MAY use Pure Pursuit for debugging and
  training demonstrations, but MUST NOT grant that access to submitted models.
- MUST verify behavioral changes with the focused regression suite and an actual
  race smoke run. UI changes additionally require checking the visible viewer.
- MUST NOT commit virtual environments, credentials, caches, private datasets,
  or large training artifacts. Preserve the upstream license and attribution.

## 8. Commands

From the repository root after the README's installation steps:

```bash
source ../.venv-f1tenth/bin/activate

# Demonstration only: two privileged hand-coded waypoint followers.
python examples/ghost_race.py --demo --render --laps 1

# Official default match. Replace paths with the frozen submitted files.
python examples/ghost_race.py person_a.onnx person_b.onnx \
  --config examples/config_example_map.yaml --laps 3 --timeout 180 --seed 12345

# The same match with display-only ghosts.
python examples/ghost_race.py person_a.onnx person_b.onnx \
  --config examples/config_example_map.yaml --laps 3 --timeout 180 --seed 12345 --render

# Verify the evaluator before freezing a match.
python -m pytest examples/test_ghost_race.py examples/test_ghost_policy.py -q
```

Never substitute `--demo` or the upstream `waypoint_follow.py` for an official
ONNX match. The README describes installation and implementation details;
this file defines the competition agreement and modification boundaries.
