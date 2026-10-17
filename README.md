![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# ML Racing: Ghost Time Trials

A shared ONNX racing competition built on
[F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym). Train independently,
exchange models, and compare ghost cars by elapsed simulation time.

**Read [agents.md](agents.md) before training, submitting a model, or changing
the evaluator.** It defines the strict competition rules and protected files.
The upstream MIT license and attribution are preserved. Upstream simulator
documentation is available [here](https://f1tenth-gym.readthedocs.io/en/latest/).

## Quickstart
Clone this competition repository, not the unmodified upstream simulator:

```bash
git clone https://github.com/B3rK-3/ml-racing.git
cd ml-racing
```

Follow the Python 3.9 installation steps under **Ghost time trials** below,
then run the demo or supply your two ONNX files. The original waypoint
demonstration remains available as `examples/waypoint_follow.py`.

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
````
Then the same example can be ran.

## Ghost time trials

`examples/ghost_race.py` runs two **independent single-car simulations** on
the same track, from the same standing-start pose and with the same sensor
seed. Ghosts neither collide with nor appear in each other's sensors. The
viewer combines their positions only; it never feeds observations back into
either simulation. Upstream multi-agent behavior is unchanged.

### Run

From this repository's root, using the environment installed alongside it:

```bash
source ../.venv-f1tenth/bin/activate
python examples/ghost_race.py --demo --render --laps 1
python examples/ghost_race.py person_a.onnx person_b.onnx --render
```

Omit `--render` for unpaced headless evaluation. Both modes use the same
simulation-time scoring, independent of rendering or inference wall time.
The defaults are three laps and a 180-second simulated timeout per entrant.
Use `--laps`, `--timeout`, and `--seed` to set shared race conditions.
Closing the viewer cancels active attempts as DNF and prints their results.

**The demo is not machine learning.** It adapts the repository's hand-coded
Pure Pursuit waypoint follower, with different speed gains for the two cars.
It has privileged pose/racing-line access and is not an eligible submission.
The older `examples/waypoint_follow.py` demonstration uses that same controller.

For a fresh Python 3.9 installation (the old Gym metadata needs legacy pip):

```bash
uv venv --python 3.9 ../.venv-f1tenth
source ../.venv-f1tenth/bin/activate
uv pip install --python ../.venv-f1tenth/bin/python \
  pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
python -m pip install --no-build-isolation -e .
python -m pip install -r examples/requirements-ghost.txt
```

The requirements record the versions exercised on macOS arm64. Pyglet 1.5.27
deliberately overrides upstream's `<1.5` constraint to load macOS OpenGL.
Pillow 9.5 retains the image API used by the upstream renderer.

### Policy contract

One self-contained ONNX file, at most 100 MiB, with no external tensor files:

| Tensor | Name | Type | Shape |
|---|---|---|---|
| Input | `obs` | `float32` | `[1, 156]` |
| Output | `action` | `float32` | `[1, 2]` |

The input is four consecutive 39-value frames, **oldest first**, flattened.
All four frames initially contain the reset observation. Each subsequent
frame is sampled after one 40 ms control interval:

| Frame indices | Feature | Normalization |
|---|---|---|
| 0–35 | Wall-only range samples | `clip(distance / 30 metres, 0, 1)` |
| 36 | Signed forward speed | `clip(speed / 8 m/s, -1, 1)` |
| 37 | Signed yaw rate | `clip(yaw_rate / 10 rad/s, -1, 1)` |
| 38 | Actual steering angle | `clip(steering / 0.4189 rad, -1, 1)` |

The range samples use indices `round(linspace(0, 1079, 36))` from the native
1,080-beam, 4.7-radian scan. Beam angle is
`-2.35 + index * 4.7 / 1079` radians relative to heading. Positive angle,
yaw rate, and steering turn counterclockwise/left. The simulator's seeded
range noise remains enabled; normalized no-hit/maximum range is 1.
There are no coordinates, checkpoint directions, lap counters, track IDs,
or opponent features in the policy input. State must fit in this fixed
history window; there are no persistent hidden-state tensors.

`action[0,0]` is clipped to `[-1,1]` and multiplied by 0.4189 radians.
`action[0,1]` is clipped to `[0,1]` and multiplied by 8 m/s: this is a
**target speed**, not throttle. Zero requests stopping; reverse targets
are unavailable. Actions are held for four 10 ms physics steps (25 Hz
control, 100 Hz physics). Vehicle dynamics otherwise use upstream defaults,
with maximum speed set to 8 m/s.

ONNX Runtime uses CPU execution with one intra-op and one inter-op thread.
Wrong tensor signatures are rejected before racing; inference failures or
nonfinite outputs make the affected entrant DNF. Local submissions are
assumed trusted: this runner is not an OS sandbox and does not enforce a
hard per-inference deadline.

### Timing and track rules

Gate 0 is the start/finish. The runner generates 23 additional gates at
equal distances around the supplied racing line, bounded by the adjacent
road edges. Every gate must be crossed in order, in the forward direction,
within its endpoints, before a lap counts. Crossing only the start line
repeatedly cannot score laps. Finish times are interpolated within the
10 ms physics tick; upstream's start-zone lap count and ego termination
are not used for judging.

Wall collision, an off-track vehicle footprint, invalid inference, or timeout
ends only that entrant's attempt. The other continues. A finisher beats a
DNF; two finishers are ranked by total elapsed simulation time. Identical
finish times tie; two DNFs have no winner. JSON results include individual
lap times and failure reasons.

`--config` accepts the existing example configuration schema; map and
waypoint paths resolve relative to that YAML file. Start pose comes from
the first racing-line point and its forward tangent, not `sx/sy/stheta`.
The line must be a closed loop ordered in the race direction and stay on
the road. Both entrants always use that same map: upstream caches its
scan geometry globally, so do not interleave trials on different maps
in one process. Course gates and the supplied default track should be
reviewed before using a new custom map for official results.

Regression checks:

```bash
python -m pip install pytest==8.4.2
python -m pytest examples/test_ghost_race.py examples/test_ghost_policy.py -q
```

## Known issues
- Library support issues on Windows. You must use Python 3.8 as of 10-2021
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.20
```
And you might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
