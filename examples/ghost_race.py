"""Independent, checkpoint-validated ghost time trials on a shared F1TENTH map."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PHYSICS_DT = 0.01
CONTROL_STEPS = 4  # 25 Hz policy, 100 Hz physics
MAX_SPEED = 8.0
MAX_STEER = 0.4189
RAY_INDICES = np.rint(np.linspace(0, 1079, 36)).astype(int)
DEFAULT_CONFIG = Path(__file__).with_name('config_example_map.yaml')


class Track:
    """One fixed map and ordered gates derived from its supplied racing line."""

    def __init__(self, config_path=DEFAULT_CONFIG):
        config_path = Path(config_path).resolve()
        with config_path.open() as stream:
            self.config = yaml.safe_load(stream)
        self.map_path = (config_path.parent / self.config['map_path']).resolve()
        self.map_ext = self.config['map_ext']
        self.config['wpt_path'] = str((config_path.parent / self.config['wpt_path']).resolve())
        waypoints = np.loadtxt(self.config['wpt_path'], delimiter=self.config['wpt_delim'],
                               skiprows=self.config['wpt_rowskip'])
        xy = waypoints[:, [self.config['wpt_xind'], self.config['wpt_yind']]]
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        segments = np.roll(xy, -1, axis=0) - xy
        lengths = np.linalg.norm(segments, axis=1)
        if len(xy) < 4 or np.any(lengths <= 0):
            raise ValueError('Track needs a closed, nondegenerate racing line.')
        distance = np.r_[0.0, np.cumsum(lengths)]
        gate_distance = np.linspace(0, distance[-1], 24, endpoint=False)
        centers = np.column_stack([np.interp(gate_distance, distance, np.r_[xy[:, axis], xy[0, axis]])
                                   for axis in range(2)])
        indices = np.searchsorted(distance, gate_distance, side='right') - 1
        tangents = segments[indices] / lengths[indices, None]
        self.start_pose = np.r_[centers[0], np.arctan2(tangents[0, 1], tangents[0, 0])]
        with self.map_path.with_suffix('.yaml').open() as stream:
            metadata = yaml.safe_load(stream)
        self.resolution = float(metadata['resolution'])
        self.origin = np.asarray(metadata['origin'], dtype=float)
        angle = self.origin[2]
        self.rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                  [np.sin(angle), np.cos(angle)]])
        with Image.open(str(self.map_path) + self.map_ext) as image:
            self.road = np.flipud(np.asarray(image.convert('L')) > 128).copy()
        self.gates = []
        for center, tangent in zip(centers, tangents):
            normal = np.array([-tangent[1], tangent[0]])
            limits = []
            for sign in (-1, 1):
                distances = np.arange(0, 20, self.resolution / 2)
                free = self.on_road(center + sign * distances[:, None] * normal)
                blocked = np.flatnonzero(~free)
                if not len(blocked) or blocked[0] == 0:
                    raise ValueError('Cannot find road boundaries for a checkpoint.')
                limits.append(distances[blocked[0] - 1])
            self.gates.append([center - limits[0] * normal, center + limits[1] * normal])
        self.gates = np.asarray(self.gates)

    def on_road(self, points):
        local = (np.asarray(points) - self.origin[:2]) @ self.rotation
        pixels = np.floor(local / self.resolution).astype(int)
        inside = ((pixels[..., 0] >= 0) & (pixels[..., 0] < self.road.shape[1]) &
                  (pixels[..., 1] >= 0) & (pixels[..., 1] < self.road.shape[0]))
        result = np.zeros(inside.shape, dtype=bool)
        result[inside] = self.road[pixels[..., 1][inside], pixels[..., 0][inside]]
        return result

    def contains_car(self, pose, length=0.58, width=0.31):
        # Sample the complete footprint at <= half-pixel spacing, not just the center.
        x = np.linspace(-length / 2, length / 2, int(np.ceil(2 * length / self.resolution)) + 1)
        y = np.linspace(-width / 2, width / 2, int(np.ceil(2 * width / self.resolution)) + 1)
        xx, yy = np.meshgrid(x, y)
        local = np.column_stack((xx.ravel(), yy.ravel()))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        points = local @ np.array([[c, s], [-s, c]]) + pose[:2]
        return bool(np.all(self.on_road(points)))


class LapCounter:
    """Only forward, in-bounds crossings of every ordered gate complete a lap."""

    def __init__(self, gates):
        self.gates = np.asarray(gates)
        self.next_gate = 1
        self.lap_times = []

    def advance(self, previous, current, start_time, dt):
        last_fraction = -1.0
        for _ in range(len(self.gates)):
            left, right = self.gates[self.next_gate]
            lateral = right - left
            width = np.linalg.norm(lateral)
            lateral = lateral / width
            forward = np.array([lateral[1], -lateral[0]])
            a = np.dot(previous - left, forward)
            b = np.dot(current - left, forward)
            if not (a <= 0 < b):
                break
            fraction = -a / (b - a)
            hit = previous + fraction * (current - previous)
            across = np.dot(hit - left, lateral)
            if not (last_fraction < fraction and 0 <= across <= width):
                break
            last_fraction = fraction
            if self.next_gate == 0:
                self.lap_times.append(start_time + fraction * dt)
            self.next_gate = (self.next_gate + 1) % len(self.gates)


class GhostTrial:
    """A private one-car world. No opponent is ever added to this simulator."""

    def __init__(self, name, track, laps=3, timeout=180.0, seed=12345):
        from f110_gym.envs import F110Env
        if laps < 1 or not np.isfinite(timeout) or timeout < PHYSICS_DT:
            raise ValueError('Laps must be positive and timeout at least one physics tick.')
        self.name, self.track = name, track
        self.target_laps = laps
        self.max_steps = int(np.floor(timeout / PHYSICS_DT + 1e-9))
        self.env = F110Env(map=str(track.map_path), map_ext=track.map_ext,
                           num_agents=1, timestep=PHYSICS_DT, seed=seed)
        # Keep upstream's negative velocity bound: its braking gain divides by it.
        # The public action mapping independently disallows reverse targets.
        self.env.update_params(dict(self.env.params, v_max=MAX_SPEED))
        self.raw_obs, _, _, _ = self.env.reset(track.start_pose[None, :].copy())
        self.counter = LapCounter(track.gates)
        self.steps = 0
        self.status = 'running'
        self.reason = None
        self.finish_time = None
        self.history = np.tile(self._frame(), (4, 1))
        if self.raw_obs['collisions'][0] or not track.contains_car(self.pose):
            raise ValueError('Starting grid overlaps the wall or lies off track.')

    @property
    def pose(self):
        return np.array([self.raw_obs['poses_x'][0], self.raw_obs['poses_y'][0],
                         self.raw_obs['poses_theta'][0]])

    @property
    def observation(self):
        return self.history.reshape(1, 156).copy()

    def _frame(self):
        frame = np.empty(39, dtype=np.float32)
        frame[:36] = np.clip(np.asarray(self.raw_obs['scans'][0])[RAY_INDICES] / 30, 0, 1)
        state = self.env.sim.agents[0].state
        frame[36:] = np.clip([state[3] / MAX_SPEED, state[5] / 10, state[2] / MAX_STEER], -1, 1)
        return frame

    def fail(self, reason):
        self.status, self.reason = 'DNF', reason

    def step(self, action):
        if self.status != 'running':
            return
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all():
            self.fail('invalid action')
            return
        control = np.array([[np.clip(action[0], -1, 1) * MAX_STEER,
                             np.clip(action[1], 0, 1) * MAX_SPEED]])
        for _ in range(CONTROL_STEPS):
            previous = self.pose[:2]
            # Upstream done/lap counts are intentionally not the race judge.
            self.raw_obs, _, _, _ = self.env.step(control)
            self.steps += 1
            if self.raw_obs['collisions'][0] or not self.track.contains_car(self.pose):
                self.fail('wall collision/off track')
                break
            self.counter.advance(previous, self.pose[:2], (self.steps - 1) * PHYSICS_DT, PHYSICS_DT)
            if len(self.counter.lap_times) >= self.target_laps:
                self.finish_time = self.counter.lap_times[self.target_laps - 1]
                self.status = 'finished'
                break
            if self.steps >= self.max_steps:
                self.fail('timeout')
                break
        self.history[:-1] = self.history[1:]
        self.history[-1] = self._frame()

    def snapshot(self):
        return dict(name=self.name, pose=self.pose, laps=len(self.counter.lap_times),
                    elapsed=self.steps * PHYSICS_DT, status=self.status,
                    finish_time=self.finish_time)

    def result(self):
        cumulative = np.asarray([0.0] + self.counter.lap_times)
        return dict(name=self.name, status=self.status, reason=self.reason,
                    laps=len(self.counter.lap_times), finish_time=self.finish_time,
                    elapsed=self.steps * PHYSICS_DT, lap_times=np.diff(cumulative).tolist())


class DemoPolicy:
    """Privileged waypoint follower for demos only, never an eligible submission."""

    def __init__(self, track, trial, gain):
        from argparse import Namespace
        from waypoint_follow import PurePursuitPlanner
        self.planner = PurePursuitPlanner(Namespace(**track.config), 0.3302)
        self.trial, self.gain = trial, gain

    def act(self, observation):
        speed, steer = self.planner.plan(*self.trial.pose, 0.8246, self.gain)
        return np.array([steer / MAX_STEER, min(speed, 3.0) / MAX_SPEED])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*', help='Two self-contained ONNX files; input obs[1,156], output action[1,2].')
    parser.add_argument('--demo', action='store_true', help='Race two hand-coded waypoint followers, not ML models.')
    parser.add_argument('--render', action='store_true', help='Show translucent ghosts at real-time speed.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--laps', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180.0, help='Maximum simulated seconds per entrant.')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()
    if (args.demo and args.models) or (not args.demo and len(args.models) != 2):
        parser.error('Use --demo OR supply exactly two ONNX files.')
    if args.laps < 1 or not np.isfinite(args.timeout) or args.timeout < PHYSICS_DT:
        parser.error('--laps must be positive; --timeout must be finite and >= 0.01.')
    track = Track(args.config)
    names = ['Pure Pursuit A (demo)', 'Pure Pursuit B (demo)'] if args.demo else args.models
    try:
        if not args.demo:
            from ghost_policy import OnnxPolicy
            policies = [OnnxPolicy(path) for path in args.models]
        trials = [GhostTrial(name, track, args.laps, args.timeout, args.seed) for name in names]
        if args.demo:
            policies = [DemoPolicy(track, trial, gain) for trial, gain in zip(trials, [0.5, 0.42])]
    except ValueError as error:
        parser.error(str(error))
    viewer = None
    if args.render:
        from ghost_viewer import GhostViewer
        viewer = GhostViewer(track.map_path, track.map_ext, track.start_pose, track.gates)
    print('DEMO: privileged hand-coded waypoint followers, not trained models.' if args.demo else
          'ONNX time trial: wall-only observations; ghosts cannot interact.', flush=True)
    tick = 0
    started = time.monotonic()
    try:
        while any(trial.status == 'running' for trial in trials):
            if viewer and not viewer.draw([trial.snapshot() for trial in trials]):
                for trial in trials:
                    if trial.status == 'running':
                        trial.fail('viewer closed')
                break
            for trial, policy in zip(trials, policies):
                if trial.status != 'running':
                    continue
                try:
                    trial.step(policy.act(trial.observation))
                except ValueError as error:
                    trial.fail('policy error: ' + str(error))
            tick += 1
            if viewer:
                time.sleep(max(0, started + tick * CONTROL_STEPS * PHYSICS_DT - time.monotonic()))
        if viewer:
            viewer.draw([trial.snapshot() for trial in trials])
    finally:
        if viewer:
            viewer.close()
    finishers = [trial for trial in trials if trial.status == 'finished']
    winner = None
    if len(finishers) == 1:
        winner = finishers[0].name
    elif len(finishers) == 2 and abs(finishers[0].finish_time - finishers[1].finish_time) > 1e-9:
        winner = min(finishers, key=lambda trial: trial.finish_time).name
    print(json.dumps(dict(mode='demo' if args.demo else 'onnx', winner=winner,
                          results=[trial.result() for trial in trials]), indent=2))


if __name__ == '__main__':
    main()
