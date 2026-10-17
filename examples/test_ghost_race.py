import numpy as np
import pytest

from ghost_race import GhostTrial, LapCounter, PHYSICS_DT, Track


def square_gates():
    return np.array([[[1, -.4], [1, .4]],
                     [[2.4, 1], [1.6, 1]],
                     [[1, 2.4], [1, 1.6]],
                     [[-.4, 1], [.4, 1]]])


def test_lap_requires_all_gates_and_interpolates_finish_time():
    counter = LapCounter(square_gates())
    route = np.array([[1, 0], [2, 0], [2, 2], [0, 2], [0, 0], [2, 0]])
    for tick, (previous, current) in enumerate(zip(route, route[1:])):
        counter.advance(previous, current, tick, 1.0)
        if tick < 4:
            assert counter.lap_times == []
    assert counter.lap_times == [4.5]
    assert counter.next_gate == 1


def test_backwards_outside_or_skipped_gates_do_not_advance():
    counter = LapCounter(square_gates())
    counter.advance(np.array([2, 2]), np.array([2, 0]), 0, 1)
    counter.advance(np.array([3, 0]), np.array([3, 2]), 1, 1)
    for _ in range(4):
        counter.advance(np.array([0, 0]), np.array([2, 0]), 2, 1)
        counter.advance(np.array([2, 0]), np.array([0, 0]), 3, 1)
    assert counter.next_gate == 1
    assert counter.lap_times == []


def test_ghost_trajectory_and_sensors_do_not_depend_on_other_entrant():
    track = Track()
    solo = GhostTrial('solo', track)
    action = np.array([0., .15])
    expected = []
    for _ in range(12):
        solo.step(action)
        expected.append((solo.pose, solo.observation))
    ghost = GhostTrial('ghost', track)
    other = GhostTrial('other', track)
    # Identical overlapping starts are legal: neither simulator contains an opponent.
    np.testing.assert_array_equal(ghost.pose, other.pose)
    for position, observation in expected:
        other.step(np.array([.5, .05]))
        ghost.step(action)
        assert ghost.status == 'running'
        np.testing.assert_array_equal(ghost.pose, position)
        np.testing.assert_array_equal(ghost.observation, observation)


def test_timeout_uses_simulation_time_and_terminal_trial_cannot_move():
    trial = GhostTrial('parked', Track(), timeout=.05)
    trial.step(np.zeros(2))
    assert trial.status == 'running'
    trial.step(np.zeros(2))
    assert trial.status == 'DNF'
    assert trial.reason == 'timeout'
    assert trial.steps * PHYSICS_DT == pytest.approx(.05)
    pose = trial.pose
    trial.step(np.ones(2))
    np.testing.assert_array_equal(trial.pose, pose)


def test_wall_collision_ends_only_that_ghost():
    track = Track()
    crash = GhostTrial('crash', track)
    parked = GhostTrial('parked', track)
    for _ in range(500):
        crash.step(np.array([0., .5]))
        if crash.status != 'running':
            break
    assert crash.status == 'DNF'
    assert crash.reason == 'wall collision/off track'
    assert parked.status == 'running'
    assert parked.steps == 0
    np.testing.assert_array_equal(parked.pose[:2], track.start_pose[:2])
