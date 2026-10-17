"""Independent, display-only rendering for ghost-car time trials.

The caller owns simulation and pacing. No environment renderer or observation
callbacks are used; snapshots are the only source of car and scoreboard state.
"""

import numpy as np
import pyglet
from pyglet import gl

from f110_gym.envs.collision_models import get_vertices
from f110_gym.envs.rendering import CAR_LENGTH, CAR_WIDTH, EnvRenderer


_COLORS = ((61, 219, 255), (255, 167, 66))
_SCALE = 50.0  # EnvRenderer's map coordinates are 50 times world metres.
_HUD_HEIGHT = 116


class _GhostWindow(EnvRenderer):
    """Reuse map loading and mouse navigation, not upstream race rendering."""

    def __init__(self):
        self.closed = False
        self.viewer = None
        super().__init__(1024, 768, caption="F1TENTH Ghost Time Trial")
        # The upstream ego scoreboard belongs to its world-space batch.
        self.score_label.delete()

    def on_draw(self):
        if self.viewer is not None and not self.closed:
            self.viewer._render()

    def on_resize(self, width, height):
        result = super().on_resize(width, height)
        if self.viewer is not None:
            self.viewer._fit_camera(self.viewer._bounds)
        return result

    def on_close(self):
        # EnvRenderer.on_close raises an exception; deliberately bypass it.
        self.close()
        return pyglet.event.EVENT_HANDLED

    def close(self):
        if not self.closed:
            self.closed = True
            super().close()


class GhostViewer:
    """Draw two independent ghosts and the caller's lap/timing snapshots.

    ``map_path`` is the map stem (Path or str), ``map_ext`` its image suffix,
    ``start_pose`` is [x, y, heading], and ``gates`` has shape (N, 2, 2).
    Gate zero is the finish gate; remaining gates are ordered checkpoints.
    ``window`` exposes the independent pyglet window for buffer capture.
    """

    def __init__(self, map_path, map_ext, start_pose, gates):
        self.window = _GhostWindow()
        self.window.switch_to()
        self.window.update_map(str(map_path), map_ext)
        self._gates = np.asarray(gates, dtype=np.float64).reshape((-1, 2, 2)) * _SCALE
        self._start_pose = np.asarray(start_pose, dtype=np.float64)
        self._snapshots = []
        self._car_batch = pyglet.graphics.Batch()
        self._cars = []
        self._outlines = []
        for color in _COLORS:
            self._cars.append(self._car_batch.add(
                4, gl.GL_QUADS, None, ('v2f/stream', [0.0] * 8),
                ('c4B', (*color, 160) * 4)))
            self._outlines.append(pyglet.graphics.vertex_list(
                4, ('v2f/stream', [0.0] * 8), ('c4B', (*color, 255) * 4)))
        self._title = pyglet.text.Label(
            'GHOST TIME TRIAL  |  Ghosts ignore each other: no collisions or sensor visibility',
            font_size=12, color=(255, 255, 255, 255), x=16, anchor_y='top')
        self._rows = [pyglet.text.Label(
            '', font_size=12, color=(*color, 255), x=16, anchor_y='top')
            for color in _COLORS]
        self._help = pyglet.text.Label(
            'Drag to pan  |  Scroll to zoom  |  Numbered markers identify overlapping ghosts',
            font_size=10, color=(190, 202, 220, 255), x=16, anchor_y='top')
        self._markers = [pyglet.text.Label(
            str(i + 1), font_size=12, bold=True, color=(*color, 255),
            anchor_x='center', anchor_y='center') for i, color in enumerate(_COLORS)]
        self._gate_labels = [pyglet.text.Label(
            'FINISH' if i == 0 else str(i), font_size=9,
            color=(110, 255, 170, 255) if i == 0 else (210, 215, 235, 255),
            anchor_x='center', anchor_y='center') for i in range(len(self._gates))]
        points = [self._start_pose[:2][None, :] * _SCALE]
        if self.window.map_points.size:
            points.append(self.window.map_points[:, :2])
        if self._gates.size:
            points.append(self._gates.reshape((-1, 2)))
        points = np.concatenate(points)
        self._bounds = np.array([points.min(axis=0), points.max(axis=0)])
        self.window.viewer = self
        self._fit_camera(self._bounds)

    def _fit_camera(self, bounds):
        width, height = self.window.get_size()
        available_height = max(height - _HUD_HEIGHT, 1)
        span = np.maximum(bounds[1] - bounds[0], 100.0)
        scale = max(span[0] / max(width - 60, 1),
                    span[1] / max(available_height - 60, 1))
        center = (bounds[0] + bounds[1]) / 2
        self.window.left = center[0] - scale * width / 2
        self.window.right = self.window.left + scale * width
        self.window.bottom = center[1] - scale * available_height / 2
        self.window.top = self.window.bottom + scale * height
        self.window.zoom_level = scale
        self.window.zoomed_width = scale * width
        self.window.zoomed_height = scale * height

    def draw(self, snapshots):
        """Draw one frame without sleeping; return False after window close."""
        if self.window.closed:
            return False
        self.window.switch_to()
        self.window.dispatch_events()
        if self.window.closed:
            return False
        self._snapshots = list(snapshots)
        poses = [np.asarray(snapshot['pose'], dtype=np.float64)
                 for snapshot in self._snapshots[:2]]
        # Keep both ghosts visible, including when a manual pan/zoom loses one.
        if poses:
            positions = np.array([pose[:2] for pose in poses]) * _SCALE
            margin = 25 * self.window.zoom_level
            visible_top = self.window.top - _HUD_HEIGHT * self.window.zoom_level
            if (np.any(positions[:, 0] < self.window.left + margin)
                    or np.any(positions[:, 0] > self.window.right - margin)
                    or np.any(positions[:, 1] < self.window.bottom + margin)
                    or np.any(positions[:, 1] > visible_top - margin)):
                bounds = np.array([
                    np.minimum(self._bounds[0], positions.min(axis=0)),
                    np.maximum(self._bounds[1], positions.max(axis=0))])
                self._fit_camera(bounds)
        for i in range(2):
            pose = poses[i] if i < len(poses) else self._start_pose
            vertices = _SCALE * get_vertices(pose, CAR_LENGTH, CAR_WIDTH)
            self._cars[i].vertices = vertices.ravel()
            # The second outline is slightly wider, so identical poses still
            # show both colors without moving either ghost's actual rectangle.
            center = pose[:2] * _SCALE
            outline = center + (vertices - center) * (1.0 if i == 0 else 1.22)
            self._outlines[i].vertices = outline.ravel()
        self._render()
        self.window.flip()
        return True

    def _screen_point(self, point):
        width, height = self.window.get_size()
        return ((point[0] - self.window.left) * width / self.window.zoomed_width,
                (point[1] - self.window.bottom) * height / self.window.zoomed_height)

    def _render(self):
        window = self.window
        width, height = window.get_size()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(window.left, window.right, window.bottom, window.top, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        window.batch.draw()
        gl.glLineWidth(2)
        for i, gate in enumerate(self._gates):
            color = (110, 255, 170, 220) if i == 0 else (180, 190, 220, 120)
            pyglet.graphics.draw(2, gl.GL_LINES,
                                 ('v2f', gate.ravel()), ('c4B', color * 2))
        self._car_batch.draw()
        for outline in self._outlines:
            outline.draw(gl.GL_LINE_LOOP)
        gl.glLineWidth(1)

        # HUD and identity markers use pixel coordinates, independent of zoom.
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width, 0, height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        for gate, label in zip(self._gates, self._gate_labels):
            label.x, label.y = self._screen_point(gate.mean(axis=0))
            label.y += 10
            label.draw()
        for i, marker in enumerate(self._markers):
            pose = (self._snapshots[i]['pose'] if i < len(self._snapshots)
                    else self._start_pose)
            x, y = self._screen_point(np.asarray(pose)[:2] * _SCALE)
            marker.x, marker.y = x + (-24 if i == 0 else 24), y + 22
            pyglet.graphics.draw(2, gl.GL_LINES,
                                 ('v2f', (x, y, marker.x, marker.y - 7)),
                                 ('c4B', (*_COLORS[i], 220) * 2))
            marker.draw()
        pyglet.graphics.draw(4, gl.GL_QUADS,
                             ('v2f', (0, height - _HUD_HEIGHT, width,
                                      height - _HUD_HEIGHT, width, height, 0, height)),
                             ('c4B', (8, 18, 37, 242) * 4))
        self._title.y = height - 12
        self._title.draw()
        for i, row in enumerate(self._rows):
            if i < len(self._snapshots):
                snapshot = self._snapshots[i]
                finish = snapshot['finish_time']
                timing = 'elapsed {:.2f}s'.format(snapshot['elapsed'])
                if finish is not None:
                    timing += ' / finish {:.2f}s'.format(finish)
                row.text = '{}  {}  |  laps {}  |  {}  |  {}'.format(
                    i + 1, snapshot['name'], snapshot['laps'], timing, snapshot['status'])
            else:
                row.text = '{}  Waiting for ghost snapshot'.format(i + 1)
            row.y = height - 40 - i * 25
            row.draw()
        self._help.y = height - 92
        self._help.draw()

    def close(self):
        """Close this viewer only; safe to call more than once."""
        self.window.close()
