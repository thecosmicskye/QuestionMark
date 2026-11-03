"""RLBot v5 packets -> the float32 tables policy.pt2's manifest names. Only packet fields as reported (plus the
rotator's orientation basis and the controls the bot holds); every derived value is computed inside the archive."""
import numpy as np

CONTROLLER = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
BUTTONS = {'jump', 'boost', 'handbrake', 'use_item'}


def basis(pitch, yaw, roll):
    """Rotator -> forward, right, up (float32 arrays of shape (..., 3))."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    forward = np.stack((cp * cy, cp * sy, sp), -1)
    up = np.stack((-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cp * cr), -1)
    return forward, np.cross(up, forward), up


# Player columns this reader can fill: name -> expression of the PlayerInfo `p` (`ci` is its last_input).
# present/basis/held are filled separately.
PLAYER_FIELDS = {
    'team': 'int(p.team)', 'x': 'ph.location.x', 'y': 'ph.location.y', 'z': 'ph.location.z', 'vx': 'ph.velocity.x',
    'vy': 'ph.velocity.y', 'vz': 'ph.velocity.z', 'wx': 'ph.angular_velocity.x', 'wy': 'ph.angular_velocity.y',
    'wz': 'ph.angular_velocity.z', 'pitch': 'ph.rotation.pitch', 'yaw': 'ph.rotation.yaw', 'roll': 'ph.rotation.roll',
    'boost': 'p.boost', 'demolished_timeout': 'p.demolished_timeout', 'is_supersonic': 'p.is_supersonic',
    'air_state': 'int(p.air_state)', 'has_jumped': 'p.has_jumped', 'has_double_jumped': 'p.has_double_jumped',
    'has_dodged': 'p.has_dodged', 'dodge_elapsed': 'p.dodge_elapsed', 'dodge_dir.x': 'p.dodge_dir.x',
    'dodge_dir.y': 'p.dodge_dir.y', 'dodge_timeout': 'p.dodge_timeout', 'hitbox.length': 'p.hitbox.length',
    'hitbox.width': 'p.hitbox.width', 'hitbox.height': 'p.hitbox.height', 'hitbox_offset.x': 'p.hitbox_offset.x',
    'hitbox_offset.y': 'p.hitbox_offset.y', 'hitbox_offset.z': 'p.hitbox_offset.z', 'is_bot': 'p.is_bot',
    'spawn_id': 'p.spawn_id', **{f'last_input.{n}': f'(float(ci.{n}) if ci is not None else 0.)' for n in CONTROLLER},
}


def row_function(names):
    """One compiled function p -> tuple of the named PlayerInfo fields (as fast as a hand-written tuple)."""
    body = ', '.join(PLAYER_FIELDS[n] for n in names)
    return eval(f'lambda p, ph, ci: ({body},)', {})


BASIS = ('forward.x', 'forward.y', 'forward.z', 'right.x', 'right.y', 'right.z', 'up.x', 'up.y', 'up.z')
BALL_FIELDS = {'x': lambda b: b.location.x, 'y': lambda b: b.location.y, 'z': lambda b: b.location.z,
               'vx': lambda b: b.velocity.x, 'vy': lambda b: b.velocity.y, 'vz': lambda b: b.velocity.z,
               'wx': lambda b: b.angular_velocity.x, 'wy': lambda b: b.angular_velocity.y,
               'wz': lambda b: b.angular_velocity.z, 'pitch': lambda b: b.rotation.pitch,
               'yaw': lambda b: b.rotation.yaw, 'roll': lambda b: b.rotation.roll}
GAME_FIELDS = {'phase': lambda p: int(p.match_info.match_phase), 'seconds_elapsed': lambda p: p.match_info.seconds_elapsed,
               'game_time_remaining': lambda p: p.match_info.game_time_remaining,
               'is_overtime': lambda p: p.match_info.is_overtime, 'is_unlimited_time': lambda p: p.match_info.is_unlimited_time,
               'blue_score': lambda p: p.teams[0].score, 'orange_score': lambda p: p.teams[1].score,
               'frame_num': lambda p: p.match_info.frame_num}


def check(names, known, table):
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f'policy.pt2 asks for {table} columns this bot cannot read: {unknown}')


class PacketReader:
    """Fills one flat float32 buffer: players (max_players rows), ball, pads (the manifest's pad order), game."""

    def __init__(self, manifest, field_info):
        self.players, self.ball, self.pads, self.game = (manifest[k] for k in ('players', 'ball', 'pads', 'game'))
        self.max_players, self.controls = int(manifest['max_players']), list(manifest['controls'])
        check(self.players, {'present', *PLAYER_FIELDS, *BASIS, *(f'held.{n}' for n in self.controls)}, 'players')
        check(self.ball, {'present', *BALL_FIELDS}, 'ball')
        check(self.pads, {'timer', 'is_active', 'is_big', 'x', 'y', 'z'}, 'pads')
        check(self.game, GAME_FIELDS, 'game')
        wanted = manifest['pad_locations']
        locs = np.array([(p.location.x, p.location.y) for p in field_info.boost_pads]).reshape(-1, 2)
        self.pad_map = [int(np.linalg.norm(locs - np.array([x, y]), axis=1).argmin()) if len(locs) else -1 for x, y, _ in wanted]
        if len(locs) != len(wanted) or len(set(self.pad_map)) != len(wanted) or any(
                np.linalg.norm(locs[i] - np.array([x, y])) > 3 for i, (x, y, _) in zip(self.pad_map, wanted)):
            raise ValueError("this field's boost pads differ from the layout policy.pt2 was trained on")
        pads = [field_info.boost_pads[i] for i in self.pad_map]
        self.pad_static = {'is_big': np.array([p.is_full_boost for p in pads], np.float32),
                           'x': np.array([p.location.x for p in pads], np.float32),
                           'y': np.array([p.location.y for p in pads], np.float32),
                           'z': np.array([p.location.z for p in pads], np.float32)}
        self.sizes = [self.max_players * len(self.players), len(self.ball), len(wanted) * len(self.pads), len(self.game)]
        self.held_cols = [(k, self.controls.index(n[5:])) for k, n in enumerate(self.players) if n.startswith('held.')]
        self.basis_cols = [(k, BASIS.index(n)) for k, n in enumerate(self.players) if n in BASIS]
        self.field_cols = [k for k, n in enumerate(self.players) if n in PLAYER_FIELDS]
        self.row = row_function([n for n in self.players if n in PLAYER_FIELDS])
        self.present_col = self.players.index('present') if 'present' in self.players else None

    def read(self, packet, held, out):
        """Write this packet's tables into `out` (flat float32); held: (max_players, controls) the bot is sending."""
        players = packet.players
        if len(players) > self.max_players:
            raise ValueError(f'policy.pt2 reads at most {self.max_players} players')
        a, b, c, _ = np.cumsum(self.sizes)
        table = out[:a].reshape(self.max_players, -1)
        table[:] = 0.
        n = len(players)
        if n:
            table[:n, self.field_cols] = np.array([self.row(p, p.physics, p.last_input) for p in players], np.float32)
            if self.present_col is not None:
                table[:n, self.present_col] = 1.
            if self.basis_cols:
                rot = np.array([(p.physics.rotation.pitch, p.physics.rotation.yaw, p.physics.rotation.roll) for p in players], np.float32)
                full = np.concatenate(basis(*np.ascontiguousarray(rot.T)), -1)
                table[:n, [k for k, _ in self.basis_cols]] = full[:, [j for _, j in self.basis_cols]]
            if self.held_cols:
                table[:n, [k for k, _ in self.held_cols]] = held[:n, [j for _, j in self.held_cols]]
        ball = packet.balls[0].physics if len(packet.balls) else None
        out[a:b] = [1. if name == 'present' else BALL_FIELDS[name](ball) for name in self.ball] if ball is not None else 0.
        pads = out[b:c].reshape(len(self.pad_map), -1)
        live = {'timer': lambda: [packet.boost_pads[i].timer for i in self.pad_map],
                'is_active': lambda: [packet.boost_pads[i].is_active for i in self.pad_map]}
        for k, name in enumerate(self.pads):
            pads[:, k] = live[name]() if name in live else self.pad_static[name]
        out[c:] = [GAME_FIELDS[name](packet) for name in self.game]
