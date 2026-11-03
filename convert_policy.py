"""policy.pt2 -> the ONNX policy the botpack build runs on CPU with onnxruntime (no PyTorch in the bot).

    python convert_policy.py policy.pt2 policy

Writes <out>/manifest.json (policy.pt2's manifest) and <out>/<program>.onnx for the step program and each matchup's
program, then checks every ONNX program against the exported program on CPU over random inputs: decisions equal,
state within 1e-5. Needs the torch version the archive records, onnx, onnxscript and onnxruntime.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
import torch
import onnxruntime as ort
from torch.export.pt2_archive._package import load_pt2

FORMAT = 'question_mark_policy_v2'
NAMES = {'step': ['players', 'state', 'dt'], 'matchup': ['players', 'state', 'ball', 'pads', 'game', 'rows']}


def inputs(manifest, rng, cars):
    """Random tables shaped as the bot fills them; players rows get the manifest's columns, team 0/1."""
    m = int(manifest['max_players'])
    players = rng.normal(0, 1000, (m, len(manifest['players']))).astype(np.float32)
    for k, name in enumerate(manifest['players']):
        if name in ('present', 'team', 'is_supersonic', 'has_jumped', 'has_double_jumped', 'has_dodged', 'is_bot') \
                or name.startswith(('held.', 'last_input.')):
            players[:, k] = rng.integers(0, 2, m)
        elif name in ('air_state', 'spawn_id'):
            players[:, k] = rng.integers(0, 5, m)
        elif name in ('boost', 'demolished_timeout', 'dodge_elapsed', 'dodge_timeout'):
            players[:, k] = rng.uniform(0, 100 if name == 'boost' else 3, m)
        elif name in ('pitch', 'yaw', 'roll') or '.' in name:
            players[:, k] = rng.uniform(-1, 1, m)
    state = rng.uniform(0, 1, (m, int(manifest['state_width']))).astype(np.float32)
    ball = rng.normal(0, 1000, len(manifest['ball'])).astype(np.float32)
    ball[manifest['ball'].index('present')] = 1.
    pads = rng.uniform(0, 1, (len(manifest['pad_locations']), len(manifest['pads']))).astype(np.float32)
    game = np.array([3 if n == 'phase' else rng.uniform(0, 300) for n in manifest['game']], np.float32)
    rows = rng.choice(cars * 2, cars, replace=False).astype(np.int64)
    return {'players': players, 'state': state, 'ball': ball, 'pads': pads, 'game': game, 'rows': rows,
            'dt': np.array(rng.choice([0., 1 / 120, 8 / 120]), np.float32)}


def float_clamp_bounds(program):
    """clamp(x, 0, 1.25) on a float x -> clamp(x, 0., 1.25): same values, and the ONNX exporter's type promotion
    otherwise turns the int bound into a tensor that aten.clamp.default rejects."""
    for node in program.graph.nodes:
        if node.target is torch.ops.aten.clamp.default and node.args[0].meta['val'].dtype.is_floating_point:
            node.args = (node.args[0], *(float(a) if isinstance(a, int) else a for a in node.args[1:]))
    program.graph_module.recompile()
    return program


def every_matchup(manifest):
    """Matchups the archive leaves out (2v1, 1v0, 4v4, ...) play the policy of the bigger team's size, or the
    largest policy beyond it: 2v1 and 1v2 play 2v2, 4v4 plays 3v3."""
    matchups = manifest['matchups']
    sizes = sorted(int(k.split('v')[0]) for k in matchups if k.split('v')[0] == k.split('v')[1])
    for own in range(1, int(manifest['max_players']) + 1):
        for other in range(int(manifest['max_players']) - own + 1):
            size = next((s for s in sizes if s >= max(own, other)), sizes[-1])
            matchups.setdefault(f'{own}v{other}', matchups[f'{size}v{size}'])
    return manifest


def main(src, out):
    contents = load_pt2(src)
    manifest = every_matchup(json.loads(contents.extra_files['manifest.json']))
    if manifest.get('format') != FORMAT:
        raise ValueError(f"{src} is {manifest.get('format')}, not {FORMAT}")
    if manifest.get('torch') != torch.__version__.split('+')[0]:
        print(f"warning: {src} was exported with torch {manifest.get('torch')}, this is {torch.__version__}")
    manifest['converted_from'] = {'policy.pt2_sha256': hashlib.sha256(Path(src).read_bytes()).hexdigest(),
                                  'torch': torch.__version__, 'onnxruntime': ort.__version__}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    rng = np.random.default_rng(0)
    for name in sorted({manifest['step'], *manifest['matchups'].values()}):
        program = contents.exported_programs[name]
        kind = 'step' if name == manifest['step'] else 'matchup'
        path = out / f'{name}.onnx'
        module = program.module()  # the reference: the archive as saved
        onnx_program = torch.onnx.export(float_clamp_bounds(copy.deepcopy(program)), (), dynamo=True,
                                         input_names=NAMES[kind], optimize=True)
        for node in onnx_program.model.graph:  # the exporter's per-node stack traces and source paths
            node.metadata_props.clear()
        onnx_program.save(str(path))
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        worst = 0.
        for trial in range(200):
            x = inputs(manifest, rng, cars=1 + trial % 3)
            feed = {n: x[n] for n in NAMES[kind]}
            with torch.no_grad():
                want = module(*(torch.as_tensor(feed[n]) for n in NAMES[kind]))
            want = [w.numpy() for w in (want if isinstance(want, (tuple, list)) else [want])]
            got = session.run(None, feed)
            if kind == 'step':
                worst = max(worst, float(np.abs(got[0] - want[0]).max()))
            elif not np.array_equal(got[0], want[0]):
                raise AssertionError(f'{name}: decisions differ on trial {trial}:\n{got[0]}\n{want[0]}')
        if worst > 1e-5:
            raise AssertionError(f'{name}: state differs by {worst}')
        match = f'state max diff {worst:.2g}' if kind == 'step' else 'decisions equal'
        print(f'{name}: {path.stat().st_size / 1e6:.1f} MB, 200 random inputs: {match}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
