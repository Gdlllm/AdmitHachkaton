"""CMU motion capture (BVH conversion) -> compact joint positions at 30 fps.

Reads BVH files straight from the downloaded zip and stores, per motion, the
world positions (metres, Y up, rest pose facing +Z, left = +X) of the 31 BVH
joints plus 6 end sites, and the head's forward/up axes. MediaPipe-style
landmarks are derived later (simulate.py), so the mapping can change without
re-reading 3 GB of text.

  python cmu_to_joints.py cmu-bvh.zip out.npz
"""
import sys, zipfile, numpy as np
from concurrent.futures import ProcessPoolExecutor

SCALE = (1.0 / 0.45) * 2.54 / 100.0  # CMU length unit -> metres
TARGET_FPS = 30.0

def parse(text):
    lines = text.split('\n')
    names, parents, offsets, channels, stack, ends = [], [], [], [], [], []
    i = 0
    while not lines[i].strip().startswith('MOTION'):
        tok = lines[i].split()
        if tok and tok[0] in ('ROOT', 'JOINT'):
            names.append(tok[1]); parents.append(stack[-1] if stack else -1); offsets.append(None); channels.append([])
            stack.append(len(names) - 1)
        elif tok and tok[0] == 'End':
            ends.append([stack[-1], None]); stack.append('end')
        elif tok and tok[0] == 'OFFSET':
            off = [float(v) for v in tok[1:4]]
            if stack[-1] == 'end': ends[-1][1] = off
            else: offsets[stack[-1]] = off
        elif tok and tok[0] == 'CHANNELS':
            channels[stack[-1]] = tok[2:]
        elif tok and tok[0] == '}':
            stack.pop()
        i += 1
    frames = int(lines[i + 1].split(':')[1]); frame_time = float(lines[i + 2].split(':')[1])
    step = max(1, round(1.0 / (frame_time * TARGET_FPS)))
    rows = [l for l in lines[i + 3:i + 3 + frames] if l.strip()][::step]
    data = np.array([np.array(r.split(), dtype=np.float64) for r in rows])
    return names, parents, np.array(offsets), channels, ends, data, frame_time * step

def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    m = np.zeros(a.shape + (3, 3)); m[..., 0, 0] = m[..., 1, 1] = m[..., 2, 2] = 1
    i, j = {'X': (1, 2), 'Y': (2, 0), 'Z': (0, 1)}[axis]
    m[..., i, i] = c; m[..., j, j] = c; m[..., i, j] = -s; m[..., j, i] = s
    return m

def forward_kinematics(names, parents, offsets, channels, ends, data):
    n = len(data); col = 0
    G = np.zeros((len(names), n, 3, 3)); P = np.zeros((len(names), n, 3))
    for j, chans in enumerate(channels):
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy(); pos = np.zeros((n, 3))
        for c in chans:
            v = data[:, col]; col += 1
            if c.endswith('position'): pos[:, 'XYZ'.index(c[0])] = v
            else: R = R @ rot(c[0], v)
        if parents[j] < 0:
            G[j] = R; P[j] = pos + offsets[j]
        else:
            p = parents[j]; G[j] = G[p] @ R; P[j] = P[p] + G[p] @ offsets[j]
    E = np.stack([P[j] + G[j] @ np.array(off) for j, off in ends])
    return P, E, G

def convert(args):
    zpath, member = args
    try:
        with zipfile.ZipFile(zpath) as z: text = z.read(member).decode('latin1')
        names, parents, offsets, channels, ends, data, dt = parse(text)
        if len(data) < 30: return member, None
        P, E, G = forward_kinematics(names, parents, offsets, channels, ends, data)
        head = names.index('Head')
        forward = G[head] @ np.array([0, 0, 1.0]); up = G[head] @ np.array([0, 1.0, 0])
        points = np.concatenate([P, E]).transpose(1, 0, 2) * SCALE  # (frames, 37, 3)
        return member, dict(points=points.astype(np.float32), head=np.stack([forward, up], 1).astype(np.float32),
                            dt=dt, names=names + [f'{names[j]}_end' for j, _ in ends])
    except Exception as error:
        return member, repr(error)

if __name__ == '__main__':
    zpath, out = sys.argv[1], sys.argv[2]
    with zipfile.ZipFile(zpath) as z: members = sorted(m for m in z.namelist() if m.endswith('.bvh'))
    points, heads, starts, clips, names, skipped = [], [], [0], [], None, []
    with ProcessPoolExecutor(8) as ex:
        for member, result in ex.map(convert, [(zpath, m) for m in members], chunksize=8):
            if not isinstance(result, dict): skipped.append((member, result)); continue
            if names is None: names = result['names']
            if result['names'] != names: skipped.append((member, 'different skeleton')); continue
            points.append(result['points']); heads.append(result['head']); starts.append(starts[-1] + len(result['points']))
            clips.append(member.split('/')[-1][:-4])
    np.savez(out, points=np.concatenate(points), head=np.concatenate(heads), starts=np.array(starts),
             clips=np.array(clips), names=np.array(names))
    print(len(clips), 'clips,', starts[-1], 'frames at 30 fps,', len(skipped), 'skipped')
    for s in skipped[:10]: print('  skipped', s)
