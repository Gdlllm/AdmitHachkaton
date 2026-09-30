"""Synthetic training windows for the body-sense model.

Motion capture (exact 3D, every joint known even behind the back) is filmed by
a random virtual camera and then damaged the way the live pipeline damages it:
MediaPipe-style visibility, jitter and guessed occluded joints, left/right
labels that flip around side views and often describe a back as a front, a
face detector that only sees large, frontal faces, and the surface network's
own body direction with its front/back mistakes and latency.

Everything is batched torch code, so windows are generated on the training
device. Features must match src/dense/inference/body-sense.js exactly.
"""
import math, numpy as np, torch

T = 32                      # frames per window (receptive field is 31)
N_POSE = 33
PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
         (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FLIP = list(range(N_POSE))
for a, b in PAIRS: FLIP[a], FLIP[b] = b, a
# Which mirror chain each pair follows: 0 face, 1 upper body (shoulders, arms), 2 lower body.
PAIR_GROUP = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
FEATURES_V8 = N_POSE * 4 + 3 + 4 + 4 + 1 + 1   # pose x,y,visibility,in-frame | motion | face | network | dt | valid

# CMU BVH point indices (see cmu_to_joints.py names).
C = dict(LUpLeg=2, LLeg=3, LFoot=4, LToe=5, RUpLeg=7, RLeg=8, RFoot=9, RToe=10, Head=16, LArm=18, LForeArm=19, LHand=20,
         RArm=25, RForeArm=26, RHand=27, LToeEnd=31, RToeEnd=32, HeadEnd=33, LIndexEnd=34, LThumbEnd=35, RIndexEnd=36, RThumbEnd=37)


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


# MediaPipe's landmarks sit differently from mocap joints. Measured on stock
# clips (torso = mid-shoulder to mid-hip): shoulder width 0.66, upper arm 0.52,
# thigh 0.83 torsos, nose 0.42 torsos above the shoulders at ear height, eyes
# ~3 cm above the nose, ears 0.21 torsos apart. The CMU rig has a 30% shorter
# torso and a head frame pitched 21° down on average.
TORSO_STRETCH = .4
HEAD_PITCH_FIX = math.radians(21)
# The 11 face points (nose, eyes, ears, mouth) are not model inputs. MediaPipe
# draws a confident face on the back of the head as well, in ways no simulation
# here matched: models fed with them learned "face points shown → facing the
# camera" (v7) or "ears shown → back" (v4) and failed on real video.
FACE_POINTS = False
# v9 ("lift"): the surface network's 3D joints are an input, and the model
# outputs the whole 3D skeleton (root-relative, in torso lengths). False
# reproduces the v8 feature layout.
NET_JOINTS = True
# MediaPipe points the surface network also has (POSE_TO_MHR in refine-pose.js).
NET_JOINT_IDS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
FEATURES = FEATURES_V8 + (len(NET_JOINT_IDS) * 3 if NET_JOINTS else 0)   # + network joints (x,y,z each)


def cmu_to_mp33(points, head):
    """points (N,38,3) metres Y up; head (N,2,3) forward/up -> (N,33,3), head centre (N,3), head forward (N,3)."""
    P = points.astype(np.float64); f0, u0 = _norm(head[:, 0].astype(np.float64)), _norm(head[:, 1].astype(np.float64))
    f = _norm(f0 * math.cos(HEAD_PITCH_FIX) + u0 * math.sin(HEAD_PITCH_FIX))
    u = _norm(u0 * math.cos(HEAD_PITCH_FIX) - f0 * math.sin(HEAD_PITCH_FIX))
    l = np.cross(u, f)
    # Ear-level centre of the head: the end of the CMU head bone.
    hc = P[:, C['HeadEnd']].copy()
    out = np.zeros((len(P), N_POSE, 3))
    def face(df, du, dl): return hc + df * f + du * u + dl * l
    out[:, 0] = face(.095, 0, 0)
    out[:, 1], out[:, 2], out[:, 3] = face(.08, .027, .017), face(.075, .027, .032), face(.068, .027, .047)
    out[:, 4], out[:, 5], out[:, 6] = face(.08, .027, -.017), face(.075, .027, -.032), face(.068, .027, -.047)
    out[:, 7], out[:, 8] = face(-.01, 0, .058), face(-.01, 0, -.058)
    out[:, 9], out[:, 10] = face(.085, -.035, .025), face(.085, -.035, -.025)
    out[:, 11], out[:, 12] = P[:, C['LArm']], P[:, C['RArm']]
    out[:, 13], out[:, 14] = P[:, C['LForeArm']], P[:, C['RForeArm']]
    out[:, 15], out[:, 16] = P[:, C['LHand']], P[:, C['RHand']]
    for side, wrist, index, thumb, (pinky_i, index_i, thumb_i) in (
            ('L', 'LHand', 'LIndexEnd', 'LThumbEnd', (17, 19, 21)), ('R', 'RHand', 'RIndexEnd', 'RThumbEnd', (18, 20, 22))):
        w = P[:, C[wrist]]; a = _norm(P[:, C[index]] - w)
        t = P[:, C[thumb]] - w; b = _norm(t - np.sum(t * a, -1, keepdims=True) * a)
        out[:, index_i] = w + .09 * a + .015 * b
        out[:, pinky_i] = w + .08 * a - .03 * b
        out[:, thumb_i] = w + .045 * a + .045 * b
    pelvis = (P[:, C['LUpLeg']] + P[:, C['RUpLeg']]) / 2
    for i, name in ((23, 'LUpLeg'), (24, 'RUpLeg')):   # MediaPipe hips sit a little wider and higher
        out[:, i] = pelvis + 1.1 * (P[:, C[name]] - pelvis) + np.array([0, .02, 0])
    out[:, 25], out[:, 26] = P[:, C['LLeg']], P[:, C['RLeg']]
    out[:, 27], out[:, 28] = P[:, C['LFoot']], P[:, C['RFoot']]
    for heel, ankle, knee, toe in ((29, 'LFoot', 'LLeg', 'LToe'), (30, 'RFoot', 'RLeg', 'RToe')):
        down = _norm(P[:, C[ankle]] - P[:, C[knee]]); fd = _norm(P[:, C[toe]] - P[:, C[ankle]])
        out[:, heel] = P[:, C[ankle]] - .045 * fd + .055 * down
    out[:, 31], out[:, 32] = P[:, C['LToeEnd']], P[:, C['RToeEnd']]
    # Lengthen the torso: the whole upper body (arms, head) moves up along it.
    torso = (out[:, 11] + out[:, 12]) / 2 - (out[:, 23] + out[:, 24]) / 2
    upper = list(range(0, 23))
    out[:, upper] += TORSO_STRETCH * torso[:, None]
    hc = hc + TORSO_STRETCH * torso
    return out.astype(np.float32), hc.astype(np.float32), f.astype(np.float32)


# Capsules for self-occlusion: (a, b, radius, group). Groups: 0 torso, 1 head,
# 2 left arm, 3 right arm, 4 left leg, 5 right leg. 'mid_sh', 'mid_hip', 'hc'
# are derived points.
CAPSULES = [('mid_sh', 'mid_hip', .12, 0), (11, 23, .08, 0), (12, 24, .08, 0), ('hc', 'hc', .10, 1), ('hc', 'mid_sh', .06, 1),
            (11, 13, .05, 2), (13, 15, .04, 2), (15, 19, .035, 2), (12, 14, .05, 3), (14, 16, .04, 3), (16, 20, .035, 3),
            (23, 25, .075, 4), (25, 27, .05, 4), (29, 31, .04, 4), (24, 26, .075, 5), (26, 28, .05, 5), (30, 32, .04, 5)]
JOINT_GROUP = [1] * 11 + [0, 0, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 0, 0, 4, 5, 4, 5, 4, 5, 4, 5]
# Joints never hidden by their own neighbouring capsules.
OWN = {11: {0, 2}, 12: {0, 3}, 23: {0, 4}, 24: {0, 5}}


def _segment_distance(p0, p1, q0, q1):
    """Closest distance between segments p0-p1 and q0-q1 (broadcast, last dim 3) and the parameter along p."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    denom = (a * e - b * b).clamp_min(1e-9)
    s = ((b * f - c * e) / denom).clamp(0, 1)
    t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0, 1)
    s = ((b * t - c) / a.clamp_min(1e-9)).clamp(0, 1)
    return ((p0 + d1 * s[..., None]) - (q0 + d2 * t[..., None])).norm(dim=-1), s


class MotionBank:
    def __init__(self, npz_path, device, holdout_subjects=()):
        d = np.load(npz_path)
        mp, hc, hf = cmu_to_mp33(d['points'], d['head'])
        starts = d['starts']; clips = d['clips']
        subject = np.array([int(str(c).split('_')[0]) for c in clips])
        keep = np.isin(subject, list(holdout_subjects))
        self.device = device
        self.data = torch.from_numpy(np.concatenate([mp, hc[:, None], hf[:, None]], 1)).to(device)   # (N,35,3)
        self.train_ranges = [(starts[i], starts[i + 1]) for i in range(len(clips)) if not keep[i]]
        self.val_ranges = [(starts[i], starts[i + 1]) for i in range(len(clips)) if keep[i]]

    def sample(self, batch, split='train', generator=None):
        ranges = self.train_ranges if split == 'train' else self.val_ranges
        lengths = np.array([b - a for a, b in ranges], dtype=np.float64)
        rng = np.random.default_rng(None if generator is None else generator)
        idx = rng.choice(len(ranges), size=batch, p=lengths / lengths.sum())
        # 30 fps base; stride 1 or 2 plus random dropped frames (variable dt).
        stride = np.where(rng.random(batch) < .3, 2, 1)
        skip_p = rng.uniform(0, .3, batch)
        steps = stride[:, None] + (rng.random((batch, T - 1)) < skip_p[:, None])
        offsets = np.concatenate([np.zeros((batch, 1), int), np.cumsum(steps, 1)], 1)
        frames = np.zeros((batch, T), np.int64)
        for k, i in enumerate(idx):
            a, b = ranges[i]
            span = offsets[k, -1]
            if b - a <= span:  # short clip: slow down to fit
                off = np.round(offsets[k] * (b - a - 1) / max(span, 1)).astype(int)
                frames[k] = a + off; steps[k] = np.maximum(np.diff(off), 1)
            else:
                frames[k] = a + rng.integers(0, b - a - span) + offsets[k]
        dt = np.concatenate([np.full((batch, 1), 1 / 30), steps / 30.0], 1)
        return self.data[torch.from_numpy(frames).to(self.device)], torch.from_numpy(dt).float().to(self.device), rng


def _rotation_y(angle):
    c, s = torch.cos(angle), torch.sin(angle); z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1), torch.stack([-s, z, c], -1)], -2)


_CONSTANTS = {}
def _const(value, device):
    if isinstance(value, torch.Tensor): return value
    key = (float(value), str(device))
    if key not in _CONSTANTS: _CONSTANTS[key] = torch.tensor(float(value), device=device)
    return _CONSTANTS[key]


_ALLOWED = {}
def _allowed_mask(device):
    """(capsules, joints): may this capsule hide this joint? Not its own limb."""
    key = str(device)
    if key not in _ALLOWED:
        m = torch.tensor([[JOINT_GROUP[j] != group and not (j in OWN and group in OWN[j]) for j in range(N_POSE)]
                          for _, _, _, group in CAPSULES])
        _ALLOWED[key] = m.to(device)
    return _ALLOWED[key]


class _Rand:
    """Random numbers generated on the training device (no host transfers)."""
    def __init__(self, device, seed):
        self.dev = device; self.g = torch.Generator(device=device); self.g.manual_seed(int(seed))
    def u(self, lo, hi, *shape): return lo + (hi - lo) * torch.rand(shape, generator=self.g, device=self.dev)
    def n(self, *shape): return torch.randn(shape, generator=self.g, device=self.dev)
    def chance(self, p, *shape): return torch.rand(shape, generator=self.g, device=self.dev) < p
    def ints(self, lo, hi, *shape): return torch.randint(lo, hi, shape, generator=self.g, device=self.dev)
    def arcsine(self, *shape): return torch.sin(math.pi * torch.rand(shape, generator=self.g, device=self.dev) / 2) ** 2  # Beta(.5,.5)


def simulate(bank, batch, split='train', seed=None, raw=False):
    """Returns dict(features (B,T,F), forward (B,3), back (B,), swap (B,16), joints (B,33,2), weight (B,),
    plus diagnostics)."""
    dev = bank.device
    # Python scalars in torch.where would be copied to the device (and sync) on every call.
    where = lambda c, x, y: torch.where(c, _const(x, dev), _const(y, dev))
    seq, dt, rng = bank.sample(batch, split, seed)
    r = _Rand(dev, rng.integers(2 ** 62))
    B = batch
    pts, hc, hf = seq[:, :, :33].clone(), seq[:, :, 33].clone(), seq[:, :, 34].clone()
    # --- augment: mirror the motion, random heading, scale, centre on the current pelvis
    mirror = r.chance(.5, B)
    flipped = pts[:, :, FLIP] * torch.tensor([-1., 1, 1], device=dev)
    pts = where(mirror[:, None, None, None], flipped, pts)
    reflect = torch.stack([where(mirror, -1., 1.), torch.ones(B, device=dev), torch.ones(B, device=dev)], -1)[:, None]
    hc, hf = hc * reflect, hf * reflect
    R = _rotation_y(r.u(0, 2 * math.pi, B))
    scale = r.u(.85, 1.15, B)
    pelvis_now = (pts[:, -1, 23] + pts[:, -1, 24]) / 2
    origin = torch.stack([pelvis_now[:, 0], torch.zeros(B, device=dev), pelvis_now[:, 2]], -1)
    pts = torch.einsum('bij,btkj->btki', R, pts - origin[:, None, None]) * scale[:, None, None, None]
    hc = torch.einsum('bij,btj->bti', R, hc - origin[:, None]) * scale[:, None, None]
    hf = torch.einsum('bij,btj->bti', R, hf)
    # --- camera: static during the window, aimed at the body at the current frame
    close = r.chance(.4, B)
    sh_now, hip_now, head_now = (pts[:, -1, 11] + pts[:, -1, 12]) / 2, (pts[:, -1, 23] + pts[:, -1, 24]) / 2, hc[:, -1]
    target_far = hip_now + (sh_now - hip_now) * r.u(0, .9, B)[:, None]
    target_close = sh_now + (head_now - sh_now) * r.u(-.4, .7, B)[:, None]
    target = where(close[:, None], target_close, target_far)
    dist = where(close, r.u(.55, 1.8, B), r.u(2.0, 6.0, B))
    azim = r.u(0, 2 * math.pi, B)
    height = (target[:, 1] + where(close, r.u(-.3, .35, B), r.u(-.6, .7, B))).clamp_min(.25)
    cam = torch.stack([target[:, 0] + dist * torch.sin(azim), height, target[:, 2] + dist * torch.cos(azim)], -1)
    portrait = r.chance(.15, B)
    W = where(portrait, 720., 1280.); H = where(portrait, 1280., 720.)
    hfov = torch.deg2rad(r.u(50, 80, B)) * where(portrait, .62, 1.)
    focal = (W / 2) / torch.tan(hfov / 2)
    aim = target + r.u(-1, 1, B, 3) * torch.tensor([.25, .15, .25], device=dev) * dist[:, None] * .4
    fwd = aim - cam; fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    up = torch.tensor([0., 1, 0], device=dev).expand(B, 3)
    right = torch.cross(fwd, up, dim=-1); right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    down = torch.cross(fwd, right, dim=-1)
    roll = torch.deg2rad(r.u(-4, 4, B))[:, None]
    right, down = right * torch.cos(roll) + down * torch.sin(roll), down * torch.cos(roll) - right * torch.sin(roll)
    Rc = torch.stack([right, down, fwd], 1)                           # rows: camera X right, Y down, Z forward
    P = torch.einsum('bij,btkj->btki', Rc, pts - cam[:, None, None])
    HC = torch.einsum('bij,btj->bti', Rc, hc - cam[:, None])
    HF = torch.einsum('bij,btj->bti', Rc, hf)
    Z = P[..., 2].clamp_min(.05)
    u = focal[:, None, None] * P[..., 0] / Z + W[:, None, None] / 2
    v = focal[:, None, None] * P[..., 1] / Z + H[:, None, None] / 2
    x, y = u / W[:, None, None], v / H[:, None, None]
    in_frame = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1) & (P[..., 2] > .1)
    # --- occlusion (camera at the origin in camera space)
    named = {'mid_sh': (P[..., 11, :] + P[..., 12, :]) / 2, 'mid_hip': (P[..., 23, :] + P[..., 24, :]) / 2, 'hc': HC}
    get = lambda k: named[k] if isinstance(k, str) else P[..., k, :]
    # A capsule hides a joint when the joint's image point falls inside the
    # capsule's projected outline and the capsule is nearer to the camera.
    occluded = torch.zeros(P.shape[:-1], dtype=torch.bool, device=dev)
    allowed = _allowed_mask(dev)                                              # (K,33)
    jz = P[..., 2].clamp_min(.05); jx, jy = P[..., 0] / jz, P[..., 1] / jz
    for k, (a, b, radius, group) in enumerate(CAPSULES):
        pa, pb = get(a), get(b)
        za, zb = pa[..., 2].clamp_min(.05), pb[..., 2].clamp_min(.05)
        ax, ay, bx, by = pa[..., 0] / za, pa[..., 1] / za, pb[..., 0] / zb, pb[..., 1] / zb
        dx, dy = (bx - ax)[..., None], (by - ay)[..., None]
        t = (((jx - ax[..., None]) * dx + (jy - ay[..., None]) * dy) / (dx * dx + dy * dy).clamp_min(1e-9)).clamp(0, 1)
        cx, cy = ax[..., None] + t * dx, ay[..., None] + t * dy
        cz = za[..., None] + t * (zb - za)[..., None]
        inside = (jx - cx) ** 2 + (jy - cy) ** 2 < (radius / cz) ** 2
        occluded |= inside & (cz + radius < jz) & allowed[k]
    # Face points are hidden when they face away from the camera; ears (at the
    # sides of the head) only on the far side of a profile: real MediaPipe shows
    # both ears on every frontal view.
    normal = P[..., :11, :] - HC[..., None, :]
    to_camera = -P[..., :11, :]
    facing_cam = (normal * to_camera).sum(-1) / (normal.norm(dim=-1) * to_camera.norm(dim=-1)).clamp_min(1e-9)
    limit = torch.tensor([-.05] * 7 + [-.95] * 2 + [-.05] * 2, device=dev)
    occluded[..., :11] |= facing_cam < limit
    # Side views: the far shoulder/hip is behind the torso.
    for a, b in ((11, 12), (23, 24)):
        gap2d = torch.hypot(u[..., a] - u[..., b], v[..., a] - v[..., b])
        depth = P[..., a, 2] - P[..., b, 2]
        torso_px = focal[:, None] * .45 / Z[..., a]
        close_pair = gap2d < .35 * torso_px
        occluded[..., a] |= close_pair & (depth > .08)
        occluded[..., b] |= close_pair & (depth < -.08)
    visible = in_frame & ~occluded
    # Framing decides how front-biased MediaPipe and the network are on backs
    # (stock clips: labels wrong on 1% of full-body backs, 6% of close crops and
    # 62% of head-and-shoulders views; the network on 11%, 21% and 90%).
    tight = ~(in_frame[..., 23] | in_frame[..., 24])                          # (B,T) hips out of the picture
    # --- MediaPipe-like scores
    shape = visible.shape
    vis = where(visible, r.u(.75, 1.0, *shape), where(in_frame, r.u(.03, .75, *shape), r.u(0, .35, *shape)))
    presence = where(in_frame, r.u(.7, 1.0, *shape), r.u(.05, .65, *shape))
    dropout = r.chance(r.u(0, .04, B)[:, None, None], *shape)
    vis = where(dropout, r.u(0, .5, *shape), vis)
    # --- positional noise in pixels: temporally correlated, larger for guessed joints
    torso_px = (focal[:, None] * .5 / HC[..., 2].clamp_min(.2))[..., None]            # (B,T,1)
    sig_vis = r.u(.004, .02, B)[:, None, None] * torso_px
    sig_occ = r.u(.03, .25, B)[:, None, None] * torso_px
    rho = r.u(.3, .9, B)[:, None, None]
    white = r.n(B, T, N_POSE, 2)
    noise = [white[:, 0]]
    for t in range(1, T): noise.append(rho * noise[-1] + (1 - rho ** 2).sqrt() * white[:, t])
    noise = torch.stack(noise, 1)
    guessed_bias = r.n(B, 1, N_POSE, 2) * .6
    sig = where(visible, sig_vis, sig_occ)
    un = u + (noise[..., 0] + where(visible, 0., guessed_bias[..., 0])) * sig
    vn = v + (noise[..., 1] + where(visible, 0., guessed_bias[..., 1])) * sig
    # --- left/right mirror state: flickers around side views, sticks to a
    # sequence-specific habit on backs, rare on fronts.
    torso_fwd = _torso_forward(P)                                          # (B,T,3) camera axes
    to_cam_dir = -(named['mid_sh'] + named['mid_hip']) / 2
    to_cam_dir = to_cam_dir / to_cam_dir.norm(dim=-1, keepdim=True)
    cos_face = (torso_fwd * to_cam_dir).sum(-1)                              # 1 facing camera, -1 back
    front, back = cos_face > math.cos(math.radians(55)), cos_face < math.cos(math.radians(125))
    # Mirroring habit on clear backs: rare with the hips in view, common in head-and-shoulders.
    habit = torch.where(tight, r.u(.35, .95, B)[:, None], (.08 * r.u(0, 1, B) ** 3)[:, None])        # (B,T)
    side_rate = r.u(0, .25, B)[:, None]; back_rate = r.u(.02, .15, B)[:, None]
    independent = r.chance(.12, B, 3)                                        # groups with their own chain
    draws = where(independent[:, None], r.u(0, 1, B, T, 3), r.u(0, 1, B, T)[..., None].expand(B, T, 3))
    p_front_on, p_front_stay = torch.tensor(.003, device=dev), torch.tensor(.35, device=dev)
    states = [draws[:, 0] < where(front[:, 0, None], .01, where(back[:, 0, None], habit[:, 0, None], .5))]
    for t in range(1, T):
        prev = states[-1]; f_t, b_t, h_t = front[:, t, None], back[:, t, None], habit[:, t, None]
        p_mirror = where(f_t, where(prev, p_front_stay, p_front_on),
                   where(b_t, where(prev, 1 - back_rate * (1 - h_t), back_rate * h_t),
                               where(prev, 1 - side_rate, side_rate)))
        states.append(draws[:, t] < p_mirror)
    states = torch.stack(states, 1)                                          # (B,T,3)
    group_of_joint = torch.tensor([0] * 11 + [1] * 12 + [2] * 10, device=dev)
    mirrored = states[:, :, group_of_joint]                                   # (B,T,33)
    flip = torch.tensor(FLIP, device=dev)
    def apply(a): return where(mirrored, a[:, :, flip], a)
    # Back-facing heads: MediaPipe often draws the face on the back of the head.
    # (on stock clips 91-95% of back views still show face points).
    hallucinate = r.chance(r.u(.85, 1., B), B)[:, None] & (cos_face < math.cos(math.radians(100)))
    face_pts = P[..., :11, :]
    along = ((face_pts - HC[..., None, :]) * HF[..., None, :]).sum(-1, keepdim=True)
    reflected = face_pts - 2 * along * HF[..., None, :]
    # ...and lower than a real face would be.
    head_up = torch.cross(torch.cross(HF, -P[..., 11, :] + P[..., 12, :], dim=-1), HF, dim=-1)
    head_up = head_up / head_up.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    head_up = torch.where((head_up[..., 1:2] < 0), head_up, -head_up)          # camera Y is down: "up" has Y < 0
    reflected = reflected - .08 * head_up[..., None, :]
    Zr = reflected[..., 2].clamp_min(.05)
    ur = focal[:, None, None] * reflected[..., 0] / Zr + W[:, None, None] / 2 + noise[..., :11, 0] * sig_vis
    vr = focal[:, None, None] * reflected[..., 1] / Zr + H[:, None, None] / 2 + noise[..., :11, 1] * sig_vis
    un = torch.cat([where(hallucinate[..., None], ur, un[..., :11]), un[..., 11:]], -1)
    vn = torch.cat([where(hallucinate[..., None], vr, vn[..., :11]), vn[..., 11:]], -1)
    # Guessed face points are shown with fair confidence; the ears less often
    # than on a real front (stock clips: ears shown on 92% of backs, 100% of fronts).
    face_vis = torch.cat([r.u(.5, 1., B, T, 7), r.u(.4, 1., B, T, 2), r.u(.5, 1., B, T, 2)], -1)
    vis = torch.cat([where(hallucinate[..., None] & in_frame[..., :11], face_vis, vis[..., :11]), vis[..., 11:]], -1)
    un, vn, vis_m, pres_m = apply(un), apply(vn), apply(vis), apply(presence)
    # Stabilizer-like smoothing (lag) and the drawConfidence rule of compactPose.
    alpha = r.u(.35, 1.0, B)[:, None]
    su, sv = [un[:, 0]], [vn[:, 0]]
    for t in range(1, T):
        su.append(alpha * un[:, t] + (1 - alpha) * su[-1]); sv.append(alpha * vn[:, t] + (1 - alpha) * sv[-1])
    un, vn = torch.stack(su, 1), torch.stack(sv, 1)
    xn, yn = un / W[:, None, None], vn / H[:, None, None]
    # The web landmarker reports no presence, so the stabilizer only tests visibility.
    inside = (xn >= 0) & (xn <= 1) & (yn >= 0) & (yn <= 1)
    shown = inside & (vis_m >= .45)
    vis_m = where(shown, vis_m, torch.zeros_like(vis_m))
    # --- face detector (short range): frontal, large enough, in frame; bursty misses
    head_to_cam = -HC / HC.norm(dim=-1, keepdim=True)
    face_cos = (HF * head_to_cam).sum(-1)
    # The web detector (short-range BlazeFace on the whole frame) only finds
    # large faces, and often none at all (light, hair, glasses): stock clips
    # with frontal faces at 2-3 m give almost no detections.
    max_angle = torch.deg2rad(r.u(30, 65, B))
    head_px = focal[:, None] * .1 / HC[..., 2].clamp_min(.05) / H[:, None]
    min_size = torch.where(r.chance(.3, B), torch.full((B,), 9.0, device=dev), r.u(.06, .16, B))[:, None]
    head_x, head_y = x[..., 0], y[..., 0]
    in_view = (head_px > min_size) & (HC[..., 2] > .2) & (head_x > .02) & (head_x < .98) & (head_y > .02) & (head_y < .98)
    # A tracked face is kept to a wider angle than a new one is detected at.
    keep_angle = torch.deg2rad(r.u(5, 20, B))
    miss_rate = r.u(0, .5, B)
    mu = r.u(0, 1, B, T)
    miss, found = [mu[:, 0] < miss_rate], []
    found.append((face_cos[:, 0] > torch.cos(max_angle)) & in_view[:, 0] & ~miss[0])
    for t in range(1, T):
        miss.append(mu[:, t] < where(miss[-1], torch.tensor(.7, device=dev), miss_rate * .4))
        limit = max_angle + keep_angle * found[-1]
        found.append((face_cos[:, t] > torch.cos(limit)) & in_view[:, t] & ~miss[-1])
    face_found = torch.stack(found, 1)
    face_dir = HF + r.n(B, T, 3) * .07
    face_dir = face_dir / face_dir.norm(dim=-1, keepdim=True)
    face_feat = torch.cat([face_found[..., None].float(), face_dir * face_found[..., None]], -1)
    # --- the surface network's body direction: noisy, held between runs, late,
    # sometimes the depth-mirrored (front/back) interpretation.
    net_on = r.chance(.85, B)
    sig_net = torch.deg2rad(r.u(4, 20, B))[:, None, None]
    net_dir = torso_fwd + r.n(B, T, 3) * sig_net
    p_back_err = where(tight, r.u(.7, .99, B)[:, None], where(close[:, None], r.u(.05, .4, B)[:, None], r.u(0, .25, B)[:, None]))
    p_err = where(front, r.u(0, .02, B)[:, None], where(back, p_back_err, r.u(0, .3, B)[:, None]))
    eu = r.u(0, 1, B, T)
    err = [eu[:, 0] < p_err[:, 0]]
    for t in range(1, T):
        err.append(eu[:, t] < where(err[-1], 1 - .15 * (1 - p_err[:, t]), .15 * p_err[:, t]))
    err = torch.stack(err, 1)
    net_dir = where(err[..., None], net_dir * torch.tensor([1., 1, -1], device=dev), net_dir)
    period = r.ints(1, 7, B)[:, None]; delay = r.ints(0, 4, B)[:, None]
    tt = torch.arange(T, device=dev)[None]
    src = ((tt - delay).clamp_min(0) // period) * period
    net_dir = torch.gather(net_dir, 1, src[..., None].expand(B, T, 3))
    net_dir = net_dir / net_dir.norm(dim=-1, keepdim=True)
    net_feat = torch.cat([net_on[:, None, None].float().expand(B, T, 1), net_dir * net_on[:, None, None]], -1)
    # --- the surface network's 3D joints: root-relative camera axes in its own
    # torso lengths; same runs, delay and front/back mistakes as its direction.
    # A front/back mistake is the depth mirror of the body with left and right relabelled.
    root = (P[..., 23, :] + P[..., 24, :]) / 2
    P_rel = P - root[..., None, :]
    ids = torch.tensor(NET_JOINT_IDS, device=dev)
    mirror_body = (P_rel * torch.tensor([1., 1, -1], device=dev))[:, :, torch.tensor(FLIP, device=dev)]
    nj = where(err[..., None, None], mirror_body, P_rel)[:, :, ids]
    torso_len = ((P[..., 11, :] + P[..., 12, :]) / 2 - root).norm(dim=-1).clamp_min(.1)          # (B,T)
    sig_j = r.u(.02, .06, B)[:, None, None, None] * where(in_frame[:, :, ids], 1., 2.5)[..., None]
    nj = nj + r.n(B, T, len(NET_JOINT_IDS), 3) * sig_j + r.n(B, 1, len(NET_JOINT_IDS), 3) * .02
    nj = torch.gather(nj, 1, src[..., None, None].expand(B, T, len(NET_JOINT_IDS), 3))
    net_joints_raw = nj
    # As the browser does it: relative to the network body's own mid-hip, in its own torso lengths.
    k = NET_JOINT_IDS.index
    n_root = (nj[..., k(23), :] + nj[..., k(24), :]) / 2
    n_torso = ((nj[..., k(11), :] + nj[..., k(12), :]) / 2 - n_root).norm(dim=-1).clamp_min(.1)
    nj = ((nj - n_root[..., None, :]) / n_torso[..., None, None]).clamp(-4, 4)
    net_joints_feat = (nj * net_on[:, None, None, None]).flatten(2)
    # --- features in the window frame of the current (last) sample
    # Track starts: the first frames of some windows do not exist yet.
    empty = r.ints(1, T - 1, B) * r.chance(.12, B)
    missing = tt < empty[:, None]
    pose_px = torch.stack([xn * W[:, None, None], yn * H[:, None, None]], -1)
    center, scale_px = frame_normalization(pose_px)                           # (B,T,2), (B,T)
    feat_pose = torch.cat([((pose_px - center[..., None, :]) / scale_px[..., None, None]).clamp(-6, 6), vis_m[..., None], inside[..., None].float()], -1)
    if not FACE_POINTS: feat_pose = torch.cat([torch.zeros_like(feat_pose[..., :11, :]), feat_pose[..., 11:, :]], -2)
    motion = motion_features(center, scale_px, missing)
    valid = torch.ones(B, T, 1, device=dev)
    features = torch.cat([feat_pose.flatten(2), motion, face_feat, net_feat, dt[..., None] * 10, valid]
                         + ([net_joints_feat] if NET_JOINTS else []), -1)
    features = torch.where(missing[..., None], torch.zeros_like(features), features)
    # --- targets at the current frame (true labels)
    true_px = torch.stack([u[:, -1], v[:, -1]], -1)
    joints = (true_px - center[:, -1, None]) / scale_px[:, -1, None, None]
    swap = states[:, -1][:, torch.tensor(PAIR_GROUP, device=dev)].float()
    # 3D skeleton at the last two frames, root-relative, in the current torso length.
    joints3d = P_rel[:, -1] / torso_len[:, -1, None, None]
    joints3d_prev = P_rel[:, -2] / torso_len[:, -1, None, None]
    # Only frames where the live pipeline would run the model: shoulders tracked.
    weight = (in_frame[:, -1, 11] & in_frame[:, -1, 12] & (vis_m[:, -1, 11] > 0) & (vis_m[:, -1, 12] > 0)).float()
    extra = dict(raw=dict(x=xn, y=yn, visibility=vis_m, presence=pres_m, face=face_feat, network=net_feat, dt=dt,
                          network_joints=net_joints_raw * net_on[:, None, None, None] if NET_JOINTS else None,
                          width=W, height=H, missing=missing)) if raw else {}
    return dict(**extra, features=features, forward=torso_fwd[:, -1], back=(cos_face[:, -1] < 0).float(), swap=swap,
                joints=joints.clamp(-6, 6), joints3d=joints3d, joints3d_prev=joints3d_prev, visible=visible[:, -1],
                in_frame=in_frame[:, -1], tight=tight[:, -1], weight=weight, cos_face=cos_face[:, -1],
                face_found=face_found[:, -1], net_forward=net_dir[:, -1], net_on=net_on, net_err=err[:, -1], close=close)


def _torso_forward(P):
    """Torso forward in camera axes from true-labelled camera-space joints (B,T,33,3)."""
    lr = (P[..., 11, :] + P[..., 23, :]) / 2 - (P[..., 12, :] + P[..., 24, :]) / 2
    up = (P[..., 11, :] + P[..., 12, :]) / 2 - (P[..., 23, :] + P[..., 24, :]) / 2
    # Camera axes have Y down; body 'up' is towards -Y, which up already encodes.
    f = torch.cross(lr, up, dim=-1)
    return f / f.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def frame_normalization(pose_px):
    """Per frame: centre = mid-shoulder, scale = a yaw-robust body size in pixels.
    Uses every landmark coordinate (MediaPipe always returns all 33)."""
    sh = (pose_px[..., 11, :] + pose_px[..., 12, :]) / 2
    hip = (pose_px[..., 23, :] + pose_px[..., 24, :]) / 2
    height = (sh - hip).norm(dim=-1)
    width = 1.3 * (pose_px[..., 11, :] - pose_px[..., 12, :]).norm(dim=-1)
    neck = 2.5 * (pose_px[..., 0, :] - sh).norm(dim=-1)
    return sh, torch.stack([height, width, neck], -1).amax(-1).clamp_min(8.0)


def motion_features(center, scale, missing):
    """Centre displacement since the previous frame (in current body sizes) and the
    log change of body size. Zero on a track's first frame."""
    prev_c = torch.cat([center[:, :1], center[:, :-1]], 1); prev_s = torch.cat([scale[:, :1], scale[:, :-1]], 1)
    first = torch.zeros_like(missing); first[:, 0] = True
    first = first | torch.cat([torch.ones_like(missing[:, :1]), missing[:, :-1]], 1)   # previous frame did not exist
    d = ((center - prev_c) / scale[..., None]).clamp(-3, 3)
    ls = torch.log(scale / prev_s).clamp(-1, 1)
    out = torch.cat([d, ls[..., None]], -1)
    return torch.where(first[..., None], torch.zeros_like(out), out)
