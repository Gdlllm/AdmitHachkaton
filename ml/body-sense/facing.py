"""Facing helpers shared by the analysis scripts.

label_facing: +1 when MediaPipe's left/right labels describe a body facing the
camera, -1 when they describe its back (image coordinates, y down).
mesh_facing: +1 when the fitted MHR torso faces the camera (camera Z forward).
"""
import numpy as np

def label_facing(pose):
    """pose: (..., 33, >=2) normalized image landmarks."""
    L = (pose[..., 11, :2] + pose[..., 23, :2]) / 2; R = (pose[..., 12, :2] + pose[..., 24, :2]) / 2
    up = (pose[..., 11, :2] + pose[..., 12, :2]) / 2 - (pose[..., 23, :2] + pose[..., 24, :2]) / 2
    lr = L - R
    cross = lr[..., 0] * up[..., 1] - lr[..., 1] * up[..., 0]
    return -np.sign(cross)

def mesh_facing(kp):
    """kp: (..., 70, 3) MHR keypoints in camera axes (X right, Y down, Z forward)."""
    lr = (kp[..., 5, :] + kp[..., 9, :]) / 2 - (kp[..., 6, :] + kp[..., 10, :]) / 2
    up = (kp[..., 5, :] + kp[..., 6, :]) / 2 - (kp[..., 9, :] + kp[..., 10, :]) / 2
    forward = np.cross(lr, up)
    # Body frame: left x up = forward in a right-handed frame. Camera axes are
    # right-handed too, so a body facing the camera has forward Z < 0.
    return -np.sign(forward[..., 2]), forward / (np.linalg.norm(forward, axis=-1, keepdims=True) + 1e-9)
