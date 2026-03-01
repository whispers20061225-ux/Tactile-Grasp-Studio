"""
Transformation utilities (minimal implementation).
"""

import numpy as np


def to_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation + translation to 4x4 homogeneous (if needed)."""
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        H = np.eye(4, dtype=float)
        H[:3, :] = matrix
        return H
    raise ValueError(f"Unsupported shape for to_homogeneous: {matrix.shape}")


def from_homogeneous(matrix: np.ndarray) -> np.ndarray:
    """Extract rotation (3x3) and translation (3x1) from 4x4."""
    if matrix.shape != (4, 4):
        raise ValueError(f"Unsupported shape for from_homogeneous: {matrix.shape}")
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    return R, t


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [x, y, z, w]."""
    M = np.array(matrix, dtype=float, copy=False)[:3, :3]
    q = np.empty((4,), dtype=float)
    t = np.trace(M)
    if t > 0:
        t = np.sqrt(t + 1.0) * 2
        q[3] = 0.25 * t
        q[0] = (M[2, 1] - M[1, 2]) / t
        q[1] = (M[0, 2] - M[2, 0]) / t
        q[2] = (M[1, 0] - M[0, 1]) / t
    else:
        i = np.argmax(np.diag(M))
        if i == 0:
            t = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[3] = (M[2, 1] - M[1, 2]) / t
            q[0] = 0.25 * t
            q[1] = (M[0, 1] + M[1, 0]) / t
            q[2] = (M[0, 2] + M[2, 0]) / t
        elif i == 1:
            t = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[3] = (M[0, 2] - M[2, 0]) / t
            q[0] = (M[0, 1] + M[1, 0]) / t
            q[1] = 0.25 * t
            q[2] = (M[1, 2] + M[2, 1]) / t
        else:
            t = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[3] = (M[1, 0] - M[0, 1]) / t
            q[0] = (M[0, 2] + M[2, 0]) / t
            q[1] = (M[1, 2] + M[2, 1]) / t
            q[2] = 0.25 * t
    return q


def rotation_matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)."""
    R = np.array(matrix, dtype=float, copy=False)[:3, :3]
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z], dtype=float)
