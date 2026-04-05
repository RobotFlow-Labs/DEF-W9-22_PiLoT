"""PiLoT utilities: config loading, seeding, coordinate transforms."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

try:
    import tomli
except ImportError:
    try:
        import tomllib as tomli  # Python 3.11+
    except ImportError:
        tomli = None


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict[str, Any]:
    """Load TOML config file.

    Args:
        path: path to .toml config file.

    Returns:
        Parsed config dictionary.
    """
    if tomli is None:
        raise ImportError("tomli is required: uv pip install tomli")

    with open(path, "rb") as f:
        cfg = tomli.load(f)
    cfg["_source"] = str(path)
    return cfg


# ---------------------------------------------------------------------------
# Coordinate transforms (WGS84 / ECEF / ENU)
# ---------------------------------------------------------------------------

# WGS84 ellipsoid constants
WGS84_A = 6_378_137.0  # semi-major axis (m)
WGS84_F = 1 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # eccentricity squared


def geodetic_to_ecef(
    lat: float, lon: float, alt: float
) -> tuple[float, float, float]:
    """Convert WGS84 geodetic (lat, lon, alt) to ECEF (x, y, z).

    Args:
        lat: latitude in degrees.
        lon: longitude in degrees.
        alt: altitude in meters above ellipsoid.

    Returns:
        (x, y, z) in meters.
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_lon = np.sin(lon_r)
    cos_lon = np.cos(lon_r)

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - WGS84_E2) + alt) * sin_lat

    return float(x), float(y), float(z)


def ecef_to_geodetic(
    x: float, y: float, z: float
) -> tuple[float, float, float]:
    """Convert ECEF (x, y, z) to WGS84 geodetic (lat, lon, alt).

    Uses iterative method (Bowring).
    """
    lon = np.degrees(np.arctan2(y, x))

    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.degrees(np.arctan2(z, p * (1 - WGS84_E2)))

    for _ in range(5):  # converges in ~3 iterations
        lat_r = np.radians(lat)
        sin_lat = np.sin(lat_r)
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
        lat = np.degrees(np.arctan2(z + WGS84_E2 * N * sin_lat, p))

    lat_r = np.radians(lat)
    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat ** 2)
    alt = p / cos_lat - N if abs(cos_lat) > 1e-10 else abs(z) / abs(sin_lat) - N * (1 - WGS84_E2)

    return float(lat), float(lon), float(alt)


def ecef_to_enu(
    x: float, y: float, z: float,
    lat0: float, lon0: float, alt0: float,
) -> tuple[float, float, float]:
    """Convert ECEF to local ENU (East-North-Up) coordinates.

    Args:
        x, y, z: ECEF coordinates.
        lat0, lon0, alt0: reference origin in geodetic.

    Returns:
        (east, north, up) in meters.
    """
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0

    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    sin_lat = np.sin(lat0_r)
    cos_lat = np.cos(lat0_r)
    sin_lon = np.sin(lon0_r)
    cos_lon = np.cos(lon0_r)

    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    return float(east), float(north), float(up)


# ---------------------------------------------------------------------------
# Kalman filter for pose prediction
# ---------------------------------------------------------------------------

class ConstantVelocityKalmanFilter:
    """Simple constant-velocity Kalman filter for 6-DoF pose prediction.

    State: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw]
    Used by the rendering thread to predict the next camera pose.
    """

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.state = np.zeros(12, dtype=np.float64)
        self.P = np.eye(12, dtype=np.float64) * 100.0  # covariance

        # Process noise
        self.Q = np.eye(12, dtype=np.float64)
        self.Q[:6, :6] *= 0.1
        self.Q[6:, 6:] *= 1.0

        # Measurement noise
        self.R = np.eye(6, dtype=np.float64) * 0.5

        self.initialized = False

    def predict(self) -> np.ndarray:
        """Predict next state. Returns (6,) [x, y, z, roll, pitch, yaw]."""
        F = np.eye(12, dtype=np.float64)
        F[:6, 6:] = np.eye(6) * self.dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        return self.state[:6].copy()

    def update(self, measurement: np.ndarray):
        """Update with measurement (6,) [x, y, z, roll, pitch, yaw]."""
        if not self.initialized:
            self.state[:6] = measurement
            self.initialized = True
            return

        H = np.zeros((6, 12), dtype=np.float64)
        H[:6, :6] = np.eye(6)

        y = measurement - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P

    def get_pose_matrix(self) -> np.ndarray:
        """Get current state as 4x4 pose matrix."""
        from scipy.spatial.transform import Rotation

        pose = self.state[:6]
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pose[:3]
        T[:3, :3] = Rotation.from_euler("xyz", pose[3:], degrees=False).as_matrix()
        return T
