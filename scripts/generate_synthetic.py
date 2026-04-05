#!/usr/bin/env python3
"""Generate a small synthetic dataset for PiLoT smoke testing.

Creates image pairs with known 6-DoF poses and depth maps
using random 3D scenes rendered via projection.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUTPUT = "/mnt/forge-data/datasets/pilot_synthetic"
N_FRAMES = 200  # small subset for smoke test
W, H = 320, 240
FX, FY = 250.0, 250.0
CX, CY = W / 2, H / 2


def random_pose(altitude_range=(50, 200), pitch_range=(20, 70)):
    """Generate a random UAV-like camera pose."""
    from scipy.spatial.transform import Rotation

    x = np.random.uniform(-500, 500)
    y = np.random.uniform(-500, 500)
    z = np.random.uniform(*altitude_range)
    pitch = np.radians(np.random.uniform(*pitch_range))
    yaw = np.radians(np.random.uniform(0, 360))
    roll = np.radians(np.random.uniform(-5, 5))

    R = Rotation.from_euler("xyz", [pitch, yaw, roll]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T.astype(np.float32)


def render_depth_and_image(T, ground_points, ground_colors):
    """Render a synthetic view of ground points from pose T."""
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
    T_inv = np.linalg.inv(T)
    R = T_inv[:3, :3]
    t = T_inv[:3, 3]

    # Transform points to camera frame
    pts_cam = (R @ ground_points.T).T + t
    z = pts_cam[:, 2]
    valid = z > 1.0

    # Project
    u = (FX * pts_cam[valid, 0] / z[valid] + CX).astype(np.int32)
    v = (FY * pts_cam[valid, 1] / z[valid] + CY).astype(np.int32)
    d = z[valid]

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[in_bounds], v[in_bounds], d[in_bounds]
    colors = ground_colors[valid][in_bounds]

    # Z-buffer
    depth = np.zeros((H, W), dtype=np.float32)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(len(u)):
        if depth[v[i], u[i]] == 0 or d[i] < depth[v[i], u[i]]:
            depth[v[i], u[i]] = d[i]
            image[v[i], u[i]] = colors[i]

    # Fill holes with dilation
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel, iterations=2)
    depth_mask = (depth > 0).astype(np.uint8)
    depth = cv2.dilate(depth, kernel, iterations=2) * cv2.dilate(
        depth_mask, kernel, iterations=2
    ).astype(np.float32)

    # Add noise
    noise = np.random.randn(H, W, 3).astype(np.float32) * 5
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, depth


def main():
    np.random.seed(42)

    # Create output dirs
    for d in ["images", "depth", "poses", "splits"]:
        os.makedirs(os.path.join(OUTPUT, d), exist_ok=True)

    # Generate random ground plane with features
    n_ground = 50000
    ground_points = np.random.uniform(-1000, 1000, (n_ground, 3)).astype(np.float32)
    ground_points[:, 2] = np.random.uniform(-2, 2, n_ground)  # near z=0 ground
    # Add some elevated structures
    n_buildings = 5000
    buildings = np.random.uniform(-500, 500, (n_buildings, 3)).astype(np.float32)
    buildings[:, 2] = np.random.uniform(0, 50, n_buildings)
    ground_points = np.vstack([ground_points, buildings])

    # Random colors
    ground_colors = np.random.randint(50, 200, (len(ground_points), 3), dtype=np.uint8)
    # Ground is brownish-green
    ground_colors[:n_ground, 1] = np.clip(
        ground_colors[:n_ground, 1].astype(int) + 30, 0, 255
    ).astype(np.uint8)

    # Generate trajectory (smooth path)
    frame_ids = []
    poses = []
    base_pose = random_pose()
    for i in range(N_FRAMES):
        # Smooth trajectory with small perturbations
        if i == 0:
            T = base_pose.copy()
        else:
            T = poses[-1].copy()
            T[0, 3] += np.random.uniform(-5, 5)  # x drift
            T[1, 3] += np.random.uniform(-5, 5)  # y drift
            T[2, 3] += np.random.uniform(-1, 1)  # altitude jitter

        poses.append(T)
        frame_id = f"{i:06d}"
        frame_ids.append(frame_id)

        # Render
        image, depth = render_depth_and_image(T, ground_points, ground_colors)

        # Save
        cv2.imwrite(os.path.join(OUTPUT, "images", f"{frame_id}.png"), image)
        np.save(os.path.join(OUTPUT, "depth", f"{frame_id}.npy"), depth)
        np.save(os.path.join(OUTPUT, "poses", f"{frame_id}.npy"), T)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{N_FRAMES} frames")

    # Save intrinsics
    with open(os.path.join(OUTPUT, "intrinsics.json"), "w") as f:
        json.dump({"fx": FX, "fy": FY, "cx": CX, "cy": CY, "width": W, "height": H}, f)

    # Save splits (90/5/5)
    n_train = int(N_FRAMES * 0.9)
    n_val = int(N_FRAMES * 0.05)
    with open(os.path.join(OUTPUT, "splits", "train.txt"), "w") as f:
        f.write("\n".join(frame_ids[:n_train]))
    with open(os.path.join(OUTPUT, "splits", "val.txt"), "w") as f:
        f.write("\n".join(frame_ids[n_train:n_train + n_val]))
    with open(os.path.join(OUTPUT, "splits", "test.txt"), "w") as f:
        f.write("\n".join(frame_ids[n_train + n_val:]))

    print(f"Done! {N_FRAMES} frames -> {OUTPUT}")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {N_FRAMES - n_train - n_val}")


if __name__ == "__main__":
    main()
