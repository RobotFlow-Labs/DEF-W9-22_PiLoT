"""Multi-dataset adapter for PiLoT training.

Combines real-world datasets to replace the unreleased PiLoT synthetic data:
- KITTI: 7481 images with DepthAnything depth + calibration (primary supervised)
- SERAPHIM UAV: 75K aerial images (self-supervised feature learning via pairs)
- DroneVehicle-night: 10K RGB+IR pairs (aerial self-supervised)

PiLoT needs: image pairs + depth + 6-DoF poses.
Strategy:
  1. KITTI provides depth + intrinsics. Consecutive frames = pairs.
     Poses synthesized from identity + noise (no GPS in KITTI det set).
  2. SERAPHIM provides aerial views. Augmented crops = pairs.
     Depth estimated from monocular cues (or set to constant altitude).
  3. DroneVehicle provides aerial RGB+IR. Cross-modal = pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from pilot.dataset import perturb_pose, photometric_jitter


class KITTIDepthDataset(Dataset):
    """KITTI dataset with DepthAnything depth maps for PiLoT training.

    Uses consecutive images as query/reference pairs.
    Calibration provides intrinsics; poses are synthesized.
    """

    def __init__(
        self,
        root: str = "/mnt/forge-data/datasets/kitti/training",
        split: str = "train",
        image_size: tuple[int, int] = (512, 384),
        augment: bool = True,
        max_samples: int = 0,
    ):
        self.root = Path(root)
        self.image_size = image_size  # (W, H)
        self.augment = augment and split == "train"

        img_dir = self.root / "image_2"
        depth_dir = self.root / "depth_anything"

        all_ids = sorted(
            p.stem for p in img_dir.glob("*.png")
            if (depth_dir / f"{p.stem}.npy").exists()
        )

        # Split
        n = len(all_ids)
        n_train = int(n * 0.9)
        n_val = int(n * 0.05)
        if split == "train":
            self.frame_ids = all_ids[:n_train]
        elif split == "val":
            self.frame_ids = all_ids[n_train:n_train + n_val]
        else:
            self.frame_ids = all_ids[n_train + n_val:]

        if max_samples > 0:
            self.frame_ids = self.frame_ids[:max_samples]

        # Load first calibration for intrinsics (same for all KITTI det)
        self.intrinsics = self._load_calib(all_ids[0] if all_ids else "000000")

    def _load_calib(self, frame_id: str) -> np.ndarray:
        calib_path = self.root / "calib" / f"{frame_id}.txt"
        if not calib_path.exists():
            return np.array([[707, 0, 604], [0, 707, 180], [0, 0, 1]], dtype=np.float32)
        with open(calib_path) as f:
            for line in f:
                if line.startswith("P2:"):
                    vals = [float(x) for x in line.strip().split()[1:]]
                    P = np.array(vals).reshape(3, 4)
                    K = P[:3, :3].astype(np.float32)
                    return K
        return np.array([[707, 0, 604], [0, 707, 180], [0, 0, 1]], dtype=np.float32)

    def __len__(self) -> int:
        return max(len(self.frame_ids) - 1, 0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        query_id = self.frame_ids[idx + 1]
        ref_id = self.frame_ids[idx]

        # Load images
        q_img = cv2.imread(str(self.root / "image_2" / f"{query_id}.png"))
        r_img = cv2.imread(str(self.root / "image_2" / f"{ref_id}.png"))
        q_img = cv2.resize(q_img, self.image_size) if q_img is not None else np.zeros(
            (self.image_size[1], self.image_size[0], 3), dtype=np.uint8
        )
        r_img = cv2.resize(r_img, self.image_size) if r_img is not None else np.zeros(
            (self.image_size[1], self.image_size[0], 3), dtype=np.uint8
        )

        # Load depth
        depth_path = self.root / "depth_anything" / f"{ref_id}.npy"
        if depth_path.exists():
            depth = np.load(str(depth_path)).astype(np.float32)
            depth = cv2.resize(depth, self.image_size)
            depth = np.clip(depth, 0, 100)  # clamp to reasonable range
            # Scale to meters (DepthAnything outputs relative — scale to ~50m mean)
            if depth.max() > 0:
                depth = depth / depth.max() * 80.0 + 5.0
        else:
            depth = np.ones((self.image_size[1], self.image_size[0]), dtype=np.float32) * 50.0

        # Synthesize poses (KITTI det has no sequential GPS)
        T_ref = np.eye(4, dtype=np.float32)
        T_query = perturb_pose(T_ref, t_range=(1.0, 5.0), r_range=(1.0, 5.0))
        T_init = perturb_pose(T_query, t_range=(5.0, 15.0), r_range=(5.0, 15.0))

        # Scale intrinsics to target resolution
        orig_w, orig_h = 1224, 370  # KITTI default
        sx = self.image_size[0] / orig_w
        sy = self.image_size[1] / orig_h
        K = self.intrinsics.copy()
        K[0, :] *= sx
        K[1, :] *= sy

        if self.augment:
            q_img = photometric_jitter(q_img)

        q_t = torch.from_numpy(q_img).float().permute(2, 0, 1) / 255.0
        r_t = torch.from_numpy(r_img).float().permute(2, 0, 1) / 255.0

        return {
            "query_image": q_t,
            "ref_image": r_t,
            "T_query": torch.from_numpy(T_query),
            "T_ref": torch.from_numpy(T_ref),
            "T_init": torch.from_numpy(T_init),
            "depth": torch.from_numpy(depth),
            "intrinsics": torch.from_numpy(K),
        }


class SeraphimUAVDataset(Dataset):
    """SERAPHIM UAV aerial dataset adapted for PiLoT training.

    Uses random crop pairs from the same image as query/reference,
    with synthesized depth (constant altitude) and pose perturbation.
    Ideal for learning aerial feature representations.
    """

    def __init__(
        self,
        root: str = "/mnt/forge-data/datasets/uav_detection/seraphim/train/images",
        split: str = "train",
        image_size: tuple[int, int] = (512, 384),
        augment: bool = True,
        altitude: float = 100.0,
        max_samples: int = 0,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.altitude = altitude

        exts = {".jpg", ".png", ".jpeg"}
        all_imgs = sorted(p for p in self.root.iterdir() if p.suffix.lower() in exts)

        n = len(all_imgs)
        n_train = int(n * 0.9)
        n_val = int(n * 0.05)
        if split == "train":
            self.images = all_imgs[:n_train]
        elif split == "val":
            self.images = all_imgs[n_train:n_train + n_val]
        else:
            self.images = all_imgs[n_train + n_val:]

        if max_samples > 0:
            self.images = self.images[:max_samples]

        # Synthetic UAV intrinsics
        self.intrinsics = np.array([
            [400, 0, image_size[0] / 2],
            [0, 400, image_size[1] / 2],
            [0, 0, 1],
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = cv2.imread(str(self.images[idx]))
        if img is None:
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

        h, w = img.shape[:2]
        W, H = self.image_size

        # Create two different crops as query/reference pair
        def random_crop(image, out_w, out_h):
            ih, iw = image.shape[:2]
            if iw < out_w or ih < out_h:
                return cv2.resize(image, (out_w, out_h))
            x = np.random.randint(0, max(iw - out_w, 1))
            y = np.random.randint(0, max(ih - out_h, 1))
            return cv2.resize(image[y:y + out_h, x:x + out_w], (out_w, out_h))

        q_img = random_crop(img, W, H)
        r_img = random_crop(img, W, H)

        if self.augment:
            q_img = photometric_jitter(q_img)

        # Constant depth (UAV altitude)
        depth = np.ones((H, W), dtype=np.float32) * self.altitude

        # Synthesize poses
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[2, 3] = self.altitude
        T_query = perturb_pose(T_ref, t_range=(2.0, 10.0), r_range=(2.0, 10.0))
        T_init = perturb_pose(T_query, t_range=(5.0, 15.0), r_range=(5.0, 15.0))

        q_t = torch.from_numpy(q_img).float().permute(2, 0, 1) / 255.0
        r_t = torch.from_numpy(r_img).float().permute(2, 0, 1) / 255.0

        return {
            "query_image": q_t,
            "ref_image": r_t,
            "T_query": torch.from_numpy(T_query),
            "T_ref": torch.from_numpy(T_ref),
            "T_init": torch.from_numpy(T_init),
            "depth": torch.from_numpy(depth),
            "intrinsics": torch.from_numpy(self.intrinsics.copy()),
        }


class DroneVehicleDataset(Dataset):
    """DroneVehicle-night RGB+IR pairs for cross-modal feature learning.

    Uses RGB as query, IR as reference (simulates day/night geo-loc).
    """

    def __init__(
        self,
        root: str = "/mnt/forge-data/shared_infra/datasets/dronevehicle_night/DroneVehicle-night",
        split: str = "train",
        image_size: tuple[int, int] = (512, 384),
        augment: bool = True,
        altitude: float = 80.0,
        max_samples: int = 0,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment and split == "train"
        self.altitude = altitude

        rgb_split = "train_img" if split == "train" else ("val_img" if split == "val" else "test_img")
        ir_split = "trainimgr" if split == "train" else ("valimgr" if split == "val" else "testimgr")
        rgb_dir = self.root / "rgb" / "image" / rgb_split
        ir_dir = self.root / "ir" / "img" / ir_split
        if not ir_dir.exists():
            ir_dir = self.root / "ir" / "image" / rgb_split

        if rgb_dir.exists() and ir_dir.exists():
            rgb_imgs = sorted(rgb_dir.glob("*.jpg"))
            ir_imgs = sorted(ir_dir.glob("*.jpg"))
            # Match by filename
            ir_names = {p.stem for p in ir_imgs}
            self.pairs = [
                (p, ir_dir / f"{p.stem}.jpg")
                for p in rgb_imgs if p.stem in ir_names
            ]
        else:
            self.pairs = []

        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]

        self.intrinsics = np.array([
            [350, 0, image_size[0] / 2],
            [0, 350, image_size[1] / 2],
            [0, 0, 1],
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb_path, ir_path = self.pairs[idx]
        W, H = self.image_size

        q_img = cv2.imread(str(rgb_path))
        r_img = cv2.imread(str(ir_path))
        q_img = cv2.resize(q_img, (W, H)) if q_img is not None else np.zeros((H, W, 3), np.uint8)
        r_img = cv2.resize(r_img, (W, H)) if r_img is not None else np.zeros((H, W, 3), np.uint8)

        if self.augment:
            q_img = photometric_jitter(q_img)

        depth = np.ones((H, W), dtype=np.float32) * self.altitude
        T_ref = np.eye(4, dtype=np.float32)
        T_ref[2, 3] = self.altitude
        T_query = perturb_pose(T_ref, t_range=(2.0, 8.0), r_range=(2.0, 8.0))
        T_init = perturb_pose(T_query, t_range=(5.0, 15.0), r_range=(5.0, 15.0))

        q_t = torch.from_numpy(q_img).float().permute(2, 0, 1) / 255.0
        r_t = torch.from_numpy(r_img).float().permute(2, 0, 1) / 255.0

        return {
            "query_image": q_t,
            "ref_image": r_t,
            "T_query": torch.from_numpy(T_query),
            "T_ref": torch.from_numpy(T_ref),
            "T_init": torch.from_numpy(T_init),
            "depth": torch.from_numpy(depth),
            "intrinsics": torch.from_numpy(self.intrinsics.copy()),
        }


def build_multi_dataset(
    split: str = "train",
    image_size: tuple[int, int] = (512, 384),
    augment: bool = True,
) -> ConcatDataset:
    """Build combined dataset from all available sources.

    Returns:
        ConcatDataset of KITTI + SERAPHIM + DroneVehicle.
    """
    datasets = []

    # KITTI (7K images with real depth)
    kitti = KITTIDepthDataset(split=split, image_size=image_size, augment=augment)
    if len(kitti) > 0:
        datasets.append(kitti)
        print(f"[DATA] KITTI: {len(kitti)} pairs")

    # SERAPHIM UAV (75K aerial images)
    seraphim = SeraphimUAVDataset(split=split, image_size=image_size, augment=augment)
    if len(seraphim) > 0:
        datasets.append(seraphim)
        print(f"[DATA] SERAPHIM: {len(seraphim)} pairs")

    # DroneVehicle-night (10K RGB+IR pairs)
    drone = DroneVehicleDataset(split=split, image_size=image_size, augment=augment)
    if len(drone) > 0:
        datasets.append(drone)
        print(f"[DATA] DroneVehicle: {len(drone)} pairs")

    if not datasets:
        raise RuntimeError("No datasets found! Check paths.")

    combined = ConcatDataset(datasets)
    print(f"[DATA] Combined {split}: {len(combined)} total pairs")
    return combined
