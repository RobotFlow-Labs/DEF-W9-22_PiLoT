"""PiLoT dataset loaders.

Paper: arXiv 2603.20778
Training data: synthetic image pairs with 6-DoF pose + depth ground truth.
Evaluation data: query images with GT poses.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def fourier_noise(image: np.ndarray, severity: float = 0.1) -> np.ndarray:
    """Add high-frequency Fourier noise to image.

    Simulates sensor noise and rendering artifacts for domain randomization.
    """
    h, w, c = image.shape
    noise = np.random.randn(h, w, c).astype(np.float32)

    # Apply FFT, boost high frequencies, IFFT
    for ch in range(c):
        f = np.fft.fft2(noise[:, :, ch])
        # High-pass emphasis
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        high_pass = 1 - np.exp(-(x * x + y * y) / (2 * (min(h, w) * 0.3) ** 2))
        f_shifted = np.fft.fftshift(f) * high_pass
        noise[:, :, ch] = np.fft.ifft2(np.fft.ifftshift(f_shifted)).real

    noise = noise / (noise.std() + 1e-8) * severity
    return np.clip(image.astype(np.float32) / 255.0 + noise, 0, 1) * 255


def photometric_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
) -> np.ndarray:
    """Random photometric jitter (brightness, contrast, saturation)."""
    img = image.astype(np.float32)

    # Brightness
    img += np.random.uniform(-brightness, brightness) * 255

    # Contrast
    gray = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).mean()
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    img = gray + alpha * (img - gray)

    # Saturation
    hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 + np.random.uniform(-saturation, saturation)
    img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return np.clip(img, 0, 255).astype(np.uint8)


def perturb_pose(
    T: np.ndarray,
    t_range: tuple[float, float] = (5.0, 15.0),
    r_range: tuple[float, float] = (5.0, 15.0),
) -> np.ndarray:
    """Add random noise to a 4x4 pose matrix.

    Args:
        T: (4, 4) pose matrix.
        t_range: (min, max) translation noise in meters.
        r_range: (min, max) rotation noise in degrees.

    Returns:
        T_noisy: (4, 4) perturbed pose.
    """
    from scipy.spatial.transform import Rotation

    # Translation noise
    t_mag = np.random.uniform(*t_range)
    t_dir = np.random.randn(3)
    t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-8) * t_mag

    # Rotation noise
    r_mag = np.random.uniform(*r_range)
    r_axis = np.random.randn(3)
    r_axis = r_axis / (np.linalg.norm(r_axis) + 1e-8)
    r_angle = np.radians(r_mag)
    R_noise = Rotation.from_rotvec(r_axis * r_angle).as_matrix()

    T_noisy = T.copy()
    T_noisy[:3, :3] = R_noise @ T[:3, :3]
    T_noisy[:3, 3] = T[:3, 3] + t_dir

    return T_noisy


# ---------------------------------------------------------------------------
# Synthetic Training Dataset
# ---------------------------------------------------------------------------

class PiLoTSyntheticDataset(Dataset):
    """PiLoT synthetic training dataset.

    Expected directory structure:
        root/
          images/          -- RGB images (*.png or *.jpg)
          depth/           -- depth maps (*.npy or *.png 16-bit)
          poses/           -- 4x4 pose matrices (*.npy or *.txt)
          intrinsics.json  -- camera intrinsics {fx, fy, cx, cy, width, height}
          splits/
            train.txt      -- list of frame IDs for training
            val.txt        -- list of frame IDs for validation
            test.txt       -- list of frame IDs for testing

    Each sample returns a pair of consecutive frames for pixel-to-3D registration
    training, with the reference frame's pose perturbed to simulate the initial
    estimate that JNGO must refine.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: tuple[int, int] = (512, 384),
        augment: bool = True,
        pose_noise_t: tuple[float, float] = (5.0, 15.0),
        pose_noise_r: tuple[float, float] = (5.0, 15.0),
        fourier_noise_severity: float = 0.1,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size  # (W, H)
        self.augment = augment and split == "train"
        self.pose_noise_t = pose_noise_t
        self.pose_noise_r = pose_noise_r
        self.fourier_severity = fourier_noise_severity

        # Load frame IDs
        split_file = self.root / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file) as f:
                self.frame_ids = [line.strip() for line in f if line.strip()]
        else:
            # Auto-discover from images directory
            img_dir = self.root / "images"
            if img_dir.exists():
                exts = {".png", ".jpg", ".jpeg"}
                self.frame_ids = sorted(
                    p.stem for p in img_dir.iterdir() if p.suffix.lower() in exts
                )
            else:
                self.frame_ids = []

        # Load intrinsics
        intr_path = self.root / "intrinsics.json"
        if intr_path.exists():
            with open(intr_path) as f:
                intr = json.load(f)
            self.intrinsics = np.array([
                [intr["fx"], 0, intr["cx"]],
                [0, intr["fy"], intr["cy"]],
                [0, 0, 1],
            ], dtype=np.float32)
        else:
            # Default intrinsics (will be overridden by config)
            self.intrinsics = np.array([
                [500, 0, image_size[0] / 2],
                [0, 500, image_size[1] / 2],
                [0, 0, 1],
            ], dtype=np.float32)

    def __len__(self) -> int:
        return max(len(self.frame_ids) - 1, 0)

    def _load_image(self, frame_id: str) -> np.ndarray:
        """Load RGB image, resize."""
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.root / "images" / f"{frame_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    return img
        # Return black image as fallback
        return np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)

    def _load_depth(self, frame_id: str) -> np.ndarray:
        """Load depth map."""
        npy_path = self.root / "depth" / f"{frame_id}.npy"
        if npy_path.exists():
            depth = np.load(str(npy_path))
            h, w = self.image_size[1], self.image_size[0]
            if depth.shape != (h, w):
                depth = cv2.resize(depth.astype(np.float32), self.image_size)
            return depth.astype(np.float32)

        png_path = self.root / "depth" / f"{frame_id}.png"
        if png_path.exists():
            depth = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                depth = depth.astype(np.float32) / 1000.0  # mm to meters
                depth = cv2.resize(depth, self.image_size)
                return depth

        return np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)

    def _load_pose(self, frame_id: str) -> np.ndarray:
        """Load 4x4 pose matrix."""
        npy_path = self.root / "poses" / f"{frame_id}.npy"
        if npy_path.exists():
            return np.load(str(npy_path)).astype(np.float32)

        txt_path = self.root / "poses" / f"{frame_id}.txt"
        if txt_path.exists():
            return np.loadtxt(str(txt_path)).reshape(4, 4).astype(np.float32)

        return np.eye(4, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get training sample: query + reference pair.

        Returns:
            dict with keys:
                query_image: (3, H, W) normalized query frame
                ref_image: (3, H, W) normalized reference frame
                T_query: (4, 4) GT query pose
                T_ref: (4, 4) GT reference pose
                T_init: (4, 4) noisy initial pose estimate (for JNGO)
                depth: (H, W) depth map for reference frame
                intrinsics: (3, 3) camera intrinsics
        """
        query_id = self.frame_ids[idx + 1]
        ref_id = self.frame_ids[idx]

        # Load data
        query_img = self._load_image(query_id)
        ref_img = self._load_image(ref_id)
        depth = self._load_depth(ref_id)
        T_query = self._load_pose(query_id)
        T_ref = self._load_pose(ref_id)

        # Augmentation
        if self.augment:
            query_img = photometric_jitter(query_img)
            query_img = fourier_noise(query_img, self.fourier_severity).astype(np.uint8)

        # Create noisy initial pose
        T_init = perturb_pose(T_query, self.pose_noise_t, self.pose_noise_r)

        # Normalize images to [0, 1] and convert to CHW tensors
        query_t = torch.from_numpy(query_img).float().permute(2, 0, 1) / 255.0
        ref_t = torch.from_numpy(ref_img).float().permute(2, 0, 1) / 255.0

        return {
            "query_image": query_t,
            "ref_image": ref_t,
            "T_query": torch.from_numpy(T_query),
            "T_ref": torch.from_numpy(T_ref),
            "T_init": torch.from_numpy(T_init.astype(np.float32)),
            "depth": torch.from_numpy(depth),
            "intrinsics": torch.from_numpy(self.intrinsics.copy()),
        }


# ---------------------------------------------------------------------------
# Evaluation Dataset
# ---------------------------------------------------------------------------

class PiLoTEvalDataset(Dataset):
    """PiLoT evaluation dataset.

    Expected structure:
        root/
          images/          -- query RGB images
          poses/           -- GT 4x4 pose matrices
          intrinsics.json  -- camera intrinsics

    For evaluation, we load individual frames with their GT poses.
    The reference view and geo-anchors are generated from the 3D map at runtime.
    """

    def __init__(
        self,
        root: str,
        image_size: tuple[int, int] = (512, 384),
    ):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size

        # Discover frames
        img_dir = self.root / "images"
        if img_dir.exists():
            exts = {".png", ".jpg", ".jpeg"}
            self.frame_ids = sorted(
                p.stem for p in img_dir.iterdir() if p.suffix.lower() in exts
            )
        else:
            self.frame_ids = []

        # Load intrinsics
        intr_path = self.root / "intrinsics.json"
        if intr_path.exists():
            with open(intr_path) as f:
                intr = json.load(f)
            self.intrinsics = np.array([
                [intr["fx"], 0, intr["cx"]],
                [0, intr["fy"], intr["cy"]],
                [0, 0, 1],
            ], dtype=np.float32)
        else:
            self.intrinsics = np.array([
                [500, 0, image_size[0] / 2],
                [0, 500, image_size[1] / 2],
                [0, 0, 1],
            ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get evaluation sample.

        Returns:
            dict with keys:
                image: (3, H, W) query frame
                T_gt: (4, 4) ground truth pose
                intrinsics: (3, 3) camera intrinsics
                frame_id: str
        """
        frame_id = self.frame_ids[idx]

        # Load image
        img = None
        for ext in [".png", ".jpg", ".jpeg"]:
            path = self.root / "images" / f"{frame_id}{ext}"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                break
        if img is None:
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, self.image_size)

        # Load GT pose
        T_gt = np.eye(4, dtype=np.float32)
        for ext, loader in [(".npy", np.load), (".txt", lambda p: np.loadtxt(p).reshape(4, 4))]:
            path = self.root / "poses" / f"{frame_id}{ext}"
            if path.exists():
                T_gt = loader(str(path)).astype(np.float32)
                break

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0

        return {
            "image": img_t,
            "T_gt": torch.from_numpy(T_gt),
            "intrinsics": torch.from_numpy(self.intrinsics.copy()),
            "frame_id": frame_id,
        }


# ---------------------------------------------------------------------------
# Geo-anchor generation from depth
# ---------------------------------------------------------------------------

try:
    from pilot.backends.cuda.kernels import cuda_available, cuda_depth_to_geo_anchors
except ImportError:
    cuda_available = lambda: False  # noqa: E731
    cuda_depth_to_geo_anchors = None


def depth_to_geo_anchors(
    depth: torch.Tensor,
    T_ref: torch.Tensor,
    intrinsics: torch.Tensor,
    num_anchors: int = 500,
    min_depth: float = 1.0,
) -> torch.Tensor:
    """Back-project depth-valid pixels to 3D world-frame geo-anchors.

    Args:
        depth: (B, H, W) reference depth map.
        T_ref: (B, 4, 4) reference camera pose (world-from-camera).
        intrinsics: (B, 3, 3) camera intrinsics.
        num_anchors: number of anchors to sample.
        min_depth: minimum valid depth.

    Returns:
        anchors: (B, N, 3) world-frame 3D points.
    """
    # Use CUDA-accelerated version when available
    if cuda_available() and depth.is_cuda and cuda_depth_to_geo_anchors is not None:
        return cuda_depth_to_geo_anchors(depth, T_ref, intrinsics, num_anchors, min_depth)

    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)

    anchors_list = []
    for b in range(B):
        d = depth[b]  # (H, W)
        valid = d > min_depth
        valid_indices = valid.nonzero(as_tuple=False)  # (M, 2)

        if valid_indices.shape[0] < num_anchors:
            # Pad with zeros if not enough valid pixels
            n_valid = valid_indices.shape[0]
            if n_valid == 0:
                anchors_list.append(torch.zeros(num_anchors, 3, device=device, dtype=dtype))
                continue
            # Repeat to fill
            repeat_idx = torch.randint(0, n_valid, (num_anchors,), device=device)
            sampled = valid_indices[repeat_idx]
        else:
            # Random sample
            perm = torch.randperm(valid_indices.shape[0], device=device)[:num_anchors]
            sampled = valid_indices[perm]

        # Back-project
        rows, cols = sampled[:, 0], sampled[:, 1]
        d_vals = d[rows, cols]  # (N,)
        pixels = uv1[rows, cols]  # (N, 3)

        K_inv = torch.linalg.inv(intrinsics[b])  # (3, 3)
        pts_cam = (K_inv @ pixels.T).T * d_vals.unsqueeze(-1)  # (N, 3)

        # Transform to world frame
        R = T_ref[b, :3, :3]
        t = T_ref[b, :3, 3]
        pts_world = (R @ pts_cam.T).T + t.unsqueeze(0)  # (N, 3)

        anchors_list.append(pts_world)

    return torch.stack(anchors_list, dim=0)  # (B, N, 3)
