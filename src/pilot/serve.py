"""PiLoT serving node for ANIMA platform.

Implements PiLoTNode(AnimaNode) for Docker/ROS2 deployment.
Subscribes to camera images, publishes 6-DoF pose estimates.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from anima_serve.node import AnimaNode
except ImportError:
    # Fallback: define minimal base class for development
    class AnimaNode:
        """Minimal AnimaNode stub for development without anima_serve."""

        def __init__(self):
            self.module_name = "pilot"
            self.module_version = "0.1.0"

        def setup_inference(self):
            raise NotImplementedError

        def process(self, input_data: Any) -> Any:
            raise NotImplementedError

        def get_status(self) -> dict:
            return {}

from pilot.model import PiLoTModelConfig, PiLoTSystem
from pilot.utils import load_config


class PiLoTNode(AnimaNode):
    """PiLoT inference node for ANIMA serving platform.

    Setup:
        1. Downloads/loads model weights
        2. Initializes PiLoT feature net + JNGO optimizer
        3. Listens for camera images
        4. Returns 6-DoF pose estimates

    API Endpoints:
        POST /predict: image -> pose (4x4 matrix)
        GET /health: module health status
        GET /ready: readiness check
        GET /info: module information
    """

    def __init__(self, config_path: str = "configs/paper.toml"):
        super().__init__()
        self.config_path = config_path
        self.model: PiLoTSystem | None = None
        self.device = "cpu"
        self.ready = False

    def setup_inference(self):
        """Load model weights and prepare for inference."""
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load config
        cfg = load_config(self.config_path)

        # Build model
        model_cfg = PiLoTModelConfig(
            backbone=cfg["model"]["backbone"],
            backbone_depth=cfg["model"]["backbone_depth"],
            backbone_pretrained=False,  # load from checkpoint instead
            feature_channels=cfg["model"]["feature_channels"],
            pyramid_channels=cfg["model"]["pyramid_channels"],
            uncertainty_heads=cfg["model"]["uncertainty_heads"],
            num_hypotheses=cfg["jngo"]["num_hypotheses"],
            pitch_range_deg=cfg["jngo"]["pitch_range_deg"],
            yaw_range_deg=cfg["jngo"]["yaw_range_deg"],
            angle_step_deg=cfg["jngo"]["angle_step_deg"],
            translation_std=cfg["jngo"]["translation_std"],
            lm_iterations=[
                cfg["jngo"]["lm_iterations_coarse"],
                cfg["jngo"]["lm_iterations_mid"],
                cfg["jngo"]["lm_iterations_fine"],
            ],
            num_geo_anchors=cfg["jngo"]["num_geo_anchors"],
        )
        self.model = PiLoTSystem(model_cfg).to(self.device)
        self.model.eval()

        # Load weights if available
        weight_path = Path("/data/weights/best.pth")
        if weight_path.exists():
            ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            print(f"[PiLoT] Loaded weights from {weight_path}")

        self.ready = True
        print(f"[PiLoT] Inference ready on {self.device}")

    def process(self, input_data: dict) -> dict:
        """Run inference on input image.

        Args:
            input_data: dict with:
                'image': numpy array (H, W, 3) BGR or (3, H, W) tensor
                'T_pred': optional (4, 4) predicted pose
                'intrinsics': optional (3, 3) camera matrix
                'geo_anchors': optional (N, 3) 3D points

        Returns:
            dict with:
                'pose': (4, 4) estimated pose as list
                'features': multi-scale feature shapes
                'latency_ms': inference time
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        t0 = time.time()

        # Parse input
        image = input_data.get("image")
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                # HWC -> CHW, normalize
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features
            feats = self.model.extract_features(image)

            # If full pipeline inputs provided, run JNGO
            T_pred = input_data.get("T_pred")
            geo_anchors = input_data.get("geo_anchors")
            intrinsics = input_data.get("intrinsics")

            if T_pred is not None and geo_anchors is not None and intrinsics is not None:
                T_pred_t = torch.tensor(T_pred, dtype=torch.float32).unsqueeze(0).to(self.device)
                anchors_t = torch.tensor(
                    geo_anchors, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                K_t = torch.tensor(intrinsics, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Need reference features (normally from rendered view)
                # For API mode, use query features as reference (self-localization)
                T_est = self.model.jngo(T_pred_t, feats, feats, anchors_t, K_t)
                pose = T_est[0].cpu().numpy().tolist()
            else:
                pose = np.eye(4).tolist()

        latency = (time.time() - t0) * 1000

        return {
            "pose": pose,
            "feature_shapes": [
                [f.shape[1], f.shape[2], f.shape[3]] for f, _ in feats
            ],
            "latency_ms": round(latency, 2),
        }

    def get_status(self) -> dict:
        """Module-specific status fields."""
        return {
            "model_loaded": self.model is not None,
            "device": self.device,
            "ready": self.ready,
        }
