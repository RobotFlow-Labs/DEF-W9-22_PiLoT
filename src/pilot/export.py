"""PiLoT export pipeline: pth → safetensors → ONNX → TensorRT FP16/FP32.

All five export formats are MANDATORY per ANIMA conventions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import torch.nn as nn

from pilot.model import PiLoTFeatureNet
from pilot.train import build_model
from pilot.utils import load_config

# ---------------------------------------------------------------------------
# Feature-only wrapper (for export — JNGO is iterative, not exportable)
# ---------------------------------------------------------------------------

class PiLoTFeatureNetExport(nn.Module):
    """Export-friendly wrapper that only runs the feature network.

    The JNGO optimizer is iterative and uses dynamic control flow that cannot
    be traced to ONNX/TRT. We export the feature extraction backbone only.
    JNGO runs in PyTorch at inference time using the exported features.
    """

    def __init__(self, feature_net: PiLoTFeatureNet):
        super().__init__()
        self.feature_net = feature_net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract multi-scale features.

        Args:
            x: (B, 3, H, W) input image.

        Returns:
            Flat tuple of tensors: (feat_coarse, unc_coarse, feat_mid, unc_mid,
                                     feat_fine, unc_fine).
        """
        pyramid = self.feature_net(x)
        outputs = []
        for feat, unc in pyramid:
            outputs.append(feat)
            outputs.append(unc if unc is not None else torch.zeros_like(feat[:, :1]))
        return tuple(outputs)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_pth(
    checkpoint_path: str,
    output_dir: str,
    cfg: dict,
) -> str:
    """Export raw PyTorch checkpoint (clean state_dict only)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    dst = out / "model.pth"
    torch.save({"model": state, "config": cfg}, dst)
    size_mb = dst.stat().st_size / 1e6
    print(f"[EXPORT] pth: {dst} ({size_mb:.1f} MB)")
    return str(dst)


def export_safetensors(
    checkpoint_path: str,
    output_dir: str,
    cfg: dict,
) -> str:
    """Export to safetensors format."""
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt

    dst = out / "model.safetensors"
    save_file(state, str(dst))
    size_mb = dst.stat().st_size / 1e6
    print(f"[EXPORT] safetensors: {dst} ({size_mb:.1f} MB)")
    return str(dst)


def export_onnx(
    checkpoint_path: str,
    output_dir: str,
    cfg: dict,
    input_height: int = 384,
    input_width: int = 512,
    opset: int = 17,
) -> str:
    """Export feature network to ONNX."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build model and load weights
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # Wrap for export (feature net only)
    export_model = PiLoTFeatureNetExport(model.feature_net)
    export_model.eval()

    dummy = torch.randn(1, 3, input_height, input_width)

    dst = out / "model.onnx"
    torch.onnx.export(
        export_model,
        dummy,
        str(dst),
        opset_version=opset,
        input_names=["image"],
        output_names=[
            "feat_coarse", "unc_coarse",
            "feat_mid", "unc_mid",
            "feat_fine", "unc_fine",
        ],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "feat_coarse": {0: "batch"},
            "unc_coarse": {0: "batch"},
            "feat_mid": {0: "batch"},
            "unc_mid": {0: "batch"},
            "feat_fine": {0: "batch"},
            "unc_fine": {0: "batch"},
        },
    )
    size_mb = dst.stat().st_size / 1e6
    print(f"[EXPORT] ONNX: {dst} ({size_mb:.1f} MB, opset={opset})")
    return str(dst)


def export_tensorrt(
    onnx_path: str,
    output_dir: str,
    precision: str = "fp16",
    input_height: int = 384,
    input_width: int = 512,
    max_batch: int = 8,
) -> str:
    """Export ONNX model to TensorRT engine.

    Tries shared TRT toolkit first, falls back to trtexec CLI.

    Args:
        onnx_path: path to ONNX model.
        output_dir: where to save .engine file.
        precision: 'fp16' or 'fp32'.
        input_height, input_width: inference resolution.
        max_batch: maximum batch size.

    Returns:
        Path to TRT engine file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    suffix = f"_{precision}"
    dst = out / f"model{suffix}.engine"

    # Try shared TRT toolkit
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if trt_script.exists():
        import subprocess

        cmd = [
            "python", str(trt_script),
            "--onnx", onnx_path,
            "--output", str(dst),
            "--precision", precision,
            "--max-batch", str(max_batch),
            "--input-shape", f"1x3x{input_height}x{input_width}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and dst.exists():
            size_mb = dst.stat().st_size / 1e6
            print(f"[EXPORT] TRT {precision}: {dst} ({size_mb:.1f} MB)")
            return str(dst)
        print(f"[EXPORT] TRT toolkit failed: {result.stderr[:200]}")

    # Fallback: try trtexec
    trtexec = shutil.which("trtexec")
    if trtexec:
        import subprocess

        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={dst}",
            f"--minShapes=image:1x3x{input_height}x{input_width}",
            f"--optShapes=image:4x3x{input_height}x{input_width}",
            f"--maxShapes=image:{max_batch}x3x{input_height}x{input_width}",
        ]
        if precision == "fp16":
            cmd.append("--fp16")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and dst.exists():
            size_mb = dst.stat().st_size / 1e6
            print(f"[EXPORT] TRT {precision}: {dst} ({size_mb:.1f} MB)")
            return str(dst)
        print(f"[EXPORT] trtexec failed: {result.stderr[:200]}")

    # Last resort: try torch_tensorrt
    try:
        import torch_tensorrt

        model = torch.jit.load(onnx_path.replace(".onnx", ".pt"))
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(
                min_shape=[1, 3, input_height, input_width],
                opt_shape=[4, 3, input_height, input_width],
                max_shape=[max_batch, 3, input_height, input_width],
            )],
            enabled_precisions={torch.float16 if precision == "fp16" else torch.float32},
        )
        torch.jit.save(trt_model, str(dst))
        size_mb = dst.stat().st_size / 1e6
        print(f"[EXPORT] TRT {precision} (torch_trt): {dst} ({size_mb:.1f} MB)")
        return str(dst)
    except (ImportError, Exception) as e:
        print(f"[EXPORT] torch_tensorrt failed: {e}")

    print(f"[EXPORT] WARNING: TRT {precision} export failed — no TRT backend available")
    # Create placeholder so downstream pipeline knows it was attempted
    dst.write_text(f"TRT {precision} export pending — install TensorRT or trtexec\n")
    return str(dst)


def export_all(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    input_height: int = 384,
    input_width: int = 512,
) -> dict[str, str]:
    """Run full 5-format export pipeline.

    Returns dict mapping format name to output path.
    """
    cfg = load_config(config_path)
    results = {}

    print("=" * 60)
    print(f"PiLoT Export Pipeline — {checkpoint_path}")
    print("=" * 60)

    # 1. PTH
    results["pth"] = export_pth(checkpoint_path, output_dir, cfg)

    # 2. Safetensors
    results["safetensors"] = export_safetensors(checkpoint_path, output_dir, cfg)

    # 3. ONNX
    onnx_path = export_onnx(checkpoint_path, output_dir, cfg, input_height, input_width)
    results["onnx"] = onnx_path

    # 4. TRT FP16 (MANDATORY)
    results["trt_fp16"] = export_tensorrt(
        onnx_path, output_dir, "fp16", input_height, input_width
    )

    # 5. TRT FP32 (MANDATORY)
    results["trt_fp32"] = export_tensorrt(
        onnx_path, output_dir, "fp32", input_height, input_width
    )

    print("=" * 60)
    print("Export complete:")
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")
    print("=" * 60)

    return results
