# PRD-06: Export Pipeline

## Objective
Export trained PiLoT model to ONNX, TensorRT, and safetensors formats.

## Export Targets
| Format | Use Case | Tool |
|--------|----------|------|
| safetensors | HuggingFace checkpoint | safetensors library |
| ONNX | Cross-platform inference | torch.onnx.export |
| TensorRT FP16 | Jetson Orin deployment | shared TRT toolkit |
| TensorRT FP32 | Reference inference | shared TRT toolkit |

## Deliverables
- `src/pilot/export.py`:
  - `export_safetensors(model, path)`: save model weights
  - `export_onnx(model, path, input_shape)`: ONNX with dynamic axes
  - `export_trt(onnx_path, output_path, precision)`: via shared toolkit
- `scripts/export.py`: CLI with --checkpoint, --format, --output-dir

## Export Notes
- Feature extractor exports cleanly (standard CNN ops)
- JNGO optimizer has iterative loops -- export feature net only; JNGO stays in PyTorch
- TRT toolkit at /mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py
- Push to HF: `ilessio-aiflowlab/project_pilot-checkpoint`

## Acceptance Criteria
- ONNX model validates with onnx.checker
- TRT engine builds for fp16 and fp32
- safetensors loads back correctly
- Exported models produce same output as PyTorch (within tolerance)
