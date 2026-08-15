# V17 Cloud Residual-RGT

V17 trains a pure-PyTorch relational graph residual model on Ascend NPU. The
graph tower excludes raw title, location and vendor values. V11 OOF/test logits
enter only through the residual correction head.

## Cloud flow

1. Build `cloud_dataset/` with `prepare_dataset.py` on the local machine.
2. Upload `rootcause-v17-v1.zip` and register it as a custom dataset.
3. Use `pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64` on
   `huanxin-all-resource` (Ascend 910B).
4. Run `python run_pipeline.py --data-root <mounted dataset> --output outputs`.
5. Download `outputs/` only when `gate_report.json` has `passed: true`.

The online champion is read-only. Submission files are generated under the V17
output directory and always include a diff report and SHA256.
