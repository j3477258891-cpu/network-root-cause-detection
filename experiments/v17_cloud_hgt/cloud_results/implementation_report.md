# V17 Cloud Run Report

Run date: 2026-08-03

## Environment

- Image: `pytorch_v1.1:2.4.0-npu-py310-ubuntu22.04-aarch64`
- Device: one Ascend 910B (`npu:0`)
- Resources: 20 CPU cores, 160 GB memory
- Persistent root: `/root/work/v17_cloud`

## Smoke Test

- PyTorch: 2.4.0
- NPU available: true
- Batch: 21 nodes, 37 edges
- Forward/backward: passed
- Finite loss: 0.3460836112499237
- Note: `aten::logit` used the supported CPU fallback.

## Full Training

- Completed all 15 fold/seed runs.
- Seeds: 20260803, 20260817, 20260831
- Five Group OOF folds per seed
- Checkpoints, fold JSON, OOF/test NPZ files, aggregated scores and logs are stored remotely.

## Gate Result

The strict gate failed. No submission CSV was generated.

- Joint OOF TP delta: -213 (required at least +30)
- Seed TP deltas: -312, -340, -238 (required each at least +24)
- Fold TP deltas: -31, -25, -77, -30, -48 (required each at least -2)
- Bootstrap 95% lower bound: -258.0 (required greater than 0)
- Test changed orders: 94 (allowed 20-80)
- Max template share: 0.0106383 (passed maximum 0.2)

The node ranker showed small fixed-K gains during per-fold early stopping, but the count head and global count allocation destroyed those gains. The next experiment should disable the count head, keep champion K per order, and evaluate rank-only OOF before any further cloud submission candidate is considered.

## Remote Artifacts

- Object storage: `v17-results-failed-gate-v1.tar.gz` (about 28.85 MiB)
- Data object: `rootcause-v17-v1-text.zip`
- Fixed code object: `v17-cloud-hgt-code-v3.zip`
- Code v3 SHA256: `cfbd1f04cb27b182585213e84e67a6f86a4829e9cdf833fa40664ae1d736a836`
- Dataset text package SHA256: `f8ba7977c8c2bfb9807be7fa0d70863197425f3588063e373698df5039e5d568`

The validated champion remains unchanged at SHA256 `6b59ad62b5d1c3d5153a91309f514532228f72a5498a591bf56c86913c703e93`.
