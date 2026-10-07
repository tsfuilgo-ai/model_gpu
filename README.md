# HSFD_16_0916 velocity inversion

This repository contains the inference code and the requested checkpoint for
the HSFD 16-tile velocity-inversion model.

## Checkpoint

The original checkpoint is larger than GitHub's 100 MB per-file limit. It is
stored losslessly as numbered parts under `checkpoints_16tiles/`.

Reassemble it before inference:

```bash
python3 reconstruct_checkpoint.py
sha256sum -c checkpoints_16tiles/tiles16_batch4_lr0.0005_tikh0.6_best.pt.sha256
```

Then run:

```bash
python3 infer_v2_representative_three.py
```

The inference script expects the aligned-v2 dataset under
`/data/1/lwt/archive/v10_samples_velocity_aligned_v2/samples` and writes its
outputs to `/data/2/lost+found`.
