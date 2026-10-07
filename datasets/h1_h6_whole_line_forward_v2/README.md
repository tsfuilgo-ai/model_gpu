# H1–H6 uncropped whole-line forward data v2

This release contains the final whole-line forward-modeling outputs generated
from the measured H1–H6 velocity slices at `sha-sec` commit
`a2cc3b2b1f869bed598120d924410c3565fc6f28`.

These are the original full-line outputs under `lines/H1` through `lines/H6`.
They have **not** been divided into 5120-trace training windows and have not
undergone the later `[880, 5120]` velocity/label cropping workflow.

Each line contains:

- `forward.npy`: signed postprocessed whole-line waveform, shape
  `[1, 4128, shot_count]`, dtype `float32`;
- `meta.json`: forward and postprocessing metadata, including line-level
  amplitude normalization parameters;
- `plan.json`: forward plan and acquisition parameters;
- `shot_chainage_m.npy`, `shot_xy_m.npy`, `shot_model_columns.npy`, and
  `shot_seafloor_depth_m.npy`: per-shot geometry;
- `_FORWARD_SUCCESS`: source completion marker.

The output sampling rate is 96 kHz. The stored waveform uses the line-level
robust scaling documented in `meta.json`. Multiply it by
`postprocess.normalization.scale` to restore its postprocessed physical-amplitude
scale.

The large `forward.npy` arrays are stored with Git LFS. After cloning, run:

```bash
git lfs pull
python datasets/h1_h6_whole_line_forward_v2/verify_forward_data.py
```

`forward_manifest.json` records each line's shape, byte size, SHA-256 digest,
shot count, and normalization parameters.
