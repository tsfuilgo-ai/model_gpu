# 1007 semantic segmentation: matched direct cascade

Run from the repository root (Python 3.9+, PyTorch, NumPy, Matplotlib):

```bash
python3 reconstruct_checkpoint.py
(cd checkpoints_16tiles && sha256sum -c tiles16_batch4_lr0.0005_tikh0.6_best.pt.sha256)
sha256sum -c segmentation/checkpoints/seg_tiles16_batch8_lr0.0001_best_1007.pt.sha256
python3 segmentation/infer_full_lines.py \
  --data-root /data/1/lwt/archive/v10_samples_velocity_aligned_v2 \
  --output ./outputs/seg1007_full_lines --device cuda:0
```

The segmentation checkpoint is stored as a complete file (87,021,906 bytes),
epoch 100, recorded best validation mIoU 0.6086174135690391.
Use the matched `tiles16_batch4_lr0.0005_tikh0.6_best.pt` velocity checkpoint.

The cascade passes each normalized 880 x 354 velocity output directly to U-Net
before cropping 17 halo traces from either side. It stitches 320-trace cores
by source trace index. Earlier windows own overlap; later windows fill only
uncovered traces. No segmentation smoothing is applied.

Required dataset layout: `samples/H*_line_*/{forward.npy,meta.json,class_label.npy}`
and `full_lines/H*/{meta.json,class_label.npy}`. The repository's whole-line
forward dataset is a different layout and cannot be passed directly to this
entry point. Supply the aligned-v2 dataset separately.

Outputs per line: predicted velocity (m/s), true labels, predicted labels,
maximum-softmax confidence, and a four-panel comparison PNG. NPY arrays retain
full resolution; PNG previews sample at most 6000 columns. Class colors use
integer IDs 0-28 consistently. Target -1 is ignored in metrics. These aligned
samples overlap training/validation; metrics describe fitting, not held-out
generalization. Optional Chinese plot font: WenQuanYi Micro Hei.

`infer_h1_h6_full_lines_0916.py` is the original helper module; its historical
standalone defaults reference older checkpoints. Use `infer_full_lines.py` for
the matched 1007 configuration above.
