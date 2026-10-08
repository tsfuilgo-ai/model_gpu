#!/usr/bin/env python3
"""Infer and stitch H1-H6 semantic labels from aligned 5120-trace windows."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from model_16 import UNet
from Tomo_model import TomoNet


DATA = Path('/data/1/lwt/archive/v10_samples_velocity_aligned_v1')
OUTPUT = Path('/data/2/lost+found/HSFD_0916_seg_h1_h6_full_lines')
VELOCITY_CHECKPOINT = Path('/data/2/lwt/HSFD_16_0916/checkpoints_16tiles/lr0.0005_best.pt')
SEGMENTATION_CHECKPOINT = Path('/data/2/lwt/HSFD_seg_ImagingResults_as_input/checkpoints/seg_tiles16_batch8_lr0.0001_best.pt')
CORE, HALO, NT, WINDOW = 320, 17, 3000, 5120
FMIN, FMAX = -1.21907763e-6, 1.21907763e-6
WMIN, WMAX = -2.0, 37.0


def load_model(model, checkpoint_file, device):
    checkpoint = torch.load(str(checkpoint_file), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def tile(forward, start, stop):
    lo, hi = max(0, start), min(WINDOW, stop)
    array = np.asarray(forward[:, :NT, lo:hi], dtype=np.float32)
    left, right = max(0, -start), max(0, stop-WINDOW)
    if left or right:
        array = np.pad(array, ((0,0),(0,0),(left,right)), mode='edge')
    assert array.shape == (1, NT, CORE + 2*HALO)
    return array


@torch.no_grad()
def infer_window(velocity_model, segmentation_model, sample_dir, device):
    forward = np.load(sample_dir/'forward.npy', mmap_mode='r')
    meta = json.loads((sample_dir/'meta.json').read_text(encoding='utf-8'))
    scale = np.float32(meta['normalization']['scale'])
    water = np.clip((meta['seafloor_depth_m']['mean'] - 3.0 - WMIN)/(WMAX-WMIN),0,1)
    water = torch.tensor([water],dtype=torch.float32,device=device)
    prediction = np.empty((880, WINDOW), dtype=np.int16)
    for i in range(16):
        start = i*CORE-HALO
        data = tile(forward,start,start+CORE+2*HALO)*scale
        data = np.clip((data-FMIN)/(FMAX-FMIN),0,1).astype(np.float32,copy=False)
        data = torch.from_numpy(np.ascontiguousarray(data.transpose(0,2,1))[None]).to(device)
        velocity = velocity_model(data,water,crop_output=False)
        logits = segmentation_model(velocity)
        labels = logits[...,HALO:-HALO].argmax(1)[0].to('cpu').numpy()
        prediction[:,i*CORE:(i+1)*CORE] = labels
    return prediction, meta


def render_preview(labels, line, spacing_m):
    plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','axes.unicode_minus':False})
    colors = list(plt.cm.tab20.colors)+list(plt.cm.Set3.colors[:9])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5,29.5,1),cmap.N)
    fig, ax = plt.subplots(figsize=(22,5.5),dpi=160,constrained_layout=True)
    ax.imshow(labels,cmap=cmap,norm=norm,interpolation='nearest',aspect='auto',
              extent=[0,(labels.shape[1]-1)*spacing_m,19,-3])
    ax.set(title=f'{line} 全线语义分割预测｜0916 反演速度输入',
           xlabel='沿测线里程 (m)',ylabel='海床相对深度 (m)')
    output = OUTPUT/f'{line}_full_line_preview.png'
    fig.savefig(output)
    plt.close(fig)
    return output


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(16)
    velocity_model, vckpt = load_model(TomoNet(),VELOCITY_CHECKPOINT,device)
    segmentation_model, sckpt = load_model(UNet(num_classes=29),SEGMENTATION_CHECKPOINT,device)
    summary = {
        'model_pipeline':'forward.npy -> original HSFD_16_0916 TomoNet -> imaging-input UNet',
        'device':str(device),
        'velocity_checkpoint':str(VELOCITY_CHECKPOINT),'velocity_epoch':int(vckpt['epoch']),
        'segmentation_checkpoint':str(SEGMENTATION_CHECKPOINT),'segmentation_epoch':int(sckpt['epoch']),
        'stitch_rule':'Window predictions are placed by source_trace_start; later overlapping windows only fill previously uncovered traces.',
        'label_matrix_axes':'[seafloor_relative_depth, source_trace_index]',
        'labels':'Integer class IDs 0-28; semantic names in the source semantic_label_legend_zh.json.',
        'lines':{}
    }
    print('Device:',device,flush=True)
    for line in ('H1','H2','H3','H4','H5','H6'):
        line_meta=json.loads((DATA/'full_lines'/line/'meta.json').read_text(encoding='utf-8'))
        count=int(line_meta['shot_count'])
        spacing=float(line_meta['shot_interval_m'])
        full=np.full((880,count),-1,dtype=np.int16)
        covered=np.zeros(count,dtype=bool)
        records=[]
        windows=sorted((DATA/'samples').glob(line+'_line_*'))
        for sample in windows:
            labels,meta=infer_window(velocity_model,segmentation_model,sample,device)
            start,stop=int(meta['source_trace_start']),int(meta['source_trace_stop_exclusive'])
            assert stop-start == WINDOW and stop<=count
            missing=~covered[start:stop]
            full[:,start:stop][:,missing]=labels[:,missing]
            covered[start:stop]=True
            records.append({'sample':sample.name,'start':start,'stop_exclusive':stop,
                            'new_traces':int(missing.sum()),'overlap_traces':int((~missing).sum())})
            print(line,sample.name,'new',int(missing.sum()),'overlap',int((~missing).sum()),flush=True)
        assert covered.all(), f'{line} has {int((~covered).sum())} uncovered traces'
        assert ((full>=0)&(full<29)).all()
        matrix=OUTPUT/f'{line}_full_line_predicted_class_label.npy'
        np.save(matrix,full)
        preview=render_preview(full,line,spacing)
        values,counts=np.unique(full,return_counts=True)
        summary['lines'][line]={
            'shape':list(full.shape),'dtype':str(full.dtype),
            'shot_spacing_m':spacing,'chainage_start_m':0.0,
            'chainage_end_m':(count-1)*spacing,
            'covered_traces':int(covered.sum()),'window_count':len(windows),
            'class_histogram':{str(int(v)):int(n) for v,n in zip(values,counts)},
            'windows':records,'matrix':str(matrix),'preview':str(preview)
        }
        (OUTPUT/'inference_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',line,matrix,full.shape,flush=True)
    print('COMPLETE',OUTPUT,flush=True)


if __name__=='__main__':
    main()
