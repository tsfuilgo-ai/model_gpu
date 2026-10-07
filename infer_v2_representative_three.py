#!/usr/bin/env python3
"""Infer three representative aligned-v2 samples from one checkpoint snapshot."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from custom_dataset_16 import SeismicTileDataset
from model_16 import TomoNet

PROJECT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT / 'checkpoints_16tiles/tiles16_batch4_lr0.0005_tikh0.6_best.pt'
DATA_ROOT = Path('/data/1/lwt/archive')
SAMPLES = ('H1_line_000', 'H2_line_005', 'H6_line_005')


def metrics(prediction, target):
    error = prediction-target
    return {
        'rmse_m_s': float(np.sqrt(np.mean(error**2))),
        'mae_m_s': float(np.mean(np.abs(error))),
        'bias_m_s': float(np.mean(error)),
        'absolute_error_p95_m_s': float(np.percentile(np.abs(error),95)),
    }


def main():
    torch.set_num_threads(16)
    payload = CHECKPOINT.read_bytes()
    checkpoint = torch.load(io.BytesIO(payload),map_location='cpu')
    model = TomoNet()
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    dataset = SeismicTileDataset(
        'val', dataset_root=DATA_ROOT,
        aligned_data_dir='v10_samples_velocity_aligned_v2',
    )
    output = Path('/data/2/lost+found') / (
        'HSFD_16_0916_v2_inference_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    output.mkdir(parents=True)
    summary = {
        'checkpoint':str(CHECKPOINT),
        'checkpoint_sha256':hashlib.sha256(payload).hexdigest(),
        'checkpoint_epoch':int(checkpoint['epoch']),
        'checkpoint_best_val_loss':float(checkpoint['best_val_loss']),
        'checkpoint_args':checkpoint['args'],
        'dataset':'/data/1/lwt/archive/v10_samples_velocity_aligned_v2/samples',
        'samples':{},
    }
    del payload,checkpoint
    plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','axes.unicode_minus':False})
    print('OUTPUT',output,flush=True)
    for name in SAMPLES:
        sample_index = next(i for i,s in enumerate(dataset.samples) if s.path.name==name)
        pieces=[]
        with torch.no_grad():
            for tile_index in range(16):
                item=dataset[sample_index*16+tile_index]
                prediction=model(item['data'][None],item['water_depth'][None])
                pieces.append(prediction[0,0].numpy())
        prediction=np.concatenate(pieces,axis=1)*(
            dataset.VELOCITY_MAX-dataset.VELOCITY_MIN
        )+dataset.VELOCITY_MIN
        sample=dataset.samples[sample_index].path
        target=np.asarray(np.load(sample/'vp.npy'),dtype=np.float32)
        assert prediction.shape==target.shape==(880,5120)
        assert np.isfinite(prediction).all()
        result=metrics(prediction,target)
        np.save(output/(name+'_predicted_velocity_m_s.npy'),prediction.astype(np.float32))
        meta=json.loads((sample/'meta.json').read_text(encoding='utf-8'))
        extent=[meta['chainage_start_m'],meta['chainage_end_m'],
                meta['label_depth_bottom_relative_m'],meta['label_depth_top_relative_m']]
        error=prediction-target
        limit=max(10,float(np.ceil(np.percentile(np.abs(error),99)/5)*5))
        fig,axes=plt.subplots(3,1,figsize=(19,9),dpi=170,constrained_layout=True)
        panels=((target,'真实速度','jet',1500,2000),
                (prediction,'当前最佳检查点反演速度','jet',1500,2000),
                (error,'有符号误差：预测 − 真实','coolwarm',-limit,limit))
        for ax,(array,title,cmap,vmin,vmax) in zip(axes,panels):
            image=ax.imshow(array,aspect='auto',extent=extent,interpolation='nearest',
                            cmap=cmap,vmin=vmin,vmax=vmax)
            fig.colorbar(image,ax=ax,label='m/s',pad=0.008,fraction=0.018)
            ax.set_title(title)
            ax.set_ylabel('海床相对深度 (m)')
        axes[-1].set_xlabel('沿测线里程 (m)')
        fig.suptitle(
            f'{name}｜aligned_v2｜Epoch {summary["checkpoint_epoch"]}｜'
            f'RMSE={result["rmse_m_s"]:.2f} m/s｜MAE={result["mae_m_s"]:.2f} m/s\n'
            '该样本参与训练/验证，仅反映拟合效果',fontsize=16
        )
        image_file=output/(name+'_velocity_inversion.png')
        fig.savefig(image_file)
        plt.close(fig)
        summary['samples'][name]={**result,
            'image':str(image_file),
            'prediction_array':str(output/(name+'_predicted_velocity_m_s.npy')),
            'chainage_start_m':meta['chainage_start_m'],
            'chainage_end_m':meta['chainage_end_m'],
            'class_histogram':meta.get('class_histogram',{}),
        }
        (output/'inference_summary.json').write_text(
            json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'
        )
        print(name,result,flush=True)
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    main()
