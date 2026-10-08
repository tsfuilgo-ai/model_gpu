import sys, io, json, hashlib, shutil, argparse
from pathlib import Path
from datetime import datetime
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
import infer_h1_h6_full_lines_0916 as base
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

parser=argparse.ArgumentParser(description='Matched Tikh0.6 -> seg1007 full-line cascade with uncut halo.')
parser.add_argument('--data-root', type=Path, required=True, help='Aligned-v2 root containing samples/ and full_lines/.')
parser.add_argument('--output', type=Path, default=Path('outputs')/datetime.now().strftime('%Y%m%d_%H%M%S'))
parser.add_argument('--velocity-checkpoint', type=Path, default=PROJECT.parent/'checkpoints_16tiles/tiles16_batch4_lr0.0005_tikh0.6_best.pt')
parser.add_argument('--segmentation-checkpoint', type=Path, default=PROJECT/'checkpoints/seg_tiles16_batch8_lr0.0001_best_1007.pt')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--lines', nargs='+', choices=['H1','H2','H3','H4','H5','H6'], default=['H1','H2','H3','H4','H5','H6'])
args=parser.parse_args()
ROOT=args.data_root
OUT=args.output
OUT.mkdir(parents=True)
torch.set_num_threads(8)
device=torch.device(args.device)
summary={'pipeline':'forward -> Tikh0.6 TomoNet(crop_output=False) -> seg1007 UNet -> crop 17 on each side', 'dataset':str(ROOT),'stitch_rule':'Place by source_trace_start; earlier windows own overlap, later windows fill uncovered traces only. All four matrices use identical ownership.', 'lines':{}}
def load(cls,path,key):
    p=Path(path); payload=p.read_bytes(); ck=torch.load(io.BytesIO(payload),map_location='cpu')
    m=cls(); m.load_state_dict(ck['model_state_dict'],strict=True); m.to(device).eval()
    summary[key]={'path':str(p),'sha256':hashlib.sha256(payload).hexdigest(),'epoch':int(ck['epoch'])}
    return m
vm=load(base.TomoNet,args.velocity_checkpoint,'velocity_checkpoint')
sm=load(base.UNet,args.segmentation_checkpoint,'segmentation_checkpoint')
plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','axes.unicode_minus':False})
colors=list(plt.cm.tab20.colors)+list(plt.cm.Set3.colors[:9])
cmap=ListedColormap(colors); cmap.set_bad('#eeeeee'); norm=BoundaryNorm(np.arange(-.5,29.5),29)
print('OUTPUT',OUT,flush=True)
for line in args.lines:
    lm=json.loads((ROOT/'full_lines'/line/'meta.json').read_text()); n=lm['shot_count']; shape=(880,n)
    windows=[]; coverage=np.zeros(n,bool)
    for sample in (ROOT/'samples').glob(line+'_line_*'):
        meta=json.loads((sample/'meta.json').read_text()); windows.append((meta['source_trace_start'],sample,meta))
    windows.sort(key=lambda x:x[0])
    for start,_,meta in windows:
        stop=meta['source_trace_stop_exclusive']; assert stop-start==5120 and 0<=start<stop<=n
        coverage[start:stop]=True
    assert coverage.all(), (line,'missing traces',int((~coverage).sum()))
    velocity=np.lib.format.open_memmap(str(OUT/(line+'_velocity_m_s.npy')),mode='w+',dtype='float32',shape=shape)
    labels=np.lib.format.open_memmap(str(OUT/(line+'_predicted_class_label.npy')),mode='w+',dtype='int16',shape=shape)
    confidence=np.lib.format.open_memmap(str(OUT/(line+'_confidence.npy')),mode='w+',dtype='float32',shape=shape)
    truth=np.load(ROOT/'full_lines'/line/'class_label.npy',mmap_mode='r')
    assert truth.shape==shape
    coverage[:]=False; records=[]
    for start,sample,meta in windows:
        f=np.load(sample/'forward.npy',mmap_mode='r')
        water=torch.tensor([np.clip((meta['seafloor_depth_m']['mean']-3-base.WMIN)/(base.WMAX-base.WMIN),0,1)],dtype=torch.float32,device=device)
        local_truth=np.load(sample/'class_label.npy',mmap_mode='r')
        fresh=~coverage[start:start+5120]
        with torch.no_grad():
            for i in range(16):
                lo=i*320; mask=fresh[lo:lo+320]
                if not mask.any(): continue
                x=base.tile(f,lo-17,lo+337)*np.float32(meta['normalization']['scale'])
                x=np.clip((x-base.FMIN)/(base.FMAX-base.FMIN),0,1).astype(np.float32)
                v=vm(torch.from_numpy(np.ascontiguousarray(x.transpose(0,2,1))[None]).to(device),water,crop_output=False)
                assert tuple(v.shape)==(1,1,880,354)
                logits=sm(v)[...,17:-17]; prob,pred=logits.softmax(1).max(1)
                sl=slice(start+lo,start+lo+320)
                assert np.array_equal(truth[:,sl][:,mask],local_truth[:,lo:lo+320][:,mask])
                velocity[:,sl][:,mask]=(v[0,0,:,17:-17].cpu().numpy()*1800+1000)[:,mask]
                labels[:,sl][:,mask]=pred[0].cpu().numpy()[:,mask]
                confidence[:,sl][:,mask]=prob[0].cpu().numpy()[:,mask]
        coverage[start:start+5120]=True
        records.append({'sample':sample.name,'start':start,'new_traces':int(fresh.sum())})
        print(line,sample.name,'covered',int(coverage.sum()),'/',n,flush=True)
    assert coverage.all()
    for a in (velocity,labels,confidence): a.flush()
    cm=np.zeros((29,29),np.int64)
    for k in range(0,n,1024):
        t=np.asarray(truth[:,k:k+1024]); p=np.asarray(labels[:,k:k+1024]); valid=t>=0
        assert np.all((p>=0)&(p<29)) and np.isfinite(velocity[:,k:k+1024]).all()
        assert np.all((confidence[:,k:k+1024]>=0)&(confidence[:,k:k+1024]<=1))
        cm+=np.bincount(t[valid].astype(int)*29+p[valid],minlength=841).reshape(29,29)
    union=cm.sum(0)+cm.sum(1)-cm.diagonal(); ids=np.flatnonzero(union)
    acc=float(cm.trace()/cm.sum()); miou=float(np.mean(cm.diagonal()[ids]/union[ids]))
    shutil.copyfile(str(ROOT/'full_lines'/line/'class_label.npy'),str(OUT/(line+'_true_class_label.npy')))
    # Full matrices are retained; raster previews sample at most 6000 columns.
    cols=np.linspace(0,n-1,min(n,6000)).astype(int)
    vshow=velocity[:,cols]; lo=float(np.floor(np.percentile(vshow,.5)/5)*5); hi=float(np.ceil(np.percentile(vshow,99.5)/5)*5)
    extent=[0,(n-1)*lm['shot_interval_m'],19,-3]
    fig,axes=plt.subplots(4,1,figsize=(25,13),dpi=200,constrained_layout=True)
    im=axes[0].imshow(vshow,cmap='jet',vmin=lo,vmax=hi,extent=extent,aspect='auto',interpolation='nearest'); fig.colorbar(im,ax=axes[0],label='m/s',pad=.008,fraction=.015)
    axes[0].set_title(f'速度反演｜聚焦色标 {lo:.0f}–{hi:.0f} m/s')
    for ax,arr,title in ((axes[1],truth,'真实语义标签'),(axes[2],labels,'语义分割预测')):
        ax.imshow(np.ma.masked_less(arr[:,cols],0),cmap=cmap,norm=norm,extent=extent,aspect='auto',interpolation='nearest'); ax.set_title(title)
    axes[2].legend(handles=[Patch(color=colors[int(i)],label='类别 '+str(i)) for i in ids],ncol=min(15,len(ids)),loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=8)
    im=axes[3].imshow(confidence[:,cols],cmap='viridis',vmin=0,vmax=1,extent=extent,aspect='auto',interpolation='nearest'); fig.colorbar(im,ax=axes[3],label='最大 softmax 概率',pad=.008,fraction=.015)
    axes[3].set_title('预测置信度（模型概率，不等同于正确率）')
    for ax in axes: ax.set_ylabel('海床相对深度 (m)')
    axes[-1].set_xlabel('沿测线里程 (m)')
    fig.suptitle(f'{line} 整线｜Tikhonov 0.6 → seg1007｜准确率 {acc:.2%}｜mIoU {miou:.4f}\n概览横向抽样显示；完整分辨率见 NPY 矩阵',fontsize=16)
    fig.savefig(OUT/(line+'_full_line_comparison.png')); plt.close(fig)
    summary['lines'][line]={'shape':list(shape),'trace_spacing_m':lm['shot_interval_m'],'depth_spacing_m':lm['depth_spacing_m'],'chainage_end_m':extent[1],'covered_traces':int(coverage.sum()),'accuracy':acc,'miou':miou,'class_iou':{str(i):float(cm[i,i]/union[i]) for i in ids},'velocity_color_limits':[lo,hi],'preview_columns':len(cols),'windows':records}
    (OUT/'inference_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('SAVED',line,'accuracy',acc,'miou',miou,flush=True)
    del velocity,labels,confidence,truth
shutil.copyfile(__file__,str(OUT/'infer_six_direct_full.py'))
print('COMPLETE',OUT,flush=True)
