"""Reproduce error, metadata, stage and visual-review outputs after baseline runs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parent

def tables():
    crop=pd.read_csv(ROOT/'results/resnet18/patient_predictions.csv',dtype={'patient_id':str})
    pad=pd.read_csv(ROOT/'results/resnet18_padded/patient_predictions.csv',dtype={'patient_id':str})
    labels=pd.read_csv(ROOT/'data/OCTDL_labels.csv',dtype={'patient_id':str})
    qc=pd.read_csv(ROOT/'results/resnet18/image_qc.csv')
    retained=labels.merge(qc[['file_name']],on='file_name',validate='one_to_one')
    assert len(retained)==len(qc)
    d=crop.merge(pad,on='patient_id',suffixes=('_crop','_pad'),validate='one_to_one')
    assert len(d)==len(crop)==len(pad)
    assert (d.label_crop==d.label_pad).all() and (d.fold_crop==d.fold_pad).all()
    d['crop_correct']=(d.oof_probability_crop>=.5).astype(int)==d.label_crop
    d['pad_correct']=(d.oof_probability_pad>=.5).astype(int)==d.label_pad
    d.to_csv(ROOT/'results/preprocessing_comparison.csv',index=False)
    errors=crop[(crop.oof_probability>=.5).astype(int)!=crop.label].copy()
    errors['predicted_label']=(errors.oof_probability>=.5).astype(int)
    errors['diagnosis']=errors.label.map({0:'Normal',1:'AMD'})
    errors.to_csv(ROOT/'results/resnet18/misclassified_patients.csv',index=False)
    shared=d[~d.crop_correct & ~d.pad_correct].copy()
    shared['diagnosis']=shared.label_crop.map({0:'Normal',1:'AMD'})
    shared[['patient_id','diagnosis','oof_probability_crop','oof_probability_pad']].to_csv(ROOT/'results/shared_errors.csv',index=False)
    def values(x): return '; '.join(sorted(set(x.dropna().astype(str)))) or 'unspecified'
    review=retained.merge(shared[['patient_id','diagnosis']],on='patient_id',validate='many_to_one')
    review.groupby(['patient_id','diagnosis']).agg(retained_scans=('file_name','size'),subcategory=('subcategory',values),condition=('condition',values),eyes=('eye',values)).reset_index().to_csv(ROOT/'results/shared_error_metadata.csv',index=False)
    stages=retained[retained.disease=='AMD'].groupby('patient_id').subcategory.agg(values).rename('stage').reset_index()
    output=[]
    for name,p in [('crop',crop),('padding',pad)]:
        amd=p[p.label==1].merge(stages,on='patient_id',validate='one_to_one')
        assert len(amd)==int((p.label==1).sum())
        amd['detected']=amd.oof_probability>=.5
        t=amd.groupby('stage').agg(patients=('patient_id','size'),detected=('detected','sum')).reset_index()
        t['missed']=t.patients-t.detected; t['sensitivity']=t.detected/t.patients; t['method']=name; output.append(t)
    pd.concat(output,ignore_index=True).to_csv(ROOT/'results/amd_stage_performance.csv',index=False)
    print(f'Reproduced tables: {len(errors)} crop errors, {len(shared)} shared errors.')
    return errors,retained

def figures(errors,retained):
    images={p.stem:p for p in (ROOT/'data/images').rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}}
    if not images: raise FileNotFoundError('Run download.py and analyse.py first; extracted images are required.')
    def first(patient):
        r=retained[retained.patient_id==patient].sort_values('file_name').iloc[0]
        return r,Image.open(images[r.file_name]).convert('RGB')
    nrows=(len(errors)+3)//4
    fig,axes=plt.subplots(nrows,4,figsize=(16,2.6*nrows),squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax,row in zip(axes.flat,errors.itertuples()):
        r,im=first(row.patient_id);ax.imshow(im)
        ax.set_title(f'{row.patient_id} | {row.diagnosis}\nAMD score: {row.oof_probability:.3f}',fontsize=10)
    fig.suptitle('Crop errors: first retained scan per patient (patient-mean predictions)')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(ROOT/'results/resnet18/error_review.png',dpi=160);plt.close(fig)
    import torch
    from torchvision.models import ResNet18_Weights
    transform=ResNet18_Weights.DEFAULT.transforms()
    selected=['3165551','4006332','1781906','5504078','8156034','8689539']
    selected=[p for p in selected if p in set(retained.patient_id)]
    fig,axes=plt.subplots(len(selected),2,figsize=(12,2.5*len(selected)),squeeze=False)
    for i,p in enumerate(selected):
        r,im=first(p);tensor=transform(im)
        mean=torch.tensor(transform.mean).view(3,1,1);std=torch.tensor(transform.std).view(3,1,1)
        display=(tensor*std+mean).clamp(0,1).permute(1,2,0).numpy()
        axes[i,0].imshow(im);axes[i,0].set_title(f'{p} | {r.disease} | original')
        axes[i,1].imshow(display);axes[i,1].set_title('Actual central-crop model input: 224 x 224')
        for ax in axes[i]: ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'results/resnet18/crop_review.png',dpi=150);plt.close(fig)
    print('Reproduced error and crop review figures using retained scans.')
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--tables-only',action='store_true');args=ap.parse_args()
    e,r=tables()
    if not args.tables_only: figures(e,r)
