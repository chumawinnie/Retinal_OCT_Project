"""OCTDL AMD/normal feasibility study; patient-level, out-of-fold evaluation."""
import argparse, hashlib, json, zipfile, importlib.metadata, platform
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def prepare():
    data=ROOT/'data'; archive=data/'OCTDL.zip'
    expected='dec9766fc916202c45a333c4f5a09576'
    if hashlib.md5(archive.read_bytes()).hexdigest()!=expected:
        raise ValueError('Dataset archive checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        target=(data/'images').resolve()
        for member in z.infolist():
            path=(target/member.filename).resolve()
            if not path.is_relative_to(target): raise ValueError('Unsafe ZIP member')
        z.extractall(target)
    df=pd.read_csv(data/'OCTDL_labels.csv',dtype={'patient_id':str})
    required=['file_name','disease','patient_id']
    if df[required].isna().any().any() or df.file_name.duplicated().any():
        raise ValueError('Missing identifiers or repeated metadata filenames')
    df=df[df.disease.isin(['AMD','NO'])].copy()
    index={}
    for p in target.rglob('*'):
        if p.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff','.bmp']:
            if p.stem in index: raise ValueError('Ambiguous image basename')
            index[p.stem]=p
    df['path']=df.file_name.map(index)
    if df.path.isna().any(): raise ValueError('Images missing from archive')
    if df.groupby('patient_id').disease.nunique().max()>1:
        raise ValueError('Mixed diagnostic labels per patient: review before aggregation')
    df['label']=(df.disease=='AMD').astype(int)
    decoded=[]
    for row in df.itertuples():
        with Image.open(row.path) as im:
            raw=np.asarray(im.convert('L'))
            decoded.append(hashlib.sha256(raw.tobytes()+str(raw.shape).encode()).hexdigest())
    df['decoded_sha256']=decoded
    counts=df.groupby('decoded_sha256').patient_id.nunique()
    shared=set(counts[counts>1].index)
    affected=set(df.loc[df.decoded_sha256.isin(shared),'patient_id'])
    audit=df[df.patient_id.isin(affected)].drop(columns=['path'])
    (ROOT/'results').mkdir(exist_ok=True)
    audit.to_csv(ROOT/'results'/'excluded_duplicate_patients.csv',index=False)
    print(f'Excluded {len(affected)} patients ({len(audit)} scans) due to cross-patient identical images',flush=True)
    df=df[~df.patient_id.isin(affected)].copy()
    # Retain only one copy of an identical scan within a patient.
    df=df.drop_duplicates(['patient_id','decoded_sha256']).reset_index(drop=True)
    return df

def extract(df, encoder):
    features=[]; rows=[]; hashes={}
    if encoder=='resnet18':
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        torch.set_num_threads(4)
        weights=ResNet18_Weights.DEFAULT; model=resnet18(weights=weights)
        model.fc=torch.nn.Identity(); model.eval(); base_transform=weights.transforms()
        from PIL import ImageOps
        from torchvision.transforms import functional as TF
        def transform(image):
            image=ImageOps.pad(
                image, (224,224),
                method=Image.Resampling.BILINEAR,
                color=(0,0,0), centering=(0.5,0.5)
            )
            return TF.normalize(
                TF.to_tensor(image),
                mean=base_transform.mean,
                std=base_transform.std
            )
    for row in df.itertuples():
        with Image.open(row.path) as im:
            gray=im.convert('L'); raw=np.asarray(gray)
            digest=hashlib.sha256(raw.tobytes()+str(raw.shape).encode()).hexdigest()
            if digest in hashes and hashes[digest]!=row.patient_id:
                raise ValueError('Identical decoded image occurs across patients; review duplicates')
            hashes[digest]=row.patient_id
            rows.append({'file_name':row.file_name,'width':im.width,'height':im.height,
                         'mean_intensity':float(raw.mean()),'std_intensity':float(raw.std()),'sha256':digest})
            if encoder=='resnet18':
                with torch.inference_mode():
                    f=model(transform(im.convert('RGB')).unsqueeze(0)).numpy()[0]
            else:
                # Deliberately simple non-deep baseline; resized intensity features.
                f=np.asarray(gray.resize((32,32)),dtype=float).ravel()/255
            features.append(f)
    return np.vstack(features),pd.DataFrame(rows)

def evaluate(X,y):
    if np.bincount(y).min()<5: raise ValueError('Need at least five patients per class')
    pred=np.zeros(len(y)); dummy=np.zeros(len(y))
    folds=np.zeros(len(y),dtype=int)
    for k,(train,test) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42))
        model.fit(X[train],y[train]); pred[test]=model.predict_proba(X[test])[:,1]
        baseline=DummyClassifier(strategy='prior').fit(X[train],y[train]); dummy[test]=baseline.predict_proba(X[test])[:,1]
        folds[test]=k
    tn,fp,fn,tp=confusion_matrix(y,pred>=.5,labels=[0,1]).ravel()
    rng=np.random.default_rng(42); aucs=[]
    for _ in range(1000):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))==2: aucs.append(roc_auc_score(y[idx],pred[idx]))
    return pred,folds,{'auroc':float(roc_auc_score(y,pred)),
        'auroc_bootstrap_95ci':np.quantile(aucs,[.025,.975]).tolist(),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred>=.5)),
        'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),
        'dummy_balanced_accuracy':float(balanced_accuracy_score(y,dummy>=.5)),
        'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]],
        'validation':'five-fold patient-level out-of-fold; fixed hyperparameters',
        'uncertainty_note':'Bootstrap of patient OOF predictions; does not include refitting uncertainty'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--encoder',choices=['pixels','resnet18'],default='pixels'); args=ap.parse_args()
    out=ROOT/'results'/(args.encoder+'_padded'); out.mkdir(parents=True,exist_ok=True)
    df=prepare(); X,qc=extract(df,args.encoder); qc.to_csv(out/'image_qc.csv',index=False)
    np.savez_compressed(out/'image_features.npz',features=X,file_name=df.file_name.to_numpy(dtype=str))
    # Collapse multiple scans per patient before validation: one independent outcome per patient.
    groups=df.patient_id.to_numpy(); ids=np.unique(groups)
    P=np.vstack([X[groups==p].mean(axis=0) for p in ids])
    y=np.array([df.loc[df.patient_id==p,'label'].iloc[0] for p in ids])
    pred,folds,metrics=evaluate(P,y)
    nuisance=qc[['width','height','mean_intensity','std_intensity']].to_numpy()
    nuisance_patient=np.vstack([nuisance[groups==p].mean(axis=0) for p in ids])
    _,_,nuisance_metrics=evaluate(nuisance_patient,y)
    (out/'nuisance_metrics.json').write_text(json.dumps(nuisance_metrics,indent=2))
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','Pillow','scikit-learn','matplotlib']}
    if args.encoder=='resnet18':
        versions.update({name:importlib.metadata.version(name) for name in ['torch','torchvision']})
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions},indent=2))
    metrics.update({'encoder':args.encoder,'n_images':len(df),'n_patients':len(ids),
                    'patients_per_class':{'AMD':int(y.sum()),'NO':int((y==0).sum())}})
    pd.DataFrame({'patient_id':ids,'label':y,'fold':folds,'oof_probability':pred}).to_csv(out/'patient_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # Global PCA is exploratory only; never used by predictive validation.
    coords=PCA(2).fit_transform(StandardScaler().fit_transform(P))
    for label,name in [(0,'Normal'),(1,'AMD')]:
        plt.scatter(coords[y==label,0],coords[y==label,1],label=name,alpha=.65,s=20)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.title('Patient-mean features: exploratory PCA')
    plt.tight_layout(); plt.savefig(out/'pca.png',dpi=180); plt.close()
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
