"""Validate saved predictions and regenerate comparison tables without refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
root=Path(__file__).resolve().parent
rows=[]
for folder in ['pixels','resnet18','resnet18_padded']:
    d=pd.read_csv(root/'results'/folder/'patient_predictions.csv',dtype={'patient_id':str})
    m=json.loads((root/'results'/folder/'metrics.json').read_text())
    assert not d.patient_id.duplicated().any()
    assert set(d.fold)==set(range(5))
    assert len(d)==m['n_patients']
    assert np.isclose(roc_auc_score(d.label,d.oof_probability),m['auroc'])
    assert np.isclose(balanced_accuracy_score(d.label,d.oof_probability>=.5),m['balanced_accuracy'])
    assert confusion_matrix(d.label,d.oof_probability>=.5).tolist()==m['confusion_matrix']
    rows.append({'method':folder,**{k:m[k] for k in ['auroc','balanced_accuracy','sensitivity','specificity']}})
crop=pd.read_csv(root/'results/resnet18/patient_predictions.csv',dtype={'patient_id':str})
pad=pd.read_csv(root/'results/resnet18_padded/patient_predictions.csv',dtype={'patient_id':str})
d=crop.merge(pad,on='patient_id',suffixes=('_crop','_pad'),validate='one_to_one')
assert len(d)==len(crop)==len(pad)
assert (d.label_crop==d.label_pad).all() and (d.fold_crop==d.fold_pad).all()
d['crop_correct']=(d.oof_probability_crop>=.5).astype(int)==d.label_crop
d['pad_correct']=(d.oof_probability_pad>=.5).astype(int)==d.label_pad
d.to_csv(root/'results/preprocessing_comparison.csv',index=False)
pd.DataFrame(rows).to_csv(root/'results/model_comparison.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.crosstab(d.crop_correct,d.pad_correct))
print('Saved metrics agree with all patient predictions.')
