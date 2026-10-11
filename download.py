from pathlib import Path
from urllib.request import urlretrieve
import hashlib
ROOT=Path(__file__).resolve().parent/'data'
ROOT.mkdir(exist_ok=True)
for name,expected in [('OCTDL_labels.csv','2fced789038c6ca83283066c2f8ad289'),('OCTDL.zip','dec9766fc916202c45a333c4f5a09576')]:
    path=ROOT/name
    if not path.exists():
        print('Downloading',name,flush=True)
        urlretrieve('https://zenodo.org/records/10839556/files/'+name,path)
    if hashlib.md5(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Checksum mismatch: '+name)
    print('Verified',name)
