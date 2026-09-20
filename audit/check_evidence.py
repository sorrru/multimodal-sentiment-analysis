import sys,json,re,time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_experiment as e
import numpy as np,pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
p=str(ROOT)+'/'
a=pd.read_csv(p+'train_data.csv'); b=pd.read_csv(p+'test_data_unlabeled.csv');pr=pd.read_csv(p+'predictions.csv',header=None)
x=a.text.tolist(); y=a.target.to_numpy();clean=[e.clean_text(t) for t in x]
i,j=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=42)
print('shapes/nulls',a.shape,b.shape,a.isna().sum().to_dict(),b.isna().sum().to_dict(),flush=True)
print('class_counts',a.target.value_counts().sort_index().to_dict(),flush=True)
print('predictions',pr.shape,pr.dtypes.to_list(),pr[0].value_counts().sort_index().to_dict(),flush=True)
print('duplicates_raw',a.text.duplicated().sum(),'split_overlap',len(set(a.text.iloc[i])&set(a.text.iloc[j])),'train_test_overlap',len(set(x)&set(b.text)),flush=True)
print('duplicates_clean',pd.Series(clean).duplicated().sum(),'split_overlap',len(set(clean[k] for k in i)&set(clean[k] for k in j)),flush=True)
body=[e.HEADER_RE.sub('',t,count=1).lstrip('\n') if e.HEADER_RE.match(t) else t for t in x]
v=TfidfVectorizer(**e.TFIDF_KW);tr=v.fit_transform([body[k] for k in i]);va=v.transform([body[k] for k in j]);m=LogisticRegression(max_iter=2000,random_state=42).fit(tr,y[i]);print('body_only_acc',accuracy_score(y[j],m.predict(va)),flush=True)
v=TfidfVectorizer(**e.TFIDF_KW);tr=v.fit_transform([clean[k] for k in i]);va=v.transform([clean[k] for k in j])
for model in [LogisticRegression(C=100,max_iter=2000,random_state=42),SVC(C=10,kernel='linear',random_state=42)]:
 t=time.time();model.fit(tr,y[i]);elapsed=time.time()-t;pred=model.predict(va)
 print(type(model).__name__,'fit_seconds',elapsed,'params',getattr(model,'coef_',np.empty(0)).shape,'support_vectors',getattr(model,'n_support_',None),flush=True)
 print(classification_report(y[j],pred,digits=6),flush=True);print(confusion_matrix(y[j],pred),flush=True)
 print('examples',[(int(k),int(y[k]),int(pp),re.search(r'^Subject:.*',x[k],re.M).group(0) if re.search(r'^Subject:.*',x[k],re.M) else '') for k,pp in zip(j,pred) if y[k]!=pp][:25],flush=True)

# Check incomplete header removal independently of quoted metadata in the body.
incomplete=[]
for k,t in enumerate(x):
    header=t.split('\n\n',1)[0]
    match=e.HEADER_RE.match(t)
    if match and match.end()<len(header) and re.search(r'^(From|Organization|NNTP-Posting-Host|Reply-To|Lines):',header[match.end():],re.M):
        incomplete.append(k)
print('incomplete_header_removal_with_metadata',len(incomplete), 'first_indices',incomplete[:10])
for text in set(clean[k] for k in i)&set(clean[k] for k in j):
    print('duplicate_group',[(k,int(y[k]),'train' if k in i else 'val') for k,t in enumerate(clean) if t==text])
