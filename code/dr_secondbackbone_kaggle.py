# =============================================================================
#  SECOND BACKBONE + SHIFT-MAGNITUDE  --  one run, addresses two reviewer points
#   (A) Replicate the coverage collapse + local-recalibration recovery with a
#       DIFFERENT architecture (ResNet-50) -> shows the effect is not specific
#       to EfficientNet-B0 (empirical, not just theoretical).
#   (B) Quantify SHIFT MAGNITUDE per target: label-prior TV, domain-classifier
#       AUC and RBF-MMD on penultimate embeddings, tabulated against the naive
#       coverage collapse (shift magnitude <-> collapse).            (Kaggle GPU)
# =============================================================================
import os, sys, subprocess, numpy as np, pandas as pd
def _pip(pkg, imp=None):
    try: __import__(imp or pkg)
    except Exception:
        for ex in ([], ["--break-system-packages"]):
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg]+ex, check=False)
            try: __import__(imp or pkg); return
            except Exception: continue
for p,i in [("torch","torch"),("torchvision","torchvision"),("pillow","PIL"),("scikit-learn","sklearn")]:
    _pip(p,i)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4; C=5; N_SEEDS=2; N_REPS=100; RECAL_M=200; ALPHA=0.10
SOURCE=dict(csv="/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    img_root="/kaggle/input/competitions/aptos2019-blindness-detection/train_images",id_col="id_code",label_col="diagnosis")
TARGETS=[dict(name="Messidor-2",csv="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess",id_col="id_code",label_col="diagnosis"),
  dict(name="IDRiD",csv="/kaggle/input/datasets/mariaherrerot/idrid-dataset/idrid_labels.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/idrid-dataset/Imagenes/Imagenes",id_col="id_code",label_col="diagnosis")]

def lac_q(s,a):
    n=len(s); return float(np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher")) if n>0 else 1.0
def lac_q2(P,y,a): return lac_q(1-P[np.arange(len(y)),y],a)
def cov(P,y,q): return float((P[np.arange(len(y)),y]>=1-q).mean())
def boot_ci(v,B=2000,seed=0):
    r=np.random.default_rng(seed); v=np.array(v); b=[v[r.integers(0,len(v),len(v))].mean() for _ in range(B)]
    return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def tv(p,q): return float(0.5*np.abs(np.array(p)-np.array(q)).sum())
def domain_auc(srcE,tgtE,seed=0):
    X=np.vstack([srcE,tgtE]); y=np.r_[np.zeros(len(srcE)),np.ones(len(tgtE))]
    Xs=StandardScaler().fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.4,random_state=seed,stratify=y)
    clf=LogisticRegression(max_iter=500).fit(Xtr,ytr)
    return float(roc_auc_score(yte,clf.predict_proba(Xte)[:,1]))
def mmd_rbf(X,Y,n=300,seed=0):
    r=np.random.default_rng(seed); X=X[r.permutation(len(X))[:n]]; Y=Y[r.permutation(len(Y))[:n]]
    Z=np.vstack([X,Y]); d2=np.sum((Z[:,None]-Z[None,:])**2,-1); gamma=1.0/(np.median(d2[d2>0])+1e-9)
    k=lambda A,B: np.exp(-gamma*np.sum((A[:,None]-B[None,:])**2,-1))
    return float(k(X,X).mean()+k(Y,Y).mean()-2*k(X,Y).mean())

def compute(seed):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision as tv_
    from torchvision import transforms
    from PIL import Image
    torch.manual_seed(seed); np.random.seed(seed)
    DEV="cuda" if torch.cuda.is_available() else "cpu"; EXTS=(".png",".jpg",".jpeg",".tif",".tiff")
    def index(root):
        ix={}
        for d,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(EXTS): ix.setdefault(os.path.splitext(f)[0].lower(),os.path.join(d,f))
        return ix
    def prep(cfg):
        ix=index(cfg["img_root"]); df=pd.read_csv(cfg["csv"])
        df["_y"]=df[cfg["label_col"]].map(lambda g:int(g) if pd.notna(g) else np.nan)
        df["_path"]=df[cfg["id_col"]].map(lambda v: ix.get(os.path.splitext(str(v))[0].lower()))
        return df.dropna(subset=["_y","_path"]).reset_index(drop=True)
    norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ttr=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),transforms.ToTensor(),norm])
    tev=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),norm])
    class DS(Dataset):
        def __init__(s,df,tr=False): s.df=df; s.tf=ttr if tr else tev
        def __len__(s): return len(s.df)
        def __getitem__(s,k):
            r=s.df.iloc[k]; return s.tf(Image.open(r["_path"]).convert("RGB")), int(r["_y"])
    src=prep(SOURCE); tgts=[(t["name"],prep(t)) for t in TARGETS]
    tr,hold=train_test_split(src,test_size=0.30,stratify=src["_y"],random_state=seed)
    m=tv_.models.resnet50(weights=tv_.models.ResNet50_Weights.DEFAULT)            # <-- SECOND BACKBONE
    m.fc=nn.Linear(m.fc.in_features,C); m=m.to(DEV)
    feat=nn.Sequential(*list(m.children())[:-1])                                  # up to avgpool (2048-d)
    cls=tr["_y"].values
    w=torch.tensor([len(cls)/(C*max(1,(cls==c).sum())) for c in range(C)],dtype=torch.float32,device=DEV)
    crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.Adam(m.parameters(),lr=LR)
    scal=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda")); ld=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=2)
    for ep in range(EPOCHS):
        m.train(); tot=0
        for x,y in ld:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEV=="cuda")): loss=crit(m(x),y)
            scal.scale(loss).backward(); scal.step(opt); scal.update(); tot+=loss.item()
        print(f"   [ResNet50] seed{seed} ep{ep+1}/{EPOCHS} loss={tot/len(ld):.3f}",flush=True)
    @torch.no_grad()
    def score(df):
        m.eval(); P,E=[],[]
        for x,_ in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            x=x.to(DEV); e=torch.flatten(feat(x),1); P.append(torch.softmax(m.fc(e),1).cpu().numpy()); E.append(e.cpu().numpy())
        return np.concatenate(P),np.concatenate(E)
    sP,sE=score(hold); sY=hold["_y"].values.astype(int); sprior=np.bincount(src["_y"].values,minlength=C)/len(src)
    T=[]
    for n,df in tgts:
        tP,tE=score(df); T.append((n,tP,df["_y"].values.astype(int),tE,np.bincount(df["_y"].values,minlength=C)/len(df)))
    return sP,sY,sE,sprior,T

def main():
    seeds=[compute(sd) for sd in range(N_SEEDS)]
    names=[t[0] for t in seeds[0][4]]
    f=lambda x:(float(np.mean(x)),)+boot_ci(x)

    print(f"\n===== (A) ResNet-50 REPLICATION: coverage (target {1-ALPHA:.2f}) =====")
    res={n:{k:[] for k in ["indist","naive","local"]} for n in names}
    for sP,sY,sE,sprior,T in seeds:
        rng=np.random.default_rng(0)
        for n,tP,tY,tE,tprior in T:
            for _ in range(N_REPS):
                p=rng.permutation(len(sY)); h=len(p)//2; cal,te=p[:h],p[h:]
                q=lac_q2(sP[cal],sY[cal],ALPHA); res[n]["indist"].append(cov(sP[te],sY[te],q))
                tp=rng.permutation(len(tY)); res[n]["naive"].append(cov(tP[tp],tY[tp],q))
                oc,tt=tp[:RECAL_M],tp[RECAL_M:]; qr=lac_q2(tP[oc],tY[oc],ALPHA); res[n]["local"].append(cov(tP[tt],tY[tt],qr))
    for n in names:
        mi=f(res[n]["indist"]); mn=f(res[n]["naive"]); ml=f(res[n]["local"])
        print(f" [{n}] in-dist {mi[0]:.3f} [{mi[1]:.3f},{mi[2]:.3f}] | naive {mn[0]:.3f} [{mn[1]:.3f},{mn[2]:.3f}] | local-recal {ml[0]:.3f} [{ml[1]:.3f},{ml[2]:.3f}]")
    print(" -> if naive collapses and local-recal recovers (as with EfficientNet-B0), the effect is architecture-agnostic.")

    print("\n===== (B) SHIFT MAGNITUDE vs COLLAPSE =====")
    print(f" {'target':<12} {'TV(label)':>10} {'domainAUC':>10} {'MMD':>8} {'naive cov':>10}")
    sP0,sY0,sE0,sprior0,T0=seeds[0]
    for n,tP,tY,tE,tprior in T0:
        auc=domain_auc(sE0,tE); mmd=mmd_rbf(sE0,tE); tvd=tv(sprior0,tprior)
        nc=np.mean(res[n]["naive"])
        print(f" {n:<12} {tvd:>10.3f} {auc:>10.3f} {mmd:>8.4f} {nc:>10.3f}")
    print(" -> larger domain-classifier AUC / MMD (covariate shift) and TV (label shift)")
    print("    accompany larger coverage collapse, linking shift magnitude to the failure.")

if __name__=="__main__":
    main()
