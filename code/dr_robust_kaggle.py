# =============================================================================
#  ROBUSTNESS CHECKS  --  pre-empt the top reviewer attacks (single session)
#   (1) FAIR weighted CP: domain classifier on penultimate EMBEDDINGS (1280-d),
#       not just softmax -> tests whether label-free correction really fails.
#   (2) LABEL/PRIOR SHIFT: grade-distribution divergence source vs targets
#       (explains why a covariate-shift correction is insufficient).
#   (3) BOOTSTRAP 95% CIs for the key coverage numbers (not just std).
#  Reuses the same training; everything in one run.            (Kaggle, GPU on)
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

TASK="ordinal5"; IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4
ALPHA=0.10; C=5; N_SEEDS=3; N_REPS=100; RECAL_M=200
SOURCE=dict(name="APTOS",
    csv="/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    img_root="/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
    id_col="id_code", label_col="diagnosis")
TARGETS=[dict(name="Messidor-2",
    csv="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess",
    id_col="id_code", label_col="diagnosis"),
  dict(name="IDRiD",
    csv="/kaggle/input/datasets/mariaherrerot/idrid-dataset/idrid_labels.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/idrid-dataset/Imagenes/Imagenes",
    id_col="id_code", label_col="diagnosis")]
N_CLASSES=5
def map_label(g):
    try: g=int(g)
    except Exception: return np.nan
    return g

def compute_scores(seed):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision as tv
    from torchvision import transforms
    from PIL import Image
    from sklearn.model_selection import train_test_split
    torch.manual_seed(seed); np.random.seed(seed)
    DEV="cuda" if torch.cuda.is_available() else "cpu"; EXTS=(".png",".jpg",".jpeg",".tif",".tiff")
    def index(root):
        ix={}
        for d,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(EXTS): ix.setdefault(os.path.splitext(f)[0].lower(),os.path.join(d,f))
        return ix
    def prep(cfg):
        ix=index(cfg["img_root"]); df=pd.read_csv(cfg["csv"]); df["_y"]=df[cfg["label_col"]].map(map_label)
        df["_path"]=df[cfg["id_col"]].map(lambda v: ix.get(os.path.splitext(str(v))[0].lower()))
        df=df.dropna(subset=["_y","_path"]); df["_y"]=df["_y"].astype(int); return df.reset_index(drop=True)
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
    m=tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1]=nn.Linear(m.classifier[1].in_features,N_CLASSES); m=m.to(DEV)
    cls=tr["_y"].values
    w=torch.tensor([len(cls)/(N_CLASSES*max(1,(cls==c).sum())) for c in range(N_CLASSES)],dtype=torch.float32,device=DEV)
    crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.Adam(m.parameters(),lr=LR)
    scal=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda")); ld=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=2)
    for ep in range(EPOCHS):
        m.train(); tot=0
        for x,y in ld:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEV=="cuda")): loss=crit(m(x),y)
            scal.scale(loss).backward(); scal.step(opt); scal.update(); tot+=loss.item()
        print(f"   seed{seed} ep{ep+1}/{EPOCHS} loss={tot/len(ld):.3f}",flush=True)
    @torch.no_grad()
    def score(df):
        m.eval(); P,Y,E=[],[],[]
        for x,y in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            x=x.to(DEV); feat=m.features(x); pooled=torch.flatten(m.avgpool(feat),1)  # 1280-d embedding
            logits=m.classifier(pooled); P.append(torch.softmax(logits,1).cpu().numpy())
            E.append(pooled.cpu().numpy()); Y.append(y.numpy())
        return np.concatenate(P),np.concatenate(Y),np.concatenate(E)
    sp,sy,sE=score(hold); src_full_prior=np.bincount(src["_y"].values,minlength=C)/len(src)
    return sp,sy,sE,[(n,*score(df),np.bincount(df["_y"].values,minlength=C)/len(df)) for n,df in tgts],src_full_prior

# ---------------- conformal + robustness cores -------------------------------
def lac_q(s,a):
    n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher") if n>0 else 1.0
def lac_q2(P,y,a): return lac_q(1-P[np.arange(len(y)),y],a)
def lac_sets(P,q): return P>=(1-q)
def cov(S,y): return S[np.arange(len(y)),y].mean()
def wq(s,w,level):
    o=np.argsort(s); s,w=s[o],w[o]; cw=np.cumsum(w)/np.sum(w); return s[min(int(np.searchsorted(cw,level)),len(s)-1)]
def tv(p,q): return float(0.5*np.abs(np.array(p)-np.array(q)).sum())
def boot_ci(vals,B=2000,seed=0):
    r=np.random.default_rng(seed); v=np.array(vals); b=[v[r.integers(0,len(v),len(v))].mean() for _ in range(B)]
    return float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def fit_domain(srcFeat,tgtFeat):
    X=np.vstack([srcFeat,tgtFeat]); yy=np.r_[np.zeros(len(srcFeat)),np.ones(len(tgtFeat))]
    sc=StandardScaler().fit(X); clf=LogisticRegression(max_iter=1000,C=1.0).fit(sc.transform(X),yy)
    p=np.clip(clf.predict_proba(sc.transform(srcFeat))[:,1],1e-3,1-1e-3); return p/(1-p)  # weights for source pts

def main():
    print(f"[A] training {N_SEEDS} seed(s) + extracting scores & embeddings...")
    seeds=[compute_scores(sd) for sd in range(N_SEEDS)]
    names=[t[0] for t in seeds[0][3]]

    # ---- (2) label/prior shift ----
    print("\n===== (2) LABEL / PRIOR SHIFT (grade distribution) =====")
    sp_prior=np.mean([s[4] for s in seeds],0)
    print(" source (APTOS) grade prior:", np.round(sp_prior,3))
    for ti,n in enumerate(names):
        tpri=np.mean([s[3][ti][4] for s in seeds],0)
        print(f" {n:<12} grade prior: {np.round(tpri,3)}   TV(source,{n}) = {tv(sp_prior,tpri):.3f}")

    # ---- (1)+(3) weighted CP fairness with bootstrap CIs ----
    print(f"\n===== (1)+(3) MARGINAL COVERAGE with 95% CIs ({N_SEEDS}x{N_REPS}) target {1-ALPHA:.2f} =====")
    res={n:{k:[] for k in ["naive","wcp_softmax","wcp_embed","local"]} for n in names}
    for sd,(sp,sy,sE,tgts,_) in enumerate(seeds):
        rng=np.random.default_rng(sd)
        for ti,(n,tp,ty,tE,_) in enumerate(tgts):
            # fit domain classifiers ONCE per (seed,target): softmax-feature and embedding-feature
            w_soft_all=fit_domain(sp, tp)         # weights for ALL source-holdout pts (softmax features)
            w_emb_all =fit_domain(sE, tE)         # weights using embeddings
            for _ in range(N_REPS):
                p=rng.permutation(len(sy)); h=len(p)//2; cal=p[:h]
                q=lac_q2(sp[cal],sy[cal],ALPHA)
                tp_perm=rng.permutation(len(ty))
                res[n]["naive"].append(cov(lac_sets(tp[tp_perm],q),ty[tp_perm]))
                qs=wq(1-sp[cal][np.arange(len(cal)),sy[cal]], w_soft_all[cal], 1-ALPHA)
                res[n]["wcp_softmax"].append(cov(lac_sets(tp[tp_perm],qs),ty[tp_perm]))
                qe=wq(1-sp[cal][np.arange(len(cal)),sy[cal]], w_emb_all[cal], 1-ALPHA)
                res[n]["wcp_embed"].append(cov(lac_sets(tp[tp_perm],qe),ty[tp_perm]))
                oc,tt=tp_perm[:RECAL_M],tp_perm[RECAL_M:]
                qr=lac_q2(tp[oc],ty[oc],ALPHA)
                res[n]["local"].append(cov(lac_sets(tp[tt],qr),ty[tt]))
    for n in names:
        print(f"\n [{n}]")
        for k,lab in [("naive","naive transfer"),("wcp_softmax","weighted CP (softmax feats)"),
                      ("wcp_embed","weighted CP (embeddings)"),("local","local recal (m=200)")]:
            mu=np.mean(res[n][k]); lo,hi=boot_ci(res[n][k])
            print(f"   {lab:<30} {mu:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("\n READ: if weighted CP (embeddings) still < target while local recal ~0.90,")
    print("       the 'need local labels' finding is robust to a FAIR baseline.")

if __name__=="__main__":
    main()
