# =============================================================================
#  ALL-IN-ONE  --  Cross-Population Conformal DR Screening (single session)
#  Runs end to end in ONE go (no cross-session files needed):
#    [A] train N_SEEDS models on APTOS, score source-holdout + targets
#    [B] RIGOR: marginal coverage {in-dist, naive, weightedCP, local-recal}
#               + Mondrian per-class coverage
#    [C] METHOD: class-conditional recalibration vs marginal — worst-grade
#                coverage vs labeling budget (the contribution)
#  Note: Kaggle /kaggle/working is wiped when the kernel stops, so we keep all
#        scores in memory and do everything in this single run.   (GPU on)
# =============================================================================
import os, sys, subprocess, numpy as np, pandas as pd
def _pip(pkg, imp=None):
    try: __import__(imp or pkg)
    except Exception:
        for ex in ([], ["--break-system-packages"]):
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg]+ex, check=False)
            try: __import__(imp or pkg); return
            except Exception: continue
for p,i in [("torch","torch"),("torchvision","torchvision"),("pillow","PIL"),
            ("scikit-learn","sklearn"),("matplotlib","matplotlib")]:
    _pip(p,i)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ----------------------------- CONFIG ---------------------------------------
TASK="ordinal5"; IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4
ALPHA=0.10; C=5; MIN_PC=10
N_SEEDS=3; N_REPS=100; RECAL_M=200
BUDGETS=[50,75,100,150,200,300]
OUTDIR="/kaggle/working"
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
N_CLASSES=2 if TASK=="binary" else 5
def map_label(g):
    try: g=int(g)
    except Exception: return np.nan
    return (1 if g>=2 else 0) if TASK=="binary" else g

# ----------------------------- [A] SCORES -----------------------------------
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
    sc=torch.cuda.amp.GradScaler(enabled=(DEV=="cuda")); ld=DataLoader(DS(tr,True),batch_size=BATCH,shuffle=True,num_workers=2)
    for ep in range(EPOCHS):
        m.train(); tot=0
        for x,y in ld:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEV=="cuda")): loss=crit(m(x),y)
            sc.scale(loss).backward(); sc.step(opt); sc.update(); tot+=loss.item()
        print(f"   seed{seed} ep{ep+1}/{EPOCHS} loss={tot/len(ld):.3f}",flush=True)
    @torch.no_grad()
    def score(df):
        m.eval(); out,ys=[],[]
        for x,y in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            out.append(torch.softmax(m(x.to(DEV)),1).cpu().numpy()); ys.append(y.numpy())
        return np.concatenate(out),np.concatenate(ys)
    sp,sy=score(hold); return sp,sy,[(n,*score(df)) for n,df in tgts]

# ----------------------------- conformal ------------------------------------
def lac_q(s,a):
    n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher") if n>0 else 1.0
def lac_q2(P,y,a): return lac_q(1-P[np.arange(len(y)),y],a)
def lac_sets(P,q): return P>=(1-q)
def cov(S,y): return S[np.arange(len(y)),y].mean()
def per_class(P,y,qc): return np.array([ (P[y==c][:,c]>=(1-qc[c])).mean() if (y==c).any() else np.nan for c in range(C)])
def marg_cov(P,y,qc): S=P>=(1-qc[None,:]); return S[np.arange(len(y)),y].mean()
def wq(s,w,level):
    o=np.argsort(s); s,w=s[o],w[o]; cw=np.cumsum(w)/np.sum(w); return s[min(int(np.searchsorted(cw,level)),len(s)-1)]
def domain_weights(srcP,tgtP):
    X=np.vstack([srcP,tgtP]); yy=np.r_[np.zeros(len(srcP)),np.ones(len(tgtP))]
    clf=LogisticRegression(max_iter=500).fit(X,yy); p=np.clip(clf.predict_proba(srcP)[:,1],1e-4,1-1e-4); return p/(1-p)
def thresholds(calP,calY,strategy,qm):
    if strategy=="marginal": return np.full(C,qm)
    qc=np.full(C,qm)
    for c in range(C):
        s=1-calP[calY==c][:,c]
        if len(s)>=MIN_PC: qc[c]=lac_q(s,ALPHA)
    return qc
def select(idx,tY,predY,m,strategy):
    if strategy in ("marginal","mondrian_random"): return idx[:m]
    per=max(1,m//C); key=tY if strategy=="mondrian_balanced" else predY; cal=[]
    for c in range(C): cal+=list(idx[key[idx]==c][:per])
    return np.array(cal) if len(cal) else idx[:m]
STRATS=["marginal","mondrian_random","mondrian_targeted","mondrian_balanced"]

# ----------------------------- [B] rigor ------------------------------------
def rigor(sP,sY,tP,tY,seed):
    rng=np.random.default_rng(seed); reg={k:[] for k in ["indist","naive","weighted","recal"]}; pc={k:[] for k in ["naive","recal"]}
    for _ in range(N_REPS):
        p=rng.permutation(len(sY)); h=len(p)//2; cal,te=p[:h],p[h:]
        q=lac_q2(sP[cal],sY[cal],ALPHA); reg["indist"].append(cov(lac_sets(sP[te],q),sY[te]))
        tp=rng.permutation(len(tY))
        Sn=lac_sets(tP[tp],q); reg["naive"].append(cov(Sn,tY[tp])); pc["naive"].append(per_class(tP[tp],tY[tp],np.full(C,q)))
        oc,tt=tp[:RECAL_M],tp[RECAL_M:]; qr=lac_q2(tP[oc],tY[oc],ALPHA)
        reg["recal"].append(cov(lac_sets(tP[tt],qr),tY[tt])); pc["recal"].append(per_class(tP[tt],tY[tt],np.full(C,qr)))
        w=domain_weights(sP[cal],tP[tp]); qw=wq(1-sP[cal][np.arange(len(cal)),sY[cal]],w,1-ALPHA)
        reg["weighted"].append(cov(lac_sets(tP[tp],qw),tY[tp]))
    return reg,pc

# ----------------------------- [C] method -----------------------------------
def classcond(tP,tY,seed):
    rng=np.random.default_rng(seed); predY=tP.argmax(1)
    out={s:{m:[] for m in BUDGETS} for s in STRATS}
    for _ in range(N_REPS):
        idx=rng.permutation(len(tY))
        for s in STRATS:
            for m in BUDGETS:
                cal=select(idx,tY,predY,m,s); test=np.setdiff1d(idx,cal)
                if len(test)<50: continue
                qm=lac_q2(tP[cal],tY[cal],ALPHA); qc=thresholds(tP[cal],tY[cal],s,qm)
                out[s][m].append(np.nanmin(per_class(tP[test],tY[test],qc)))
    return out

# ----------------------------- RUN ------------------------------------------
def main():
    allscores=[]
    for sd in range(N_SEEDS):
        print(f"[A] seed {sd}: training + scoring...")
        allscores.append(compute_scores(sd))
    names=[n for n,_,_ in allscores[0][2]]
    f=lambda x:(float(np.nanmean(x)),float(np.nanstd(x))) if len(x) else (np.nan,np.nan)

    # [B] rigor
    R={n:{k:[] for k in ["indist","naive","weighted","recal"]} for n in names}
    PC={n:{k:[] for k in ["naive","recal"]} for n in names}
    for sd,(sP,sY,tgts) in enumerate(allscores):
        for n,tP,tY in tgts:
            reg,pc=rigor(sP,sY,tP,tY,sd)
            for k in reg: R[n][k]+=reg[k]
            for k in pc: PC[n][k]+=pc[k]
    print(f"\n===== [B] MARGINAL COVERAGE ({N_SEEDS} seeds x {N_REPS} splits, target {1-ALPHA:.2f}) =====")
    for n in names:
        print(f" [{n}] in-dist {f(R[n]['indist'])[0]:.3f} | naive {f(R[n]['naive'])[0]:.3f} | "
              f"weightedCP {f(R[n]['weighted'])[0]:.3f} | local-recal {f(R[n]['recal'])[0]:.3f}")
    print("\n===== Mondrian per-class coverage (naive vs marginal-recal) =====")
    for n in names:
        print(f" [{n}] grade:        " + "  ".join(str(c) for c in range(C)))
        for k in ["naive","recal"]:
            mu=np.nanmean(np.array(PC[n][k]),0); print(f"   {k:<8} " + "  ".join(f"{v:.2f}" for v in mu))

    # [C] method
    CC={n:{s:{m:[] for m in BUDGETS} for s in STRATS} for n in names}
    for sd,(sP,sY,tgts) in enumerate(allscores):
        for n,tP,tY in tgts:
            r=classcond(tP,tY,sd)
            for s in STRATS:
                for m in BUDGETS: CC[n][s][m]+=r[s][m]
    print(f"\n===== [C] WORST-GRADE coverage vs labels (target {1-ALPHA:.2f}) =====")
    for n in names:
        print(f"\n [{n}]  m:    " + "   ".join(f"{m:>4}" for m in BUDGETS))
        for s in STRATS:
            print(f"   {s:<18} " + "   ".join(f"{f(CC[n][s][m])[0]:.2f}" for m in BUDGETS))

    # figures
    fig,ax=plt.subplots(1,len(names),figsize=(6.5*len(names),4.2),squeeze=False)
    st={"marginal":("C3","o"),"mondrian_random":("C0","s"),"mondrian_targeted":("C1","^"),"mondrian_balanced":("C2","D")}
    for a,n in zip(ax[0],names):
        for s in STRATS:
            a.plot(BUDGETS,[f(CC[n][s][m])[0] for m in BUDGETS],marker=st[s][1],color=st[s][0],label=s.replace("mondrian_","M-"))
        a.axhline(1-ALPHA,ls="--",c="k",lw=1); a.set_title(n); a.set_xlabel("# local labels"); a.set_ylabel("worst-grade coverage"); a.set_ylim(0.5,1.0); a.legend(fontsize=8)
    plt.tight_layout(); fp=f"{OUTDIR}/fig_classcond.png"; plt.savefig(fp,dpi=140); plt.close()
    print(f"\n figure saved: {fp}")
    print(" DONE: [B] rigor + [C] class-conditional method, single run.")

if __name__=="__main__":
    main()
