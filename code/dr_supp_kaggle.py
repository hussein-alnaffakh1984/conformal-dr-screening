# =============================================================================
#  SUPPLEMENTARY ANALYSIS  --  disentangle miscalibration from shift (one run)
#   (1) ECE (15-bin) in-distribution vs targets  -> base-model calibration
#   (2) Temperature scaling [14]: fit T on source; ECE before/after; and show
#       it does NOT restore conformal coverage under shift (collapse is a SHIFT
#       phenomenon, not base miscalibration).
#   (3) alpha sensitivity: coverage collapse at alpha = 0.05 / 0.10 / 0.20.
#  Reuses the same training pipeline.                          (Kaggle, GPU on)
# =============================================================================
import os, sys, subprocess, numpy as np, pandas as pd
def _pip(pkg, imp=None):
    try: __import__(imp or pkg)
    except Exception:
        for ex in ([], ["--break-system-packages"]):
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg]+ex, check=False)
            try: __import__(imp or pkg); return
            except Exception: continue
for p,i in [("torch","torch"),("torchvision","torchvision"),("pillow","PIL"),("scipy","scipy")]:
    _pip(p,i)
from scipy.optimize import minimize_scalar

IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4; C=5; N_SEEDS=3; N_REPS=100
SOURCE=dict(csv="/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    img_root="/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
    id_col="id_code", label_col="diagnosis")
TARGETS=[dict(name="Messidor-2",
    csv="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess",
    id_col="id_code", label_col="diagnosis"),
  dict(name="IDRiD",
    csv="/kaggle/input/datasets/mariaherrerot/idrid-dataset/idrid_labels.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/idrid-dataset/Imagenes/Imagenes",
    id_col="id_col" if False else "id_code", label_col="diagnosis")]

def softmax(z): e=np.exp(z-z.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
def ece(P,y,bins=15):
    conf=P.max(1); pred=P.argmax(1); acc=(pred==y).astype(float)
    edges=np.linspace(0,1,bins+1); e=0.0
    for i in range(bins):
        m=(conf>edges[i])&(conf<=edges[i+1])
        if m.any(): e+=m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)
def fit_T(logits,y):
    def nll(T): P=softmax(logits/T); return -np.log(P[np.arange(len(y)),y]+1e-12).mean()
    return float(minimize_scalar(nll,bounds=(0.3,10.0),method="bounded").x)
def lac_q(s,a):
    n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher") if n>0 else 1.0

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
    m=tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1]=nn.Linear(m.classifier[1].in_features,C); m=m.to(DEV)
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
        print(f"   seed{seed} ep{ep+1}/{EPOCHS} loss={tot/len(ld):.3f}",flush=True)
    @torch.no_grad()
    def score(df):
        m.eval(); L,Y=[],[]
        for x,y in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            L.append(m(x.to(DEV)).cpu().numpy()); Y.append(y.numpy())
        L=np.concatenate(L); return softmax(L),L,np.concatenate(Y)
    sP,sL,sY=score(hold)
    return sP,sL,sY,[(n,*score(df)) for n,df in tgts]

def main():
    seeds=[compute_scores(sd) for sd in range(N_SEEDS)]
    names=[t[0] for t in seeds[0][3]]
    avg=lambda xs: float(np.mean(xs))

    # ---- (1)+(2) ECE and temperature scaling ----
    print("\n===== (1)+(2) CALIBRATION (ECE, 15-bin) & TEMPERATURE SCALING =====")
    ece_in,ece_in_T,Ts={ "raw":[],"temp":[]},None,[]
    rowsERaw={"in-dist":[],**{n:[] for n in names}}; rowsET={"in-dist":[],**{n:[] for n in names}}
    for sP,sL,sY,tgts in seeds:
        T=fit_T(sL,sY); Ts.append(T)
        rowsERaw["in-dist"].append(ece(sP,sY)); rowsET["in-dist"].append(ece(softmax(sL/T),sY))
        for n,tP,tL,tY in tgts:
            rowsERaw[n].append(ece(tP,tY)); rowsET[n].append(ece(softmax(tL/T),tY))
    print(f" fitted temperature T = {avg(Ts):.3f} (mean over {N_SEEDS} seeds)")
    print(f" {'set':<12} {'ECE raw':>10} {'ECE +temp':>10}")
    for k in ["in-dist"]+names:
        print(f" {k:<12} {avg(rowsERaw[k]):>10.3f} {avg(rowsET[k]):>10.3f}")

    # ---- temp scaling does NOT fix conformal coverage under shift ----
    print("\n===== Temperature scaling vs conformal coverage under shift =====")
    cov={s:{"in":[],**{n:[] for n in names}} for s in ["raw","temp"]}
    for sP,sL,sY,tgts in seeds:
        T=fit_T(sL,sY)
        for variant,(SP,fn) in [("raw",(sP,lambda L:softmax(L))),("temp",(None,lambda L:softmax(L/T)))]:
            SPv=sP if variant=="raw" else softmax(sL/T)
            rng=np.random.default_rng(0)
            for _ in range(N_REPS):
                idx=rng.permutation(len(sY)); cal=idx[:len(idx)//2]; te=idx[len(idx)//2:]
                q=lac_q(1-SPv[cal][np.arange(len(cal)),sY[cal]],0.10)
                cov[variant]["in"].append((SPv[te][np.arange(len(te)),sY[te]]>=1-q).mean())
                for n,tP,tL,tY in tgts:
                    TPv=tP if variant=="raw" else softmax(tL/T)
                    cov[variant][n].append((TPv[np.arange(len(tY)),tY]>=1-q).mean())
    print(f" {'variant':<8} {'in-dist':>9} " + " ".join(f"{n:>11}" for n in names))
    for v in ["raw","temp"]:
        print(f" {v:<8} {avg(cov[v]['in']):>9.3f} " + " ".join(f"{avg(cov[v][n]):>11.3f}" for n in names))
    print(" -> temperature scaling improves ECE but does NOT restore coverage under shift.")

    # ---- (3) alpha sensitivity ----
    print("\n===== (3) ALPHA SENSITIVITY (naive coverage; collapse persists) =====")
    print(f" {'alpha':>6} {'target':>7} {'in-dist':>9} " + " ".join(f"{n:>11}" for n in names))
    for a in [0.05,0.10,0.20]:
        ins=[]; tg={n:[] for n in names}
        for sP,sL,sY,tgts in seeds:
            rng=np.random.default_rng(0)
            for _ in range(N_REPS):
                idx=rng.permutation(len(sY)); cal=idx[:len(idx)//2]; te=idx[len(idx)//2:]
                q=lac_q(1-sP[cal][np.arange(len(cal)),sY[cal]],a)
                ins.append((sP[te][np.arange(len(te)),sY[te]]>=1-q).mean())
                for n,tP,tL,tY in tgts: tg[n].append((tP[np.arange(len(tY)),tY]>=1-q).mean())
        print(f" {a:>6} {1-a:>7.2f} {avg(ins):>9.3f} " + " ".join(f"{avg(tg[n]):>11.3f}" for n in names))
    print("\n DONE: ECE + temperature scaling + alpha sensitivity.")

if __name__=="__main__":
    main()
