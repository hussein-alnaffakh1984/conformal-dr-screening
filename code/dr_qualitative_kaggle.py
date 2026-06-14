# =============================================================================
#  QUALITATIVE FIGURE  --  "input (fundus) -> output (prediction set)"
#  Renders REAL target fundus images with their conformal prediction sets under
#  naive transfer vs class-conditional recalibration, auto-selecting cases that
#  tell the story: (a) true grade MISSED by naive but COVERED after class-cond
#  recalibration; (b) a confident correct case; (c) an uncertain case (defer);
#  (d) a severe grade correctly covered.                       (Kaggle, GPU on)
# =============================================================================
import os, sys, subprocess, numpy as np, pandas as pd
def _pip(pkg, imp=None):
    try: __import__(imp or pkg)
    except Exception:
        for ex in ([], ["--break-system-packages"]):
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg]+ex, check=False)
            try: __import__(imp or pkg); return
            except Exception: continue
for p,i in [("torch","torch"),("torchvision","torchvision"),("pillow","PIL"),("matplotlib","matplotlib")]:
    _pip(p,i)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4; ALPHA=0.10; C=5; SEED=0; RECAL_M=200; MIN_PC=10
TARGET_NAME="Messidor-2"; N_EXAMPLES=4; OUTDIR="/kaggle/working"
SOURCE=dict(csv="/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    img_root="/kaggle/input/competitions/aptos2019-blindness-detection/train_images",
    id_col="id_code", label_col="diagnosis")
TARGET=dict(csv="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess",
    id_col="id_code", label_col="diagnosis")

def lac_q(s,a):
    n=len(s); return np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher") if n>0 else 1.0

def run():
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision as tv
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    torch.manual_seed(SEED); np.random.seed(SEED)
    DEV="cuda" if torch.cuda.is_available() else "cpu"; EXTS=(".png",".jpg",".jpeg",".tif",".tiff")
    def index(root):
        ix={}
        for d,_,fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(EXTS): ix.setdefault(os.path.splitext(f)[0].lower(),os.path.join(d,f))
        return ix
    def prep(cfg):
        ix=index(cfg["img_root"]); df=pd.read_csv(cfg["csv"])
        df["_y"]=df[cfg["label_col"]].map(lambda g: int(g) if pd.notna(g) else np.nan)
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
    src=prep(SOURCE); tgt=prep(TARGET)
    tr,hold=train_test_split(src,test_size=0.30,stratify=src["_y"],random_state=SEED)
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
        print(f"   ep{ep+1}/{EPOCHS} loss={tot/len(ld):.3f}",flush=True)
    @torch.no_grad()
    def score(df):
        m.eval(); P=[]
        for x,_ in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            P.append(torch.softmax(m(x.to(DEV)),1).cpu().numpy())
        return np.concatenate(P)
    sP=score(hold); sY=hold["_y"].values.astype(int)
    tP=score(tgt);  tY=tgt["_y"].values.astype(int); paths=tgt["_path"].values

    # thresholds: naive (source) + class-conditional recal (local labeled subset)
    q_naive=lac_q(1-sP[np.arange(len(sY)),sY],ALPHA)
    rng=np.random.default_rng(SEED); perm=rng.permutation(len(tY)); loc,test=perm[:RECAL_M],perm[RECAL_M:]
    qc=np.full(C,lac_q(1-tP[loc][np.arange(len(loc)),tY[loc]],ALPHA))
    for c in range(C):
        s=1-tP[loc][tY[loc]==c][:,c]
        if len(s)>=MIN_PC: qc[c]=lac_q(s,ALPHA)
    naive_set=lambda p:[g for g in range(C) if p[g]>=1-q_naive]
    cc_set   =lambda p:[g for g in range(C) if p[g]>=1-qc[g]]

    # auto-select illustrative examples from the held-out test pool
    chosen=[]; used=set()
    def pick(pred, tag):
        for i in test:
            if i in used: continue
            if pred(i): used.add(i); chosen.append((i,tag)); return
    pick(lambda i: (tY[i] not in naive_set(tP[i])) and (tY[i] in cc_set(tP[i])), "missed by naive → recovered")
    pick(lambda i: len(cc_set(tP[i]))==1 and tY[i] in cc_set(tP[i]),            "confident, correct")
    pick(lambda i: tY[i]>=3 and tY[i] in cc_set(tP[i]),                          "severe grade, covered")
    pick(lambda i: len(cc_set(tP[i]))>=3,                                        "uncertain → defer")
    while len(chosen)<N_EXAMPLES:  # backfill
        for i in test:
            if i not in used: used.add(i); chosen.append((i,"example")); break
    chosen=chosen[:N_EXAMPLES]

    # render
    n=len(chosen); fig,axes=plt.subplots(2,n,figsize=(3.4*n,5.2),gridspec_kw={"height_ratios":[3,2]})
    if n==1: axes=axes.reshape(2,1)
    for j,(i,tag) in enumerate(chosen):
        img=Image.open(paths[i]).convert("RGB").resize((220,220))
        axes[0,j].imshow(img); axes[0,j].axis("off")
        axes[0,j].set_title(f"True grade: {tY[i]}\n[{tag}]",fontsize=10,fontweight="bold")
        ax=axes[1,j]; ax.axis("off")
        def chips(y,label,S):
            ax.text(0.0,y,label,fontsize=9,fontweight="bold",transform=ax.transAxes)
            for k,g in enumerate(range(C)):
                ins=g in S; t=tY[i]
                col="#2e7d32" if (ins and g==t) else ("#c62828" if (g==t and not ins) else ("#90caf9" if ins else "#eeeeee"))
                ax.add_patch(plt.Rectangle((0.34+k*0.13,y-0.035),0.11,0.11,transform=ax.transAxes,fc=col,ec="#555"))
                ax.text(0.395+k*0.13,y+0.02,str(g),fontsize=8,ha="center",transform=ax.transAxes,
                        color="white" if ins or g==t else "#333")
        chips(0.74,"Naive:",naive_set(tP[i])); chips(0.42,"Class-cond:",cc_set(tP[i]))
        ax.text(0.0,0.08,"green = true grade covered   red = true grade missed   blue = other in set",
                fontsize=6.6,color="#555",transform=ax.transAxes)
    fig.suptitle(f"Qualitative examples on {TARGET_NAME}: input fundus image  →  conformal prediction set",
                 fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout(); fp=f"{OUTDIR}/fig_qualitative.png"; plt.savefig(fp,dpi=150,bbox_inches="tight"); plt.close()
    print("naive threshold q =",round(float(q_naive),3)," class-cond q_c =",np.round(qc,3))
    print("figure saved:",fp)

if __name__=="__main__":
    run()
