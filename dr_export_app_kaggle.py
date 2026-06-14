# =============================================================================
#  EXPORT FOR WEB APP  --  dumps a single dr_app_data.json that fully represents
#  our real results: cached softmax scores + true labels (3 seeds) for source
#  holdout and both targets, PLUS a few REAL Messidor-2 example images (base64)
#  with their real scores. The HTML app runs the exact same conformal maths on
#  this data, so it reproduces the paper's numbers (not a mock).   (Kaggle, GPU)
# =============================================================================
import os, sys, subprocess, io, json, base64, numpy as np, pandas as pd
def _pip(pkg, imp=None):
    try: __import__(imp or pkg)
    except Exception:
        for ex in ([], ["--break-system-packages"]):
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg]+ex, check=False)
            try: __import__(imp or pkg); return
            except Exception: continue
for p,i in [("torch","torch"),("torchvision","torchvision"),("pillow","PIL")]:
    _pip(p,i)
from PIL import Image

IMG_SIZE=224; EPOCHS=8; BATCH=32; LR=3e-4; C=5; N_SEEDS=3; RECAL_M=200; MIN_PC=10
N_EXAMPLES=6; THUMB=200; OUT="/kaggle/working/dr_app_data.json"
SOURCE=dict(name="APTOS",csv="/kaggle/input/competitions/aptos2019-blindness-detection/train.csv",
    img_root="/kaggle/input/competitions/aptos2019-blindness-detection/train_images",id_col="id_code",label_col="diagnosis")
TARGETS=[dict(name="Messidor-2",csv="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor_data.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/messidor2preprocess/messidor-2/messidor-2/preprocess",id_col="id_code",label_col="diagnosis"),
  dict(name="IDRiD",csv="/kaggle/input/datasets/mariaherrerot/idrid-dataset/idrid_labels.csv",
    img_root="/kaggle/input/datasets/mariaherrerot/idrid-dataset/Imagenes/Imagenes",id_col="id_code",label_col="diagnosis")]

def lac_q(s,a):
    n=len(s); return float(np.quantile(s,min(np.ceil((n+1)*(1-a))/n,1.0),method="higher")) if n>0 else 1.0

def compute(seed, want_paths=False):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision as tv
    from torchvision import transforms
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
        m.eval(); P=[]
        for x,_ in DataLoader(DS(df),batch_size=BATCH,num_workers=2):
            P.append(torch.softmax(m(x.to(DEV)),1).cpu().numpy())
        return np.concatenate(P)
    sP=score(hold); sY=hold["_y"].values.astype(int)
    out={"source":{"P":np.round(sP,4).tolist(),"y":sY.tolist()},"targets":{}}
    paths={}
    for n,df in tgts:
        out["targets"][n]={"P":np.round(score(df),4).tolist(),"y":df["_y"].values.astype(int).tolist()}
        if want_paths and n=="Messidor-2":
            paths={"P":np.array(out["targets"][n]["P"]),"y":np.array(out["targets"][n]["y"]),"paths":df["_path"].values}
    return out, paths

def pick_examples(P,Y,paths,sP,sY):
    rng=np.random.default_rng(0); perm=rng.permutation(len(Y)); loc,test=perm[:RECAL_M],perm[RECAL_M:]
    qn=lac_q(1-sP[np.arange(len(sY)),sY],0.10)        # naive threshold from SOURCE holdout (faithful)
    qc=np.full(C,lac_q(1-P[loc][np.arange(len(loc)),Y[loc]],0.10))
    for c in range(C):
        s=1-P[loc][Y[loc]==c][:,c]
        if len(s)>=MIN_PC: qc[c]=lac_q(s,0.10)
    nset=lambda p:[g for g in range(C) if p[g]>=1-qn]
    cset=lambda p:[g for g in range(C) if p[g]>=1-qc[g]]
    chosen=[]; used=set()
    def pk(pred,tag):
        for i in test:
            if i in used: continue
            if pred(i): used.add(i); chosen.append((int(i),tag)); return
    pk(lambda i:(Y[i] not in nset(P[i])) and (Y[i] in cset(P[i])),"missed by naive \u2192 recovered")
    pk(lambda i:len(cset(P[i]))==1 and Y[i] in cset(P[i]),"confident, correct")
    pk(lambda i:Y[i]>=3 and Y[i] in cset(P[i]),"severe grade, covered")
    pk(lambda i:len(cset(P[i]))>=3,"uncertain \u2192 defer")
    pk(lambda i:Y[i]>=2 and len(cset(P[i]))<=2,"referable, handled")
    pk(lambda i:Y[i]<=1,"non-referable")
    while len(chosen)<N_EXAMPLES:
        for i in test:
            if i not in used: used.add(i); chosen.append((int(i),"example")); break
    ex=[]
    for i,tag in chosen[:N_EXAMPLES]:
        im=Image.open(paths["paths"][i]).convert("RGB"); im.thumbnail((THUMB,THUMB))
        buf=io.BytesIO(); im.save(buf,format="JPEG",quality=72)
        ex.append({"img":"data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode(),
                   "P":P[i].tolist(),"y":int(Y[i]),"tag":tag})
    return ex,{"q_naive":qn,"q_cc":qc.tolist()}

def main():
    seeds=[]; ex=None; meta=None
    for sd in range(N_SEEDS):
        print(f"[export] seed {sd} ...")
        out,paths=compute(sd, want_paths=(sd==0))
        seeds.append(out)
        if sd==0:
            ex,meta=pick_examples(paths["P"],paths["y"],paths,
                                  np.array(out["source"]["P"]),np.array(out["source"]["y"]))
    data={"info":{"alpha":0.10,"classes":C,"n_seeds":N_SEEDS,"recal_m":RECAL_M,
                  "source":"APTOS","targets":["Messidor-2","IDRiD"]},
          "seeds":seeds,"examples":ex,"example_thresholds":meta}
    with open(OUT,"w") as f: json.dump(data,f,separators=(",",":"))
    mb=os.path.getsize(OUT)/1e6
    print(f"\n saved {OUT}  ({mb:.2f} MB)  | seeds={N_SEEDS} examples={len(ex)}")
    print(" -> download this single file and send it back to build the web app.")

if __name__=="__main__":
    main()
