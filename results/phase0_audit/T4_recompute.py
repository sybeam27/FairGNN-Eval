"""T4 (b) recomputation. CPU-only; reads frozen CSVs, writes nothing frozen."""
# ---- part 1: four protocol cells (frozen) and the X25 rerun ----
import os, sys, numpy as np, pandas as pd
ROOT="/home/sypark/workspace/FairGNN-Eval"
sys.path.insert(0, os.path.join(ROOT,"harness","experiments"))
from analyze_armA import sign_stability
from bootstrap_armA import boot

SEED=20260916; B=10000
SIGN_MIN, NEAR_ZERO = 0.75, 0.010
SEL="common_bce"; COL="int_ndp"

def load(path):
    d=pd.read_csv(path)
    d=d[(d.method=="NIFTY")&(d.dataset=="german")&(d.selector==SEL)]
    return d.set_index(["split_id","run_id"])[COL]

F={
 "P00": f"{ROOT}/harness/results/armA_german.csv",
 "P10": f"{ROOT}/harness/results/x24_nifty_german_P10.csv",
 "P01": f"{ROOT}/harness/results/x24_nifty_german_P01.csv",
 "P11": f"{ROOT}/harness/results/armB_native_NIFTY_german.csv",
}
R={
 "P00": f"{ROOT}/harness/results/armA_german.csv",          # no rerun at H=200
 "P10": f"{ROOT}/harness/results/x25/x25_R10.csv",
 "P01": f"{ROOT}/harness/results/x24_nifty_german_P01.csv",  # no rerun at H=200
 "P11": f"{ROOT}/harness/results/x25/x25_R11.csv",
}

def analyze(files, tag, seed=SEED):
    v={k:load(p) for k,p in files.items()}
    t=pd.DataFrame(v)
    assert len(t)==30, len(t)
    t["th00"]=t.P00; t["th10"]=t.P10; t["th01"]=t.P01; t["th11"]=t.P11
    t["dH"]=0.5*((t.P10-t.P00)+(t.P11-t.P01))
    t["dD"]=0.5*((t.P01-t.P00)+(t.P11-t.P10))
    t["Gamma"]=t.P11-t.P10-t.P01+t.P00
    t["dH_D1"]=t.P11-t.P01
    t["dH_D0"]=t.P10-t.P00
    t["dD_H200"]=t.P01-t.P00
    t["dD_H1000"]=t.P11-t.P10
    t["dtotal"]=t.P11-t.P00
    t=t.reset_index()
    cols=["th00","th01","th10","th11","dH","dD","Gamma","dH_D1","dH_D0","dD_H200","dD_H1000","dtotal"]
    rng=np.random.default_rng(seed)
    reps=boot(t, cols, rng, reps=B)
    rows=[]
    for i,c in enumerate(cols):
        x=t[c].to_numpy(float); mu=x.mean()
        lo,hi=np.percentile(reps[:,i],[2.5,97.5])
        s=sign_stability(pd.Series(x)); s=float(s) if s!="n/a" else float("nan")
        res=bool(np.isfinite(s) and s>=SIGN_MIN and abs(mu)>=NEAR_ZERO and lo*hi>0)
        rows.append(dict(term=c,mean=mu,lo=lo,hi=hi,sign=s,resolved=res,run_set=tag))
    return pd.DataFrame(rows)

a=analyze(F,"frozen")
b=analyze(R,"rerun_x25_H1000_with_frozen_H200")
out=pd.concat([a,b])
pd.set_option("display.width",200)
print(out.to_string(index=False,float_format=lambda x:f"{x:+.4f}"))
out.to_csv("/tmp/claude-1007/-home-sypark-workspace/4a7ed2e4-d126-46e8-9baa-fefb92ca0e27/scratchpad/t4_terms_raw.csv",index=False)

# ---- part 2: X25 replayed trajectory values ----
import os,sys,numpy as np,pandas as pd
ROOT="/home/sypark/workspace/FairGNN-Eval"
sys.path.insert(0,os.path.join(ROOT,"harness","experiments"))
from analyze_armA import sign_stability
from bootstrap_armA import boot
ct=pd.read_csv(f"{ROOT}/harness/results/x25/x25_cell_table.csv")
t=pd.DataFrame(index=ct.index)
t["split_id"]=ct.split_id; t["run_id"]=ct.run_id
t["th00"]=ct["D0.bce.h200.ndp"]; t["th01"]=ct["D1.bce.h200.ndp"]
t["th10"]=ct["D0.bce.c1000.ndp"]; t["th11"]=ct["D1.bce.c1000.ndp"]
t["dH"]=0.5*((t.th10-t.th00)+(t.th11-t.th01))
t["dD"]=0.5*((t.th01-t.th00)+(t.th11-t.th10))
t["Gamma"]=t.th11-t.th10-t.th01+t.th00
t["dH_D1"]=t.th11-t.th01; t["dH_D0"]=t.th10-t.th00
t["dD_H200"]=t.th01-t.th00; t["dD_H1000"]=t.th11-t.th10
t["dtotal"]=t.th11-t.th00
cols=["th00","th01","th10","th11","dH","dD","Gamma","dH_D1","dH_D0","dD_H200","dD_H1000","dtotal"]
rng=np.random.default_rng(20260916)
reps=boot(t,cols,rng,reps=10000)
rows=[]
for i,c in enumerate(cols):
    x=t[c].to_numpy(float);mu=x.mean();lo,hi=np.percentile(reps[:,i],[2.5,97.5])
    s=sign_stability(pd.Series(x)); s=float(s) if s!="n/a" else float("nan")
    rows.append(dict(term=c,mean=mu,lo=lo,hi=hi,sign=s,resolved=bool(s>=0.75 and abs(mu)>=0.010 and lo*hi>0),run_set="rerun_x25_replay"))
df=pd.DataFrame(rows); print(df.to_string(index=False,float_format=lambda x:f"{x:+.4f}"))
df.to_csv("/tmp/claude-1007/-home-sypark-workspace/4a7ed2e4-d126-46e8-9baa-fefb92ca0e27/scratchpad/t4_terms_replay.csv",index=False)
