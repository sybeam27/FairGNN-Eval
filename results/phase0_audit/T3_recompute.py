import numpy as np, pandas as pd, os
ROOT="/home/sypark/workspace/FairGNN-Eval"
OUT=f"{ROOT}/results/phase0_audit"
SIGN_MIN, NEAR_ZERO = 0.75, 0.010
SEED = 20260914
B = 10_000

t1 = pd.read_csv(f"{ROOT}/results/1_main_package_vs_intervention.csv")
d  = pd.read_csv(f"{ROOT}/results/per_unit_metrics.csv.gz")
d  = d[(d.protocol=="controlled") & (d.selector=="common_bce")]

def store_name(m,c): return m if c=="default" else f"{m}-{c}"

w = d.pivot_table(index=["method","dataset","split_id","run_id"],
                  columns="state", values=["auc","dp","eo"])
u = pd.DataFrame(index=w.index)
u["tau_auc"] =  w[("auc","M_plus_I")] - w[("auc","M_minus_I")]
u["tau_ndp"] = -(w[("dp","M_plus_I")] - w[("dp","M_minus_I")])
u["tau_neo"] = -(w[("eo","M_plus_I")] - w[("eo","M_minus_I")])
u = u.reset_index()

COORDS=[("dAUC","tau_auc"),("negDP","tau_ndp"),("negEO","tau_neo")]

def ss_majority(x):
    x=np.asarray(x,float); nz=x[x!=0.0]
    return 0.0 if nz.size==0 else max((nz>0).mean(),(nz<0).mean())

def ss_pointest(x):
    x=np.asarray(x,float); nz=x[x!=0.0]
    if nz.size==0: return 0.0
    s=np.sign(x.mean())
    if s==0: return 0.0
    return float((np.sign(nz)==s).mean())

def boot_reps(cells, cols, rng, reps=B):
    splits=cells.split_id.unique()
    by={s:cells[cells.split_id==s][cols].to_numpy() for s in splits}
    ns=len(splits); out=np.empty((reps,len(cols)))
    for b in range(reps):
        drawn=rng.integers(0,ns,ns); acc=[]
        for j in drawn:
            m=by[splits[j]]; acc.append(m[rng.integers(0,len(m),len(m))])
        out[b]=np.concatenate(acc,axis=0).mean(axis=0)
    return out

cols=[c for _,c in COORDS]
rows=[]
rng=np.random.default_rng(SEED)
for _,r in t1.iterrows():
    sn=store_name(r.method,r.configuration)
    g=u[(u.method==sn)&(u.dataset==r.dataset)]
    assert len(g)==30, (sn,r.dataset,len(g))
    reps=boot_reps(g.sort_values(["split_id","run_id"]), cols, rng)
    rec=dict(method=r.method,configuration=r.configuration,dataset=r.dataset,store_name=sn,n_units=len(g))
    for cn,ck in COORDS:
        x=g[ck].to_numpy(); mu=float(x.mean())
        k=cols.index(ck); lo,hi=np.percentile(reps[:,k],[2.5,97.5])
        smaj=ss_majority(x); spt=ss_pointest(x)
        sbt=float((np.sign(reps[:,k])==np.sign(mu)).mean()) if mu!=0 else 0.0
        rec[f"{cn}_mean"]=mu; rec[f"{cn}_lo"]=float(lo); rec[f"{cn}_hi"]=float(hi)
        rec[f"{cn}_frozen_sign"]=float(r[f"tau_I_{cn}_sign_stability"])
        rec[f"{cn}_frozen_resolved"]=bool(r[f"tau_I_{cn}_resolved"])
        rec[f"{cn}_sign_majority"]=smaj
        rec[f"{cn}_sign_pointest"]=spt
        rec[f"{cn}_sign_boot"]=sbt
        rec[f"{cn}_interval_excl0"]=bool(lo*hi>0)
        rec[f"{cn}_mag_ok"]=bool(abs(mu)>=NEAR_ZERO)
        for tag,s in (("majority",smaj),("pointest",spt),("boot",sbt)):
            rec[f"{cn}_resolved_{tag}"]=bool(s>=SIGN_MIN and abs(mu)>=NEAR_ZERO and lo*hi>0)
    rows.append(rec)
R=pd.DataFrame(rows)
R.to_csv(f"{OUT}/T3_sign_stability_recompute.csv",index=False)

print("SEED",SEED,"B",B)
for cn,_ in COORDS:
    print(f"\n=== {cn} ===")
    print("  frozen resolved:", int(R[f'{cn}_frozen_resolved'].sum()))
    for tag in ("majority","pointest","boot"):
        print(f"  {tag:9s} resolved:", int(R[f'{cn}_resolved_{tag}'].sum()))
    print("  max |frozen_sign - majority|:", (R[f'{cn}_frozen_sign']-R[f'{cn}_sign_majority']).abs().max())
    print("  max |frozen_sign - pointest|:", (R[f'{cn}_frozen_sign']-R[f'{cn}_sign_pointest']).abs().max())
    print("  mean/lo/hi max abs diff vs frozen: ",
          (R[f'{cn}_mean']-t1[f'tau_I_{cn}_mean'].values).abs().max(),
          (R[f'{cn}_lo']-t1[f'tau_I_{cn}_lo'].values).abs().max(),
          (R[f'{cn}_hi']-t1[f'tau_I_{cn}_hi'].values).abs().max())
    d1=R[R[f'{cn}_frozen_resolved']!=R[f'{cn}_resolved_pointest']]
    print("  differ frozen vs pointest:", [f"{a}/{b}" for a,b in zip(d1.method,d1.dataset)])
    d2=R[R[f'{cn}_frozen_resolved']!=R[f'{cn}_resolved_boot']]
    print("  differ frozen vs boot:", [f"{a}/{b}" for a,b in zip(d2.method,d2.dataset)])
    print("  RESOLVED list (boot):", [f"{a}({c})/{b}" for a,b,c in zip(R[R[f'{cn}_resolved_boot']].method,R[R[f'{cn}_resolved_boot']].dataset,R[R[f'{cn}_resolved_boot']].configuration)])
    # (d) pass interval+magnitude but fail unit-level sign
    fd=R[R[f'{cn}_interval_excl0'] & R[f'{cn}_mag_ok'] & (R[f'{cn}_sign_pointest']<SIGN_MIN)]
    print("  (d) pass int+mag, fail unit sign<0.75:",
          [(f"{a}({c})/{b}", round(s,4)) for a,b,c,s in zip(fd.method,fd.dataset,fd.configuration,fd[f'{cn}_sign_pointest'])])
