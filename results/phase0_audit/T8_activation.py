import gzip,csv,collections,os
ROOT='/home/sypark/workspace/FairGNN-Eval'
main=list(csv.DictReader(open(ROOT+'/results/1_main_package_vs_intervention.csv')))
store=collections.defaultdict(dict)
with gzip.open(ROOT+'/results/per_unit_metrics.csv.gz','rt') as f:
    for r in csv.DictReader(f):
        if r['protocol']=='controlled' and r['selector']=='common_bce':
            store[(r['method'],r['dataset'])][(r['split_id'],r['run_id'],r['state'])]=r

def sname(m,cfg):
    if m=='BIND': return 'BIND-'+cfg
    if m=='FairSIN': return 'FairSIN-'+cfg
    return m

out=[]
for r in main:
    key=(sname(r['method'],r['configuration']),r['dataset'])
    d=store.get(key)
    if d is None:
        out.append((r['method'],r['dataset'],r['configuration'],'NA','NA','NA','NA','store key %s missing'%(key,))); continue
    units=sorted(set((k[0],k[1]) for k in d))
    nid=0; mx={'auc':0.0,'dp':0.0,'eo':0.0}; n=0
    for u in units:
        p=d.get((u[0],u[1],'M_plus_I')); m=d.get((u[0],u[1],'M_minus_I'))
        if p is None or m is None: continue
        n+=1
        diffs={c:abs(float(p[c])-float(m[c])) for c in ('auc','dp','eo')}
        if all(v==0.0 for v in diffs.values()): nid+=1
        for c in diffs: mx[c]=max(mx[c],diffs[c])
    out.append((r['method'],r['dataset'],r['configuration'],nid,mx['auc'],mx['dp'],mx['eo'],'n_units=%d'%n))

with open(ROOT+'/results/phase0_audit/T8_activation.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['method','dataset','configuration','n_units_identical','max_abs_dauc','max_abs_ddp','max_abs_deo','note'])
    for o in out: w.writerow(o)
for o in out: print(o)
