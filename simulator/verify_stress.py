import sys,os,time,json,itertools
sys.path.insert(0,'/mnt/data/dpp_research')
import experiment as ex
import numpy as np, pandas as pd

def brute_opt(hungry,age):
    n=len(hungry); bc=-1; ba=-1; bmask=0
    for mask in range(1<<n):
        ok=True; c=0; a=0
        for i in range(n):
            if (mask>>i)&1:
                if not hungry[i] or ((mask>>((i+1)%n))&1): ok=False; break
                c+=1; a+=int(age[i])
        if ok and (c>bc or (c==bc and a>ba)):
            bc,ba,bmask=c,a,mask
    return bc,ba,bmask

def verify_age_mis():
    rng=np.random.default_rng(20260915); cases=0; fails=0
    for n in range(3,13):
        for _ in range(150):
            hungry=(rng.random(n)<rng.uniform(.15,.9)).astype(np.uint8)
            age=rng.integers(0,20,size=n,dtype=np.int64)
            sel=np.zeros(n,np.uint8); ex.age_mis_select(hungry,age,sel)
            c=int(sel.sum()); a=int((sel*age).sum())
            bc,ba,_=brute_opt(hungry,age)
            # independent and subset hungry
            safe=all(not(sel[i] and sel[(i+1)%n]) for i in range(n)) and all(not sel[i] or hungry[i] for i in range(n))
            cases+=1
            if c!=bc or a!=ba or not safe: fails+=1
    return {'cases':cases,'failures':fails,'n_range':'3..12','objective':'lexicographic (max cardinality, then total waiting age)'}

def stress():
    rec=[]; Ns=[10001]; ps=[.2,.4]; ws=['uniform','hotspot']; reps=5; rounds=1200; warm=200
    algs=[(0,'GRR'),(1,'RotatingGreedy'),(2,'AgeMIS'),(4,'GuardedFill')]
    for n in Ns:
      for p in ps:
       for w in ws:
        if w=='uniform': pv=np.full(n,p); ph=pl=p
        else: pv,ph,pl=ex.hotspot_pvec(n,p)
        for rep in range(reps):
          seed=900000+n+1000*int(p*100)+100*ws.index(w)+rep
          for ai,an in algs:
            st=time.perf_counter(); vals=ex.simulate(n,rounds,warm,pv,seed,ai); elapsed=time.perf_counter()-st
            rec.append(dict(N=n,p=p,workload=w,rep=rep,algorithm=an,throughput=vals[0],mean_wait=vals[1],p95_wait=vals[2],p99_wait=vals[3],max_wait=vals[4],jain=vals[5],pending_end=vals[6],max_pending_age=vals[7],runtime_s=elapsed,hot_p=ph,cold_p=pl))
    df=pd.DataFrame(rec); df.to_csv('/mnt/data/dpp_research/stress_10001_runs.csv',index=False)
    agg=df.groupby(['N','p','workload','algorithm']).agg(throughput_mean=('throughput','mean'),throughput_sd=('throughput','std'),mean_wait_mean=('mean_wait','mean'),max_wait_max=('max_wait','max'),jain_mean=('jain','mean'),runtime_mean_s=('runtime_s','mean')).reset_index()
    agg.to_csv('/mnt/data/dpp_research/stress_10001_aggregate.csv',index=False)
    return agg

v=verify_age_mis(); json.dump(v,open('/mnt/data/dpp_research/age_mis_bruteforce_verification.json','w'),indent=2)
print('verification',v)
a=stress(); print(a.to_string(index=False))
