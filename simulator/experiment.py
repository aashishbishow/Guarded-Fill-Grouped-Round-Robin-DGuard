import numpy as np, pandas as pd, time, math, platform, json, os
from numba import njit

@njit(cache=True)
def better(a_count, a_age, b_count, b_age):
    if a_count != b_count:
        return a_count > b_count
    return a_age > b_age

@njit(cache=True)
def path_dp(hungry, age, l, r, sel):
    # maximize (cardinality, total age), path l..r inclusive
    n = hungry.shape[0]
    if r < l:
        return 0, 0
    m = r-l+1
    dc = np.zeros(m+2, np.int64)
    da = np.zeros(m+2, np.int64)
    take = np.zeros(m+2, np.uint8)
    for j in range(1,m+1):
        i=l+j-1
        c0=dc[j-1]; a0=da[j-1]
        c1=dc[j-2] + (1 if hungry[i] else 0)
        a1=da[j-2] + (age[i] if hungry[i] else 0)
        if hungry[i] and better(c1,a1,c0,a0):
            dc[j]=c1; da[j]=a1; take[j]=1
        else:
            dc[j]=c0; da[j]=a0; take[j]=0
    j=m
    while j>=1:
        if take[j]==1:
            i=l+j-1
            if hungry[i]: sel[i]=1
            j-=2
        else:
            j-=1
    return dc[m], da[m]

@njit(cache=True)
def age_mis_select(hungry, age, sel):
    n=hungry.shape[0]
    for i in range(n): sel[i]=0
    if n==1:
        if hungry[0]: sel[0]=1
        return
    # case A: exclude 0
    selA=np.zeros(n,np.uint8)
    cA,aA=path_dp(hungry,age,1,n-1,selA)
    # case B: include 0 if hungry -> exclude n-1,1; else same as A effectively
    selB=np.zeros(n,np.uint8)
    if hungry[0]:
        selB[0]=1
        cB,aB=path_dp(hungry,age,2,n-2,selB)
        cB+=1; aB+=age[0]
    else:
        cB=-1; aB=-1
    if better(cB,aB,cA,aA):
        for i in range(n): sel[i]=selB[i]
    else:
        for i in range(n): sel[i]=selA[i]

@njit(cache=True)
def grr_select(n,t,sel):
    for i in range(n): sel[i]=0
    m=n//2
    start=t % n
    for k in range(m):
        sel[(start+2*k)%n]=1

@njit(cache=True)
def rotating_greedy_select(hungry,t,sel):
    n=hungry.shape[0]
    for i in range(n): sel[i]=0
    start=t % n
    # greedy in rotated linear order
    prev_selected=0
    first_selected=0
    last_idx=-1
    for j in range(n):
        i=(start+j)%n
        if hungry[i] and prev_selected==0:
            sel[i]=1
            prev_selected=1
            if j==0: first_selected=1
            last_idx=i
        else:
            prev_selected=0 if sel[i]==0 else 1
    # first and last in rotated order are adjacent on cycle
    end=(start+n-1)%n
    if first_selected==1 and sel[end]==1:
        sel[end]=0

@njit(cache=True)
def single_waiter_select(hungry,age,sel):
    n=hungry.shape[0]
    for i in range(n): sel[i]=0
    best=-1; bestage=-1
    for i in range(n):
        if hungry[i] and age[i]>bestage:
            bestage=age[i]; best=i
    if best>=0: sel[best]=1


@njit(cache=True)
def guarded_fill_select(hungry, age, t, sel):
    n=hungry.shape[0]
    base=np.zeros(n,np.uint8)
    grr_select(n,t,base)
    candidate=np.zeros(n,np.uint8)
    extra=np.zeros(n,np.uint8)
    # mandatory = currently hungry vertices in the GRR set.
    # Exclude mandatory vertices and their neighbors from adaptive fill.
    for i in range(n):
        if hungry[i] and base[i]:
            sel[i]=1
        else:
            sel[i]=0
    for i in range(n):
        if hungry[i] and sel[i]==0 and sel[(i-1)%n]==0 and sel[(i+1)%n]==0:
            candidate[i]=1
    age_mis_select(candidate,age,extra)
    for i in range(n):
        if extra[i]: sel[i]=1

@njit(cache=True)
def quantile_hist(hist,q):
    total=0
    for x in hist: total+=x
    if total==0: return 0
    target=int(math.ceil(q*total))
    c=0
    for i in range(hist.shape[0]):
        c+=hist[i]
        if c>=target: return i
    return hist.shape[0]-1

@njit(cache=True)
def simulate(n, rounds, warmup, pvec, seed, alg):
    np.random.seed(seed)
    hungry=np.zeros(n,np.uint8)
    age=np.zeros(n,np.int64)
    sel=np.zeros(n,np.uint8)
    services=np.zeros(n,np.int64)
    hist=np.zeros(rounds+2,np.int64)
    total_serv=0; total_wait=0; max_wait=0
    for t in range(rounds+warmup):
        # exogenous Bernoulli opportunity every philosopher every round; accepted only if not already pending
        for i in range(n):
            u=np.random.random()
            if hungry[i]==0 and u < pvec[i]:
                hungry[i]=1; age[i]=0
        if alg==0:
            grr_select(n,t,sel)
        elif alg==1:
            rotating_greedy_select(hungry,t,sel)
        elif alg==2:
            age_mis_select(hungry,age,sel)
        elif alg==3:
            single_waiter_select(hungry,age,sel)
        else:
            guarded_fill_select(hungry,age,t,sel)
        # serve
        for i in range(n):
            if sel[i] and hungry[i]:
                if t>=warmup:
                    w=age[i]
                    total_serv+=1; total_wait+=w; services[i]+=1
                    if w>max_wait: max_wait=w
                    if w<hist.shape[0]: hist[w]+=1
                    else: hist[-1]+=1
                hungry[i]=0; age[i]=0
        # age unserved pending
        for i in range(n):
            if hungry[i]: age[i]+=1
    meas=rounds
    throughput=total_serv/meas
    mean_wait=total_wait/total_serv if total_serv>0 else 0.0
    p95=quantile_hist(hist,0.95); p99=quantile_hist(hist,0.99)
    s=0.0; ss=0.0
    for i in range(n):
        x=services[i]
        s+=x; ss+=x*x
    jain=(s*s)/(n*ss) if ss>0 else 1.0
    pending=0; max_pending=0
    for i in range(n):
        if hungry[i]:
            pending+=1
            if age[i]>max_pending: max_pending=age[i]
    return throughput, mean_wait, p95, p99, max_wait, jain, pending, max_pending

@njit(cache=True)
def structural_check(maxn):
    failures=0
    maxgap=0
    for n in range(3,maxn+1):
        m=n//2
        counts=np.zeros(n,np.int64)
        last=np.full(n,-1,np.int64)
        first=np.full(n,-1,np.int64)
        mg=0
        sel=np.zeros(n,np.uint8)
        for t in range(n):
            grr_select(n,t,sel)
            c=0
            for i in range(n):
                if sel[i]:
                    c+=1
                    if sel[(i+1)%n]: failures+=1
                    counts[i]+=1
                    if first[i]<0: first[i]=t
                    if last[i]>=0:
                        g=t-last[i]
                        if g>mg: mg=g
                    last[i]=t
            if c!=m: failures+=1
        # wrap gap
        for i in range(n):
            if last[i]>=0:
                g=(n-first[i])+last[i] # wrong? will recompute below outside
        # derive exact gaps with a second scan over 2n
        prev=np.full(n,-1,np.int64)
        mg=0
        for t in range(2*n):
            grr_select(n,t,sel)
            for i in range(n):
                if sel[i]:
                    if prev[i]>=0:
                        g=t-prev[i]
                        if g>mg: mg=g
                    prev[i]=t
        if mg>maxgap: maxgap=mg
    return failures,maxgap

def hotspot_pvec(n,p,hot_frac=0.1,mult=4.0):
    k=max(1,int(round(n*hot_frac)))
    ph=min(0.95,p*mult)
    if n==k:
        pl=ph
    else:
        pl=max(0.0,(p*n-ph*k)/(n-k))
    arr=np.full(n,pl,np.float64); arr[:k]=ph
    return arr,ph,pl

def run_bench(outdir):
    # warm JIT
    p=np.full(5,0.2)
    for a in range(5): simulate(5,10,2,p,1,a)
    structural_check(10)
    records=[]
    Ns=[101,1001]
    ps=[0.05,0.20,0.40]
    workloads=['uniform','hotspot']
    algs=['GRR','RotatingGreedy','AgeMIS','SingleWaiter','GuardedFill']
    reps=12
    rounds=3000; warm=300
    t0=time.perf_counter()
    for n in Ns:
      for pmean in ps:
       for workload in workloads:
        if workload=='uniform': pvec=np.full(n,pmean,np.float64); ph=pl=pmean
        else: pvec,ph,pl=hotspot_pvec(n,pmean)
        for rep in range(reps):
          seed=100000*n+1000*int(pmean*100)+100*workloads.index(workload)+rep
          for ai,an in enumerate(algs):
            st=time.perf_counter(); vals=simulate(n,rounds,warm,pvec,seed,ai); elapsed=time.perf_counter()-st
            records.append(dict(N=n,p=pmean,workload=workload,rep=rep,algorithm=an,
                throughput=vals[0],mean_wait=vals[1],p95_wait=vals[2],p99_wait=vals[3],max_wait=vals[4],
                jain=vals[5],pending_end=vals[6],max_pending_age=vals[7],runtime_s=elapsed,hot_p=ph,cold_p=pl))
    df=pd.DataFrame(records)
    df.to_csv(os.path.join(outdir,'benchmark_runs.csv'),index=False)
    agg=df.groupby(['N','p','workload','algorithm']).agg(
        throughput_mean=('throughput','mean'), throughput_sd=('throughput','std'),
        mean_wait_mean=('mean_wait','mean'), mean_wait_sd=('mean_wait','std'),
        p95_wait_mean=('p95_wait','mean'), p99_wait_mean=('p99_wait','mean'), max_wait_mean=('max_wait','mean'), max_wait_max=('max_wait','max'),
        jain_mean=('jain','mean'), runtime_mean_s=('runtime_s','mean')).reset_index()
    agg['capacity']=agg['N']//2
    agg['utilization']=agg['throughput_mean']/agg['capacity']
    agg.to_csv(os.path.join(outdir,'benchmark_aggregate.csv'),index=False)
    # paired bootstrap-ish t stats / scipy
    from scipy.stats import ttest_rel, wilcoxon
    sig=[]
    for (n,pmean,w),g in df.groupby(['N','p','workload']):
        piv=g.pivot(index='rep',columns='algorithm',values=['throughput','mean_wait','max_wait','jain'])
        for metric in ['throughput','mean_wait','max_wait','jain']:
            a=piv[metric]['AgeMIS'].values; b=piv[metric]['GRR'].values
            tr=ttest_rel(a,b)
            try: wr=wilcoxon(a,b)
            except: wr=type('O',(),{'statistic':np.nan,'pvalue':np.nan})()
            diff=a-b; d=diff.mean()/(diff.std(ddof=1)+1e-12)
            sig.append(dict(N=n,p=pmean,workload=w,metric=metric,AgeMIS_mean=a.mean(),GRR_mean=b.mean(),diff_mean=diff.mean(),paired_d=d,t_stat=tr.statistic,t_p=tr.pvalue,wilcoxon_stat=wr.statistic,wilcoxon_p=wr.pvalue))
    pd.DataFrame(sig).to_csv(os.path.join(outdir,'significance.csv'),index=False)

    # saturated structural/service gap verification
    sat=[]
    for n in [5,6,7,10,101,1001]:
        m=n//2
        # directly derive GRR service gaps over 3n rounds
        times=[[] for _ in range(n)]
        sel=np.zeros(n,np.uint8)
        for t in range(3*n):
            grr_select(n,t,sel)
            for i in np.nonzero(sel)[0]: times[int(i)].append(t)
        gaps=[b-a for ts in times for a,b in zip(ts[:-1],ts[1:])]
        sat.append(dict(N=n,capacity=m,grr_min_gap=min(gaps),grr_max_gap=max(gaps),grr_mean_gap=np.mean(gaps),expected_even_or_odd=('even' if n%2==0 else 'odd')))
    pd.DataFrame(sat).to_csv(os.path.join(outdir,'saturated_gap_bounds.csv'),index=False)
    fail,maxgap=structural_check(500)
    with open(os.path.join(outdir,'structural_verification.json'),'w') as f: json.dump({'N_checked':'3..500','failures':int(fail),'maximum_observed_interservice_gap':int(maxgap)},f,indent=2)

    # selector scaling: materialized scheduling, not full simulation
    scaling=[]
    for n in [1_000,10_000,100_000,1_000_000]:
        rng=np.random.default_rng(123+n)
        hungry=(rng.random(n)<0.35).astype(np.uint8)
        age=rng.integers(0,50,size=n,dtype=np.int64)
        sel=np.zeros(n,np.uint8)
        # warm each shape
        rotating_greedy_select(hungry,0,sel); age_mis_select(hungry,age,sel); grr_select(n,0,sel)
        repscale=30 if n<=10000 else (10 if n<=100000 else 3)
        for ai,an in [(0,'GRR'),(1,'RotatingGreedy'),(2,'AgeMIS')]:
            ts=[]
            for r in range(repscale):
                st=time.perf_counter()
                if ai==0: grr_select(n,r,sel)
                elif ai==1: rotating_greedy_select(hungry,r,sel)
                else: age_mis_select(hungry,age,sel)
                ts.append(time.perf_counter()-st)
            scaling.append(dict(N=n,algorithm=an,repetitions=repscale,median_ms=1000*np.median(ts),mean_ms=1000*np.mean(ts),p95_ms=1000*np.percentile(ts,95)))
    pd.DataFrame(scaling).to_csv(os.path.join(outdir,'selector_scaling.csv'),index=False)

    meta={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'benchmark_elapsed_s':time.perf_counter()-t0,
          'rounds_measured':rounds,'warmup_rounds':warm,'replicates':reps,'algorithms':algs,'Ns':Ns,'arrival_probabilities':ps,'workloads':workloads,
          'model':'synchronous rounds; one pending hunger request per philosopher; Bernoulli arrival opportunity each round when not pending; unit eating duration'}
    with open(os.path.join(outdir,'experiment_metadata.json'),'w') as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta,indent=2))
    print('\nKey aggregate rows:')
    print(agg[(agg.N==1001)&(agg.p==0.2)].to_string(index=False))

if __name__=='__main__': run_bench('/mnt/data/dpp_research')
