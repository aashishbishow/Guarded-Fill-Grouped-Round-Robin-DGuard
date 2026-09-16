import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
base='/mnt/data/dpp_research'
agg=pd.read_csv(base+'/benchmark_aggregate.csv')
stress=pd.read_csv(base+'/stress_10001_aggregate.csv')
sc=pd.read_csv(base+'/selector_scaling.csv')
algs=['GRR','GuardedFill','AgeMIS','RotatingGreedy']
labels={'GRR':'Fixed GRR','GuardedFill':'Guarded Fill','AgeMIS':'Age-MIS','RotatingGreedy':'Rotating Greedy'}

# throughput N=1001 uniform
for workload in ['uniform','hotspot']:
    fig,ax=plt.subplots(figsize=(6.2,3.8))
    for alg in algs:
        g=agg[(agg.N==1001)&(agg.workload==workload)&(agg.algorithm==alg)].sort_values('p')
        ax.plot(g.p,g.throughput_mean,marker='o',label=labels[alg])
    ax.set_xlabel('Mean hunger-arrival probability per round')
    ax.set_ylabel('Completed meals per round')
    ax.set_title(f'Throughput at N=1001 ({workload} demand)')
    ax.grid(True,alpha=.25); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(f'{base}/fig_throughput_{workload}.pdf'); fig.savefig(f'{base}/fig_throughput_{workload}.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(6.2,3.8))
for alg in algs:
    g=agg[(agg.N==1001)&(agg.workload=='uniform')&(agg.algorithm==alg)].sort_values('p')
    ax.plot(g.p,g.mean_wait_mean,marker='o',label=labels[alg])
ax.set_xlabel('Mean hunger-arrival probability per round'); ax.set_ylabel('Mean waiting rounds')
ax.set_title('Mean wait at N=1001 (uniform demand)'); ax.grid(True,alpha=.25); ax.legend(frameon=False,ncol=2)
fig.tight_layout(); fig.savefig(base+'/fig_wait_uniform.pdf'); fig.savefig(base+'/fig_wait_uniform.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(6.2,3.8))
g=stress[(stress.p==0.4)&(stress.workload=='hotspot')].copy(); g=g.set_index('algorithm').loc[algs].reset_index()
ax.bar([labels[a] for a in g.algorithm],g.max_wait_max)
ax.set_ylabel('Maximum observed wait (rounds)'); ax.set_title('Tail waiting at N=10,001, hotspot p=0.4')
ax.tick_params(axis='x',rotation=18); ax.grid(True,axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(base+'/fig_tail_stress.pdf'); fig.savefig(base+'/fig_tail_stress.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(6.2,3.8))
for alg in ['GRR','GuardedFill','AgeMIS','RotatingGreedy']:
    if alg=='GuardedFill':
        # not in selector scaling file; omit because it is two linear passes and benchmarked end-to-end separately
        continue
    g=sc[sc.algorithm==alg].sort_values('N')
    ax.loglog(g.N,g.median_ms,marker='o',label=labels[alg])
ax.set_xlabel('Number of philosophers N'); ax.set_ylabel('Median selector time (ms)')
ax.set_title('Materialized selector scaling (Numba implementation)'); ax.grid(True,which='both',alpha=.25); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(base+'/fig_scaling.pdf'); fig.savefig(base+'/fig_scaling.png',dpi=220); plt.close(fig)

# relative improvement table helper
key=[]
for w in ['uniform','hotspot']:
  for p in [.05,.2,.4]:
    a=agg[(agg.N==1001)&(agg.p==p)&(agg.workload==w)&(agg.algorithm=='GuardedFill')].iloc[0]
    b=agg[(agg.N==1001)&(agg.p==p)&(agg.workload==w)&(agg.algorithm=='GRR')].iloc[0]
    key.append({'workload':w,'p':p,'throughput_gain_pct':100*(a.throughput_mean/b.throughput_mean-1),'mean_wait_reduction_pct':100*(1-a.mean_wait_mean/b.mean_wait_mean),'guarded_max_wait':a.max_wait_max,'grr_max_wait':b.max_wait_max})
pd.DataFrame(key).to_csv(base+'/key_improvements.csv',index=False)
print(pd.DataFrame(key).to_string(index=False))
