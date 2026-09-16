#!/usr/bin/env python3
"""
Regenerate grouped native summaries, paired Wilcoxon/Holm contrasts, and the
two final multi-architecture figures from the accepted 960-run dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "native_final" / "native_960_raw.csv"
OUT = HERE / "generated_native"
OUT.mkdir(exist_ok=True)

raw = pd.read_csv(DATA)
assert len(raw) == 960
assert set(raw["arch"]) == {"x86_64","arm64"}
assert set(raw["N"]) == {101,501}
assert set(raw["app"]) == {"sleep","ledger","kv"}
assert set(raw["workload"]) == {"saturated","bursty"}
assert set(raw["policy"]) == {"gfgrr","dguard","fifo","waiter","agemwis"}
assert set(raw["seed"]) == set(range(1,9))
assert set(raw["measure_s"]) == {2.0}

summary = raw.groupby(["arch","N","app","workload","policy"]).agg(
    throughput_s=("throughput_s","mean"),
    throughput_sd=("throughput_s","std"),
    p95_wait_ms=("p95_wait_ms","mean"),
    p99_wait_ms=("p99_wait_ms","mean"),
    max_wait_ms=("max_wait_ms","mean"),
    jain=("jain","mean"),
    zero_workers=("zero_workers","mean"),
    mean_wait_ms=("mean_wait_ms","mean"),
).reset_index()
summary.to_csv(OUT / "native_960_summary_recomputed.csv", index=False)

def holm_adjust(pvals):
    pvals=np.asarray(pvals,float); order=np.argsort(pvals)
    out=np.empty(len(pvals),float); prev=0.0; m=len(pvals)
    for rank,idx in enumerate(order):
        adj=min(1.0,(m-rank)*pvals[idx])
        adj=max(prev,adj); out[idx]=adj; prev=adj
    return out

rows=[]
for arch in ["x86_64","arm64"]:
    for app in ["sleep","ledger","kv"]:
        d=raw[(raw.arch==arch)&(raw.N==501)&(raw.app==app)&
              (raw.workload=="saturated")]
        for comparator in ["dguard","agemwis"]:
            recs=[]; ps=[]
            for metric in ["throughput_s","p99_wait_ms","jain"]:
                a=d[d.policy=="gfgrr"].sort_values("seed")[metric].to_numpy()
                b=d[d.policy==comparator].sort_values("seed")[metric].to_numpy()
                w=wilcoxon(a,b,alternative="two-sided",method="exact")
                ps.append(w.pvalue)
                recs.append(dict(
                    arch=arch,app=app,contrast=f"gfgrr_vs_{comparator}",
                    metric=metric,mean_gfgrr=a.mean(),
                    mean_comparator=b.mean(),
                    mean_difference_gf_minus_comparator=(a-b).mean(),
                    wilcoxon_p_raw=w.pvalue
                ))
            adj=holm_adjust(ps)
            for rec,padj in zip(recs,adj):
                rec["holm_p_3metric_family"]=padj
                rows.append(rec)
pd.DataFrame(rows).to_csv(OUT / "native_960_paired_stats_recomputed.csv", index=False)

# Final p99 figure: saturated sleep, N=501.
sleep=summary[(summary.N==501)&(summary.app=="sleep")&
              (summary.workload=="saturated")]
policies=["gfgrr","dguard","agemwis"]; labels=["GF-GRR","DGuard","Age-MWIS"]
arches=["x86_64","arm64"]; x=np.arange(len(policies)); width=.36
fig,ax=plt.subplots(figsize=(6.8,4.0))
for idx,arch in enumerate(arches):
    vals=[sleep[(sleep.arch==arch)&(sleep.policy==p)]["p99_wait_ms"].iloc[0]
          for p in policies]
    ax.bar(x+(idx-.5)*width,vals,width,label=arch)
ax.set_ylabel("p99 wait (ms)"); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_title("Final completion-accounted saturated sleep, N=501")
ax.legend(); fig.tight_layout()
fig.savefig(OUT/"fig_final_semantic_p99.pdf",bbox_inches="tight")
plt.close(fig)

# Throughput figure: saturated N=501 across payloads for GF-GRR/DGuard.
sat=summary[(summary.N==501)&(summary.workload=="saturated")]
apps=["sleep","ledger","kv"]; x=np.arange(len(apps)); width=.18
series=[
    ("x86_64","gfgrr","x86 GF-GRR"),
    ("x86_64","dguard","x86 DGuard"),
    ("arm64","gfgrr","ARM GF-GRR"),
    ("arm64","dguard","ARM DGuard"),
]
fig,ax=plt.subplots(figsize=(7.2,4.1))
for i,(arch,pol,label) in enumerate(series):
    vals=[sat[(sat.arch==arch)&(sat.app==app)&(sat.policy==pol)]["throughput_s"].iloc[0]
          for app in apps]
    ax.bar(x+(i-1.5)*width,vals,width,label=label)
ax.set_ylabel("Throughput (completions/s)")
ax.set_xticks(x); ax.set_xticklabels(["Sleep","Ledger","KV"])
ax.set_title("Final completion-accounted saturated throughput, N=501")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(OUT/"fig_final_semantic_throughput.pdf",bbox_inches="tight")
plt.close(fig)
print("Native analysis regenerated in", OUT)
