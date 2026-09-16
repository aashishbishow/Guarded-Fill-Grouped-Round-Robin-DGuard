#!/usr/bin/env python3
"""
Recompute the principal simulator statistics used in the manuscript from
data/simulator/benchmark_runs.csv.

Outputs:
  simulator_aggregate_recomputed.csv
  primary_gfgrr_vs_grr_recomputed.csv
  pairwise_three_families_recomputed.csv
  three_way_tradeoff_recomputed.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "simulator"
OUT = HERE / "generated_simulator"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "benchmark_runs.csv")

agg = df.groupby(["N","p","workload","algorithm"]).agg(
    throughput_mean=("throughput","mean"),
    throughput_sd=("throughput","std"),
    mean_wait_mean=("mean_wait","mean"),
    mean_wait_sd=("mean_wait","std"),
    p95_wait_mean=("p95_wait","mean"),
    p99_wait_mean=("p99_wait","mean"),
    max_wait_mean=("max_wait","mean"),
    max_wait_max=("max_wait","max"),
    jain_mean=("jain","mean"),
    runtime_mean_s=("runtime_s","mean"),
).reset_index()
agg["capacity"] = agg["N"] // 2
agg["utilization"] = agg["throughput_mean"] / agg["capacity"]
agg.to_csv(OUT / "simulator_aggregate_recomputed.csv", index=False)

def holm(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        v = min(1.0, (m-rank) * pvals[idx])
        running = max(running, v)
        adjusted[idx] = running
    return adjusted

# Primary GF-GRR vs GRR family: throughput + mean_wait across 12 conditions = 24 tests.
records = []
for (N,p,workload), g in df.groupby(["N","p","workload"]):
    for metric in ["throughput","mean_wait"]:
        piv = g.pivot(index="rep", columns="algorithm", values=metric)
        a = piv["GuardedFill"].to_numpy()
        b = piv["GRR"].to_numpy()
        diff = a-b
        n = len(diff)
        se = diff.std(ddof=1)/np.sqrt(n)
        q = t.ppf(0.975, n-1)
        tt = ttest_rel(a,b)
        ww = wilcoxon(a,b,alternative="two-sided",method="auto")
        records.append(dict(
            N=N,p=p,workload=workload,metric=metric,
            GuardedFill_mean=a.mean(),GRR_mean=b.mean(),
            diff_mean=diff.mean(),
            ci95_low=diff.mean()-q*se,ci95_high=diff.mean()+q*se,
            paired_t_p=tt.pvalue,wilcoxon_p=ww.pvalue
        ))
primary = pd.DataFrame(records)
primary["paired_t_p_holm"] = holm(primary["paired_t_p"].to_numpy())
primary["wilcoxon_p_holm"] = holm(primary["wilcoxon_p"].to_numpy())
primary.to_csv(OUT / "primary_gfgrr_vs_grr_recomputed.csv", index=False)

# Three pairwise Wilcoxon/Holm families used for the merged narrative.
families = [
    ("GF_vs_GRR","GuardedFill","GRR"),
    ("AgeMIS_vs_GRR","AgeMIS","GRR"),
    ("GF_vs_AgeMIS","GuardedFill","AgeMIS"),
]
all_rows = []
for fam,A,B in families:
    rows=[]
    for (N,p,workload), g in df.groupby(["N","p","workload"]):
        for metric in ["throughput","mean_wait"]:
            piv = g.pivot(index="rep", columns="algorithm", values=metric)
            a=piv[A].to_numpy(); b=piv[B].to_numpy()
            w=wilcoxon(a,b,alternative="two-sided",method="auto")
            rows.append(dict(
                family=fam,N=N,p=p,workload=workload,metric=metric,
                A=A,B=B,A_mean=a.mean(),B_mean=b.mean(),
                diff_mean=(a-b).mean(),wilcoxon_stat=w.statistic,
                wilcoxon_p=w.pvalue
            ))
    famdf=pd.DataFrame(rows)
    famdf["wilcoxon_p_holm"]=holm(famdf["wilcoxon_p"].to_numpy())
    all_rows.append(famdf)
pd.concat(all_rows,ignore_index=True).to_csv(
    OUT / "pairwise_three_families_recomputed.csv", index=False
)

# GF recovery fraction relative to Age-MWIS adaptive gain.
rows=[]
for (N,p,w), g in agg.groupby(["N","p","workload"]):
    d=g.set_index("algorithm")
    if not {"GRR","GuardedFill","AgeMIS"}.issubset(d.index):
        continue
    grr=float(d.loc["GRR","throughput_mean"])
    gf=float(d.loc["GuardedFill","throughput_mean"])
    age=float(d.loc["AgeMIS","throughput_mean"])
    denom=age-grr
    rows.append(dict(
        N=N,p=p,workload=w,
        GRR_throughput=grr,GF_throughput=gf,AgeMIS_throughput=age,
        GF_recovery_fraction_of_AgeMIS_gain=((gf-grr)/denom if denom else np.nan),
        GF_gap_to_AgeMIS_percent=(100*(age-gf)/age if age else np.nan),
        GRR_max_wait=int(d.loc["GRR","max_wait_max"]),
        GF_max_wait=int(d.loc["GuardedFill","max_wait_max"]),
        AgeMIS_max_wait=int(d.loc["AgeMIS","max_wait_max"]),
    ))
pd.DataFrame(rows).to_csv(OUT / "three_way_tradeoff_recomputed.csv", index=False)
print("Simulator analysis regenerated in", OUT)
