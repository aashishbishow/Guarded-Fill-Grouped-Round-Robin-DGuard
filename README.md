# Guarded-Fill Grouped-Round-Robin (GF-GRR / DGuard)

> **From Fixed Group Round-Robin to Guarded Adaptivity in the Dining Philosophers Problem: Formal Bounds, Native Multi-Architecture Validation, and Coordinator Removal**  
> *Author:* Aashish BishowKarma

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-verified-success.svg)](COMPLETE_SOURCE_MANIFEST.md)

This repository contains the complete research source code, formal/model verifications, empirical datasets, and full reproducibility suite for the paper **"From Fixed Group Round-Robin to Guarded Adaptivity in the Dining Philosophers Problem"**.

---

## 📌 Overview & Contributions

The Dining Philosophers Problem (DPP) serves as a foundational paradigm for studying concurrency, mutual exclusion, deadlocks, and starvation. Building upon prior rotating group round-robin formulations (Lee, 2023), this research investigates whether deterministic service deadlines can be preserved while reclaiming unused capacity when scheduled philosophers are not hungry.

### Key Contributions

1. **Formalization & Bounds of Fixed GRR**:
   - Graph-theoretic formalization of group round-robin as rotations of maximum independent sets on cycle graphs $C_N$.
   - Rigorous characterization of safety, capacity, and deterministic service-opportunity bounds (proving a worst-case wait bound of $N$ rounds for odd $N$).
2. **Guarded-Fill GRR (GF-GRR)**:
   - A conservative adaptive policy that strictly preserves the deterministic GRR service guard while filling idle, conflict-free capacity with nonconflicting hungry philosophers.
   - Formal proof showing GF-GRR inherits the exact GRR service-opportunity bound and weakly dominates fixed GRR in instantaneous service count for any hunger state.
3. **Age-MWIS Reference Benchmark**:
   - An adaptive reference computing exact maximum-weight independent sets on $C_N$ via cyclic dynamic programming.
4. **Coordinator-Free DGuard**:
   - A decentralized, epoch-based variant removing the centralized arbiter.
5. **Multi-Architecture Native Concurrency Validation**:
   - Full C++17 implementation using POSIX threads, fork mutexes, and 2.0-second completion-accounted measurement windows.
   - A paired empirical matrix of **960 accepted runs** across **x86_64** and **ARM64** hardware evaluating sleep, ledger, and two-shard key-value payloads under saturated and bursty demand.
6. **Platform-Dependent Frontier**:
   - Empirical findings demonstrate that coordinator removal is not universally superior: DGuard excels on ARM64 for memory-bound payloads (up to +55.2% throughput) but incurs p99 latency and fairness penalties, while on x86_64 GF-GRR remains competitive or superior with significantly better fairness.

---

## 📂 Repository Structure

```
.
├── paper/
│   ├── paper.pdf                  # Formatted manuscript (IEEE Transactions style)
│   ├── paper.tex                  # LaTeX source code
│   └── references.bib             # BibTeX bibliography
│
├── simulator/
│   ├── experiment.py              # Discrete-event synchronous simulator (720-run matrix)
│   ├── verify_stress.py           # Formal verification & stress tests up to N=10,001
│   ├── make_figures.py            # Manuscript figure generator
│   ├── requirements.txt           # Simulator-specific Python dependencies
│   └── Dining_Philosophers_FullScale_Reproduction.executed.ipynb # Executed notebook
│
├── native/
│   ├── final_native_threads.cpp   # Verified C++17 multi-threaded benchmark (960-run matrix)
│   └── process_isolated_dguard.cpp# Process-isolated DGuard implementation (shared memory)
│
├── data/
│   ├── native_final/              # Raw and aggregated data for 960 native runs
│   │   ├── native_960_raw.csv
│   │   ├── native_960_summary.csv
│   │   └── native_960_paired_stats.csv
│   ├── process_isolated/          # Data for process-isolated DGuard experiments
│   └── simulator/                 # Synchronous simulator outputs, stress test & scaling data
│
├── analysis/
│   ├── analyze_simulator.py       # Recomputes simulator paired statistics and trade-off tables
│   └── analyze_final_native.py    # Recomputes native summaries, p99 stats, and trade-off tables
│
├── audit/
│   ├── README_SUPERSEDED.md       # Audit notes documenting superseded pre-submission runs
│   └── superseded_native_SHA_9cff13.cpp # Pre-audit source retained for audit completeness
│
├── workflows/
│   └── reproduce_final_native.yml # GitHub Actions CI workflow for x86_64 and ARM64 replication
│
├── COMPLETE_SOURCE_MANIFEST.md    # Exhaustive inventory of components & reproduction notes
├── SHA256SUMS.txt                 # Cryptographic verification checksums for all source files
└── requirements.txt               # Top-level Python environment requirements
```

---

## 🔬 Algorithms Under Evaluation

| Policy | Coordination | Service Bound Guarantee | Adaptivity | Complexity per Cycle |
| :--- | :--- | :--- | :--- | :--- |
| **Fixed GRR** | Centralized | Strict deterministic bound ($\le N$) | None (static rotation) | $O(1)$ |
| **Guarded-Fill GRR (GF-GRR)** | Centralized | Strict deterministic bound ($\le N$) | Work-conserving greedy fill | $O(N)$ |
| **Age-MWIS** | Centralized | Heuristic (age-weighted) | Full cyclic DP optimization | $O(N)$ |
| **DGuard** | Decentralized | Epoch-based local coordination | Distributed opportunistic | $O(1)$ amortized |
| **FIFO Waiter** | Centralized | Strict arrival order | Queue-based admission | $O(N)$ |
| **$N-1$ Waiter** | Centralized | Bounded tickets | Asymmetric admission | $O(1)$ |

---

## 🚀 Reproduction Guide

### 1. Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/aashishbishow/Guarded-Fill-Grouped-Round-Robin-DGuard.git
cd Guarded-Fill-Grouped-Round-Robin-DGuard

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

### 2. Synchronous Simulator & Formal Verification

Run the core simulator and stress verification suite:

```bash
# Run simulator experiments (720 core conditions)
python simulator/experiment.py

# Run large-scale stress tests (N up to 10,001) and brute-force independent-set verification
python simulator/verify_stress.py

# Recompute simulator statistics and paper tables
python analysis/analyze_simulator.py
```

---

### 3. Native Multi-Threaded Concurrency Matrix

The native benchmark requires a C++17 compiler supporting POSIX threads (`pthread`).

#### Build with Sanitizers (Smoke Test)
```bash
g++ -O0 -fsanitize=address,undefined -std=c++17 -pthread native/final_native_threads.cpp -o native_smoke
./native_smoke --smoke-test
```

#### Production Compilation & Run
```bash
# Compile with -O3 optimizations
g++ -O3 -std=c++17 -pthread native/final_native_threads.cpp -o native_final

# Run individual configuration example:
# ./native_final <policy> <N> <payload> <workload> <seed> <duration_sec>
./native_final GF-GRR 101 sleep saturated 1 2.0
```

#### Recompute Native Empirical Statistics
```bash
python analysis/analyze_final_native.py
```

---

### 4. Process-Isolated DGuard Replication

Compile and run the separate process-isolated DGuard implementation:

```bash
g++ -O3 -std=c++17 -pthread native/process_isolated_dguard.cpp -o process_isolated_dguard
./process_isolated_dguard
```

---

### 5. Automated Multi-Architecture CI Replication

See [workflows/reproduce_final_native.yml](workflows/reproduce_final_native.yml) for the automated GitHub Actions workflow designed to compile and execute the complete 960-run matrix across both `ubuntu-latest` (x86_64) and ARM64 runners.

---

## 🛡️ Integrity & Checksums

To verify file integrity, all tracked sources and datasets are checked against [SHA256SUMS.txt](SHA256SUMS.txt):

```bash
# Verify integrity on Linux/macOS:
sha256sum -c SHA256SUMS.txt

# Or on Windows PowerShell:
Get-Content SHA256SUMS.txt | ForEach-Object {
    $hash, $file = $_ -split '\s+', 2
    if ($file -and (Test-Path $file)) {
        $computed = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLower()
        if ($computed -eq $hash.ToLower()) {
            Write-Host "OK: $file" -ForegroundColor Green
        } else {
            Write-Host "FAILED: $file" -ForegroundColor Red
        }
    }
}
```

---

---

## 📄 License & Attribution

This project is made available for academic research, reproduction, and educational purposes.
Copyright (c) 2026 Aashish BishowKarma.
