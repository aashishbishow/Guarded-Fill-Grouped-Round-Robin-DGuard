#include <pthread.h>
#include <semaphore.h>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
static inline uint64_t now_ns(){
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
static inline void sleep_us(uint64_t us){ if(us) std::this_thread::sleep_for(std::chrono::microseconds(us)); }

static inline uint64_t splitmix64(uint64_t &x){
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
static inline double u01(uint64_t &s){ return (splitmix64(s) >> 11) * (1.0/9007199254740992.0); }

struct Score {
    int owed = 0;
    int count = 0;
    uint64_t age = 0;
};
static inline bool better(const Score&a,const Score&b){
    if(a.owed!=b.owed) return a.owed>b.owed;
    if(a.count!=b.count) return a.count>b.count;
    return a.age>b.age;
}
static inline Score add(const Score&a,const Score&b){ return {a.owed+b.owed,a.count+b.count,a.age+b.age}; }

struct LocalMetrics {
    std::vector<double> waits_ms;
    uint64_t meals_recorded=0;
};

struct Bench;
struct WorkerCtx { Bench* b; int id; uint64_t rng; LocalMetrics metrics; uint64_t total_meals=0; };

struct Bench {
    int N;
    std::string policy;
    std::string workload;
    uint64_t seed;
    int tick_us;
    int sched_jitter_us;
    int dispatch_jitter_us;
    double warmup_s;
    double measure_s;
    int eat_lo_us, eat_hi_us;
    int think_lo_us, think_hi_us;
    std::string app = "sleep";
    int epoch_us = 1200;
    std::vector<long long> balances;
    std::vector<std::array<uint64_t,64>> kvshards;

    std::mutex fifo_mu;
    std::deque<int> fifo_q;
    std::vector<uint8_t> fifo_queued;

    pthread_mutex_t* forks=nullptr;
    pthread_mutex_t* gmu=nullptr;
    pthread_cond_t* gcv=nullptr;
    bool* grant=nullptr;
    std::atomic<uint8_t>* hungry=nullptr;
    std::atomic<uint8_t>* active=nullptr;
    std::atomic<uint8_t>* owed=nullptr;
    std::atomic<uint64_t>* req_ns=nullptr;
    std::atomic<uint64_t>* owed_since_ns=nullptr;

    sem_t room;
    bool room_init=false;

    std::atomic<bool> start_flag{false};
    std::atomic<bool> generation_stop{false};
    std::atomic<int> alive{0};
    std::atomic<uint64_t> scheduler_ticks{0};
    std::atomic<uint64_t> guard_obligations{0};
    std::atomic<uint64_t> max_owed_age_ns{0};
    uint64_t measure_start_ns=0, measure_end_ns=0;

    std::mutex sched_mu;
    std::condition_variable sched_cv;
    bool sched_ping=false;

    std::vector<pthread_t> threads;
    std::vector<WorkerCtx> ctx;
    std::thread scheduler;

    Bench(int n):N(n){}
    ~Bench(){ cleanup(); }

    void alloc(){
        forks=(pthread_mutex_t*)calloc(N,sizeof(pthread_mutex_t));
        gmu=(pthread_mutex_t*)calloc(N,sizeof(pthread_mutex_t));
        gcv=(pthread_cond_t*)calloc(N,sizeof(pthread_cond_t));
        grant=(bool*)calloc(N,sizeof(bool));
        hungry=new std::atomic<uint8_t>[N];
        active=new std::atomic<uint8_t>[N];
        owed=new std::atomic<uint8_t>[N];
        req_ns=new std::atomic<uint64_t>[N];
        owed_since_ns=new std::atomic<uint64_t>[N];
        for(int i=0;i<N;i++){
            pthread_mutex_init(&forks[i],nullptr);
            pthread_mutex_init(&gmu[i],nullptr);
            pthread_cond_init(&gcv[i],nullptr);
            hungry[i]=0; active[i]=0; owed[i]=0; req_ns[i]=0; owed_since_ns[i]=0;
        }
        if(policy=="waiter"){
            sem_init(&room,0,(unsigned)std::max(1,N-1)); room_init=true;
        }
        fifo_queued.assign(N,0);
        balances.assign(N, 1000000);
        kvshards.resize(N);
        for(int i=0;i<N;i++) for(size_t j=0;j<kvshards[i].size();j++) kvshards[i][j]=(uint64_t)(i+1)*0x9e3779b97f4a7c15ULL + j;
        threads.resize(N); ctx.resize(N);
    }
    void cleanup(){
        if(forks){ for(int i=0;i<N;i++) pthread_mutex_destroy(&forks[i]); free(forks); forks=nullptr; }
        if(gmu){ for(int i=0;i<N;i++) pthread_mutex_destroy(&gmu[i]); free(gmu); gmu=nullptr; }
        if(gcv){ for(int i=0;i<N;i++) pthread_cond_destroy(&gcv[i]); free(gcv); gcv=nullptr; }
        free(grant); grant=nullptr;
        delete[] hungry; hungry=nullptr; delete[] active; active=nullptr; delete[] owed; owed=nullptr;
        delete[] req_ns; req_ns=nullptr; delete[] owed_since_ns; owed_since_ns=nullptr;
        if(room_init){ sem_destroy(&room); room_init=false; }
    }

    int rnd_range(uint64_t &r,int lo,int hi){ if(hi<=lo) return lo; return lo + (int)(splitmix64(r) % (uint64_t)(hi-lo+1)); }
    int sample_eat(uint64_t &r){
        // bounded log-normal-ish mixture to create stragglers without pathological runs
        double a=u01(r), b=u01(r); double z=std::sqrt(-2.0*std::log(std::max(a,1e-12)))*std::cos(6.28318530718*b);
        double mid=0.5*(eat_lo_us+eat_hi_us); double v=mid*std::exp(0.45*z);
        return (int)std::max((double)eat_lo_us,std::min((double)eat_hi_us,v));
    }
    int sample_think(uint64_t &r){
        if(workload=="saturated") return 0;
        if(workload=="bursty"){
            // mixture: mostly short think, occasional longer pause
            if(u01(r)<0.15) return rnd_range(r,think_hi_us,think_hi_us*3);
            return rnd_range(r,think_lo_us,think_hi_us);
        }
        return rnd_range(r,think_lo_us,think_hi_us);
    }
    void do_payload(WorkerCtx &c, int L, int R){
        if(app=="sleep"){
            sleep_us((uint64_t)sample_eat(c.rng));
            return;
        }
        if(app=="ledger"){
            int reps=32 + (int)(splitmix64(c.rng)%97);
            long long delta=0;
            uint64_t x=c.rng ^ (uint64_t)(c.id+1);
            for(int k=0;k<reps;k++){ x ^= x<<13; x ^= x>>7; x ^= x<<17; delta += (long long)((x&7)+1); delta -= (long long)(((x>>3)&7)+1); }
            long long amt=(long long)((splitmix64(c.rng)%7)+1);
            balances[L]-=amt; balances[R]+=amt;
            balances[L]+=delta; balances[R]-=delta; // preserves total; adds real arithmetic/memory work
            return;
        }
        // kv: update two locked shards with variable memory/compute work
        int reps=128 + (int)(splitmix64(c.rng)%385);
        uint64_t x=c.rng ^ (uint64_t)(c.id+17);
        for(int k=0;k<reps;k++){
            x ^= x<<13; x ^= x>>7; x ^= x<<17;
            auto &a=kvshards[L][(x>>1)&63];
            auto &b=kvshards[R][(x>>9)&63];
            a = (a ^ x) * 0x9e3779b185ebca87ULL + (b>>1);
            b = (b + (x^a)) * 0xc2b2ae3d27d4eb4fULL;
        }
    }
    bool is_guard_member(int i, uint64_t epoch) const {
        int m=N/2;
        // i in {epoch+2k mod N, k=0..m-1}; O(N) avoided by modular parity-like direct test via bounded search for odd/even correctness.
        // For benchmark sizes this tiny loop is only used once per acquisition attempt; optimize with modular inverse when N odd.
        if((N&1)==1){
            // inverse of 2 modulo odd N is (N+1)/2
            uint64_t d=((uint64_t)i + (uint64_t)N - (epoch%(uint64_t)N))%(uint64_t)N;
            uint64_t k=(d*(uint64_t)((N+1)/2))%(uint64_t)N;
            return k<(uint64_t)m;
        } else {
            uint64_t d=((uint64_t)i + (uint64_t)N - (epoch%(uint64_t)N))%(uint64_t)N;
            return (d%2==0) && (d/2<(uint64_t)m);
        }
    }
    void ping_scheduler(){
        if(policy=="waiter" || policy=="dguard") return;
        { std::lock_guard<std::mutex> lk(sched_mu); sched_ping=true; }
        sched_cv.notify_one();
    }

    static void* worker_entry(void* p){
        WorkerCtx* c=(WorkerCtx*)p; c->b->worker(*c); return nullptr;
    }
    void worker(WorkerCtx &c){
        int i=c.id, L=i, R=(i+1)%N;
        while(!start_flag.load(std::memory_order_acquire)) std::this_thread::yield();
        // stagger initial hunger slightly in dynamic workloads
        int initial = (workload=="saturated") ? 0 : sample_think(c.rng);
        sleep_us(initial);
        while(true){
            if(generation_stop.load(std::memory_order_acquire)) break;
            uint64_t arr=now_ns();
            req_ns[i].store(arr,std::memory_order_release);
            hungry[i].store(1,std::memory_order_release);
            ping_scheduler();

            if(policy=="dguard"){
                while(true){
                    uint64_t t=now_ns();
                    uint64_t base=measure_start_ns; uint64_t epoch=(base && t>base)?((t-base)/(uint64_t)std::max(1,epoch_us*1000)):0;
                    if(is_guard_member(i,epoch)) owed[i].store(1,std::memory_order_release);
                    int pl=(i-1+N)%N, pr=(i+1)%N;
                    bool mine=owed[i].load(std::memory_order_acquire);
                    if(!mine && (owed[pl].load(std::memory_order_acquire)||owed[pr].load(std::memory_order_acquire))){ sleep_us(20); continue; }
                    int a=std::min(L,R), bb=std::max(L,R);
                    if(pthread_mutex_trylock(&forks[a])!=0){ sleep_us(mine?5:20); continue; }
                    if(pthread_mutex_trylock(&forks[bb])!=0){ pthread_mutex_unlock(&forks[a]); sleep_us(mine?5:20); continue; }
                    if(!mine && (owed[pl].load(std::memory_order_acquire)||owed[pr].load(std::memory_order_acquire))){ pthread_mutex_unlock(&forks[bb]); pthread_mutex_unlock(&forks[a]); sleep_us(20); continue; }
                    uint64_t st=now_ns();
                    hungry[i].store(0,std::memory_order_release);
                    active[i].store(1,std::memory_order_release);
                    int dj=rnd_range(c.rng,0,dispatch_jitter_us); sleep_us(dj);
                    do_payload(c,L,R);
                    active[i].store(0,std::memory_order_release);
                    owed[i].store(0,std::memory_order_release);
                    pthread_mutex_unlock(&forks[bb]); pthread_mutex_unlock(&forks[a]);
                    if(arr>=measure_start_ns && arr<measure_end_ns){ c.metrics.waits_ms.push_back((st-arr)/1e6); c.metrics.meals_recorded++; }
                    c.total_meals++;
                    break;
                }
            } else if(policy=="waiter"){
                sem_wait(&room);
                pthread_mutex_lock(&forks[L]);
                int dj=rnd_range(c.rng,0,dispatch_jitter_us); sleep_us(dj);
                pthread_mutex_lock(&forks[R]);
                active[i].store(1,std::memory_order_release);
                uint64_t st=now_ns();
                hungry[i].store(0,std::memory_order_release);
                do_payload(c,L,R);
                active[i].store(0,std::memory_order_release);
                pthread_mutex_unlock(&forks[R]); pthread_mutex_unlock(&forks[L]); sem_post(&room);
                if(arr>=measure_start_ns && arr<measure_end_ns){ c.metrics.waits_ms.push_back((st-arr)/1e6); c.metrics.meals_recorded++; }
                c.total_meals++;
            } else {
                pthread_mutex_lock(&gmu[i]);
                while(!grant[i]) pthread_cond_wait(&gcv[i],&gmu[i]);
                grant[i]=false;
                pthread_mutex_unlock(&gmu[i]);
                int dj=rnd_range(c.rng,0,dispatch_jitter_us); sleep_us(dj);
                // Central policies should be conflict-free; forks remain real mutexes to expose races/OS effects.
                pthread_mutex_lock(&forks[L]);
                pthread_mutex_lock(&forks[R]);
                uint64_t st=now_ns();
                do_payload(c,L,R);
                pthread_mutex_unlock(&forks[R]); pthread_mutex_unlock(&forks[L]);
                active[i].store(0,std::memory_order_release);
                if(arr>=measure_start_ns && arr<measure_end_ns){ c.metrics.waits_ms.push_back((st-arr)/1e6); c.metrics.meals_recorded++; }
                c.total_meals++;
                ping_scheduler();
            }
            int th=sample_think(c.rng); sleep_us(th);
        }
        // If stop hit while hungry in central mode, finish that outstanding request before exiting.
        if(policy!="waiter" && policy!="dguard" && hungry[i].load(std::memory_order_acquire)){
            ping_scheduler();
            pthread_mutex_lock(&gmu[i]);
            while(!grant[i]) pthread_cond_wait(&gcv[i],&gmu[i]);
            grant[i]=false;
            pthread_mutex_unlock(&gmu[i]);
            int dj=rnd_range(c.rng,0,dispatch_jitter_us); sleep_us(dj);
            pthread_mutex_lock(&forks[L]); pthread_mutex_lock(&forks[R]);
            uint64_t arr=req_ns[i].load(), st=now_ns();
            do_payload(c,L,R);
            pthread_mutex_unlock(&forks[R]); pthread_mutex_unlock(&forks[L]);
            active[i].store(0); hungry[i].store(0);
            if(arr>=measure_start_ns && arr<measure_end_ns){ c.metrics.waits_ms.push_back((st-arr)/1e6); c.metrics.meals_recorded++; }
            c.total_meals++; ping_scheduler();
        }
        alive.fetch_sub(1);
        ping_scheduler();
        return;
    }

    std::vector<int> solve_cycle(const std::vector<uint8_t>&cand, bool gf){
        // Exact independent set DP on cycle with lexicographic score.
        int n=N;
        auto weight=[&](int i)->Score{
            uint64_t arr=req_ns[i].load(std::memory_order_acquire);
            uint64_t age=(arr? now_ns()-arr:0);
            return {gf && owed[i].load()?1:0,1,age};
        };
        auto solve_path=[&](int a,int b)->std::pair<Score,std::vector<int>>{
            if(a>b) return {Score{}, {}};
            int len=b-a+1;
            // dp[k] is optimum over first k vertices of this path.
            std::vector<Score> dp(len+1);
            dp[0]=Score{};
            for(int k=1;k<=len;k++){
                int i=a+k-1;
                Score skip=dp[k-1];
                Score tk={-1000000000,-1000000000,0};
                if(cand[i]) tk=add((k>=2?dp[k-2]:Score{}),weight(i));
                dp[k]=(cand[i] && better(tk,skip))?tk:skip;
            }
            std::vector<int> sel;
            int k=len;
            while(k>0){
                int i=a+k-1;
                Score skip=dp[k-1];
                Score tk={-1000000000,-1000000000,0};
                if(cand[i]) tk=add((k>=2?dp[k-2]:Score{}),weight(i));
                if(cand[i] && better(tk,skip) && !better(skip,tk)){ sel.push_back(i); k-=2; }
                else k-=1;
            }
            return {dp[len],sel};
        };
        if(n==1){ if(cand[0]) return {0}; else return {}; }
        // Case A: exclude 0
        auto A=solve_path(1,n-1);
        // Case B: include 0 if candidate, force exclude 1,n-1
        Score Bs{-1000000000,-1000000000,0}; std::vector<int> Bsel;
        if(cand[0]){
            std::vector<uint8_t> c2=cand; if(n>1)c2[1]=0; if(n>2)c2[n-1]=0;
            auto B=solve_path(2,n-2);
            Bs=add(B.first,weight(0)); Bsel=B.second; Bsel.push_back(0);
        }
        return better(Bs,A.first)?Bsel:A.second;
    }

    void scheduler_loop(){
        uint64_t rng=seed ^ 0xd1b54a32d192ed03ULL;
        uint64_t phase=0, rot=0;
        std::vector<uint8_t> cand(N), selected_mask(N);
        while(alive.load(std::memory_order_acquire)>0){
            int jit=(sched_jitter_us>0)?(int)(splitmix64(rng)%(sched_jitter_us+1)):0;
            // Timed wait lets arrivals/completions wake us early but preserves a minimum scheduling quantum.
            {
                std::unique_lock<std::mutex> lk(sched_mu);
                sched_cv.wait_for(lk,std::chrono::microseconds(tick_us),[&]{return sched_ping;});
                sched_ping=false;
            }
            // Explicit scheduler-dispatch jitter: even an event wake is delayed by a bounded random amount.
            sleep_us((uint64_t)jit);
            scheduler_ticks++;
            uint64_t tnow=now_ns();
            std::fill(cand.begin(),cand.end(),0);
            std::fill(selected_mask.begin(),selected_mask.end(),0);

            // Advance GRR phase and create persistent obligations for hungry members of this rotation set.
            if(policy=="grr" || policy=="gfgrr"){
                int m=N/2;
                for(int k=0;k<m;k++){
                    int i=(int)((phase + 2ULL*(uint64_t)k) % (uint64_t)N);
                    if(hungry[i].load(std::memory_order_acquire) && !owed[i].exchange(1)){
                        owed_since_ns[i].store(tnow); guard_obligations++;
                    }
                }
                phase++;
            }

            auto eligible=[&](int i){
                if(!hungry[i].load(std::memory_order_acquire) || active[i].load(std::memory_order_acquire)) return false;
                int l=(i-1+N)%N, r=(i+1)%N;
                if(active[l].load(std::memory_order_acquire) || active[r].load(std::memory_order_acquire)) return false;
                return true;
            };

            if(policy=="fifo"){
                // Strict ticket-like FIFO: sort currently hungry, inactive requests by arrival. Grant consecutive head requests while conflict-free; stop at first blocked head.
                std::vector<int> q; q.reserve(N);
                for(int i=0;i<N;i++) if(hungry[i].load(std::memory_order_acquire) && !active[i].load(std::memory_order_acquire)) q.push_back(i);
                std::sort(q.begin(),q.end(),[&](int a,int b){ auto ra=req_ns[a].load(); auto rb=req_ns[b].load(); return ra==rb?a<b:ra<rb; });
                for(int i:q){
                    int l=(i-1+N)%N,r=(i+1)%N;
                    if(active[l].load()||active[r].load()||selected_mask[l]||selected_mask[r]) break;
                    selected_mask[i]=1;
                }
            } else if(policy=="grr"){
                for(int i=0;i<N;i++) if(owed[i].load() && eligible(i)) cand[i]=1;
                auto sel=solve_cycle(cand,true);
                for(int i:sel) selected_mask[i]=1;
            } else if(policy=="gfgrr"){
                // Fillers may not start next to a blocked owed request; this prevents opportunistic work from extending a guard obligation.
                for(int i=0;i<N;i++) if(eligible(i)){
                    int l=(i-1+N)%N,r=(i+1)%N;
                    if(!owed[i].load() && (owed[l].load() || owed[r].load())) continue;
                    cand[i]=1;
                }
                auto sel=solve_cycle(cand,true);
                for(int i:sel) selected_mask[i]=1;
            } else if(policy=="agemwis"){
                for(int i=0;i<N;i++) if(eligible(i)) cand[i]=1;
                auto sel=solve_cycle(cand,false);
                for(int i:sel) selected_mask[i]=1;
            } else if(policy=="rotgreedy"){
                // Rotating greedy independent set among eligible hungry vertices.
                for(int step=0;step<N;step++){
                    int i=(int)((rot+step)%N); int l=(i-1+N)%N,r=(i+1)%N;
                    if(eligible(i) && !selected_mask[l] && !selected_mask[r]) selected_mask[i]=1;
                }
                rot=(rot+1)%N;
            }

            // Native safety assertion: a central scheduler must never grant adjacent philosophers.
            if(policy!="waiter" && policy!="dguard"){
                for(int i=0;i<N;i++) if(selected_mask[i] && selected_mask[(i+1)%N]){
                    std::cerr << "scheduler conflict violation at " << i << " and " << ((i+1)%N) << "\n";
                    std::abort();
                }
            }
            // Grant selected workers. active is set before signaling so later grants/ticks see the reservation.
            for(int i=0;i<N;i++) if(selected_mask[i]){
                if(!hungry[i].load() || active[i].exchange(1)) continue;
                hungry[i].store(0,std::memory_order_release);
                if(owed[i].exchange(0)){
                    uint64_t os=owed_since_ns[i].load(); if(os){
                        uint64_t oa=tnow-os, cur=max_owed_age_ns.load();
                        while(oa>cur && !max_owed_age_ns.compare_exchange_weak(cur,oa)){}
                    }
                }
                pthread_mutex_lock(&gmu[i]); grant[i]=true; pthread_cond_signal(&gcv[i]); pthread_mutex_unlock(&gmu[i]);
            }
        }
    }

    bool run(){
        alloc(); alive=N;
        pthread_attr_t attr; pthread_attr_init(&attr);
        size_t stack_sz=256*1024; pthread_attr_setstacksize(&attr,stack_sz);
        for(int i=0;i<N;i++){
            ctx[i].b=this; ctx[i].id=i; ctx[i].rng=seed ^ (0x9e3779b97f4a7c15ULL*(uint64_t)(i+1));
            int rc=pthread_create(&threads[i],&attr,&Bench::worker_entry,&ctx[i]);
            if(rc!=0){ std::cerr<<"pthread_create failed at "<<i<<" rc="<<rc<<"\n"; alive=i; generation_stop=true; start_flag=true; for(int j=0;j<i;j++) pthread_join(threads[j],nullptr); pthread_attr_destroy(&attr); return false; }
        }
        pthread_attr_destroy(&attr);
        if(policy!="waiter" && policy!="dguard") scheduler=std::thread([this]{scheduler_loop();});
        uint64_t start=now_ns();
        start_flag.store(true,std::memory_order_release);
        sleep_us((uint64_t)(warmup_s*1e6));
        measure_start_ns=now_ns();
        measure_end_ns=measure_start_ns + (uint64_t)(measure_s*1e9);
        sleep_us((uint64_t)(measure_s*1e6));
        generation_stop.store(true,std::memory_order_release);
        ping_scheduler();
        for(int i=0;i<N;i++) pthread_join(threads[i],nullptr);
        if(policy!="waiter" && policy!="dguard"){ ping_scheduler(); if(scheduler.joinable()) scheduler.join(); }
        return true;
    }

    void print_result(){
        std::vector<double> waits; waits.reserve(1000000);
        std::vector<double> counts(N); uint64_t meals=0, total=0; int zeros=0; uint64_t minm=UINT64_MAX,maxm=0;
        for(int i=0;i<N;i++){
            waits.insert(waits.end(),ctx[i].metrics.waits_ms.begin(),ctx[i].metrics.waits_ms.end());
            meals+=ctx[i].metrics.meals_recorded; total+=ctx[i].total_meals; counts[i]=(double)ctx[i].metrics.meals_recorded;
            minm=std::min(minm,ctx[i].metrics.meals_recorded); maxm=std::max(maxm,ctx[i].metrics.meals_recorded); if(ctx[i].metrics.meals_recorded==0) zeros++;
        }
        std::sort(waits.begin(),waits.end());
        auto pct=[&](double p){ if(waits.empty()) return 0.0; double x=p*(waits.size()-1); size_t a=(size_t)std::floor(x), b=(size_t)std::ceil(x); double f=x-a; return waits[a]*(1-f)+waits[b]*f; };
        double mean=waits.empty()?0.0:std::accumulate(waits.begin(),waits.end(),0.0)/waits.size();
        double sum=std::accumulate(counts.begin(),counts.end(),0.0), sq=0; for(double x:counts)sq+=x*x;
        double jain=(sq>0)?(sum*sum/(N*sq)):0;
        double thr=meals/measure_s;
        std::cout<<std::fixed<<std::setprecision(6)
                 <<policy<<","<<N<<","<<workload<<","<<app<<","<<seed<<","<<measure_s<<","<<tick_us<<","<<sched_jitter_us<<","<<dispatch_jitter_us<<","<<meals<<","<<thr<<","<<mean<<","<<pct(.50)<<","<<pct(.95)<<","<<pct(.99)<<","<<(waits.empty()?0.0:waits.back())<<","<<jain<<","<<(minm==UINT64_MAX?0:minm)<<","<<maxm<<","<<zeros<<","<<scheduler_ticks.load()<<","<<guard_obligations.load()<<","<<(max_owed_age_ns.load()/1e6)<<"\n";
    }
};

int main(int argc,char**argv){
    if(argc<7){ std::cerr<<"usage: phase2_native POLICY N WORKLOAD APP SEED MEASURE_S [TICK_US] [SCHED_JITTER_US] [DISPATCH_JITTER_US] [EPOCH_US]\n"; return 2; }
    std::string pol=argv[1]; int N=atoi(argv[2]); std::string wl=argv[3]; std::string app=argv[4]; uint64_t seed=strtoull(argv[5],nullptr,10); double ms=atof(argv[6]);
    Bench b(N); b.policy=pol; b.workload=wl; b.app=app; b.seed=seed; b.measure_s=ms; b.warmup_s=std::min(0.12,ms*0.25);
    b.tick_us=(argc>7)?atoi(argv[7]):200; b.sched_jitter_us=(argc>8)?atoi(argv[8]):120; b.dispatch_jitter_us=(argc>9)?atoi(argv[9]):80; b.epoch_us=(argc>10)?atoi(argv[10]):1200;
    b.eat_lo_us=400; b.eat_hi_us=2600; b.think_lo_us=600; b.think_hi_us=5000;
    if(N<3){std::cerr<<"N>=3\n";return 2;}
    if(pol!="grr"&&pol!="gfgrr"&&pol!="agemwis"&&pol!="rotgreedy"&&pol!="waiter"&&pol!="fifo"&&pol!="dguard"){std::cerr<<"bad policy\n";return 2;}
    if(app!="sleep"&&app!="ledger"&&app!="kv"){std::cerr<<"bad app\n";return 2;}
    std::cout<<"policy,N,workload,app,seed,measure_s,tick_us,sched_jitter_us,dispatch_jitter_us,meals,throughput_s,mean_wait_ms,p50_wait_ms,p95_wait_ms,p99_wait_ms,max_wait_ms,jain,min_meals,max_meals,zero_workers,scheduler_ticks,guard_obligations,max_owed_age_ms\n";
    if(!b.run()) return 3; b.print_result();
    if(app=="ledger"){
        long long sum=std::accumulate(b.balances.begin(),b.balances.end(),0LL);
        long long expected=(long long)N*1000000LL;
        if(sum!=expected){ std::cerr<<"ledger invariant failed: "<<sum<<" != "<<expected<<"\n"; return 4; }
    }
    return 0;
}
