#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
using Clock=std::chrono::steady_clock;
static inline uint64_t now_ns(){return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
static inline void sleep_us(uint64_t x){if(x) std::this_thread::sleep_for(std::chrono::microseconds(x));}
static inline uint64_t sm64(uint64_t &x){uint64_t z=(x+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static inline double u01(uint64_t&s){return (sm64(s)>>11)*(1.0/9007199254740992.0);}
constexpr int MAXW=8192;
struct Metric{uint64_t meals=0,total=0; int nw=0; double waits[MAXW];};
struct SharedHead{
 int N; int epoch_us; int dispatch_jitter_us; int workload; int app; uint64_t start_ns, measure_start_ns, measure_end_ns; volatile int stop;
 sem_t room;
};
static inline bool guard_member(int N,int i,uint64_t epoch){int m=N/2; if(N&1){uint64_t d=((uint64_t)i+N-(epoch%(uint64_t)N))%(uint64_t)N; uint64_t k=(d*(uint64_t)((N+1)/2))%(uint64_t)N; return k<(uint64_t)m;} uint64_t d=((uint64_t)i+N-(epoch%(uint64_t)N))%(uint64_t)N; return d%2==0 && d/2<(uint64_t)m;}
static inline int rnd(uint64_t&r,int lo,int hi){return lo+(int)(sm64(r)%(uint64_t)(hi-lo+1));}
static inline int sample_think(uint64_t&r,int workload){if(workload==0)return 0;if(u01(r)<0.15)return rnd(r,5000,15000);return rnd(r,600,5000);}
static inline int sample_eat(uint64_t&r){double a=u01(r),b=u01(r),z=sqrt(-2*log(std::max(a,1e-12)))*cos(6.28318530718*b);double v=1500*exp(0.45*z);return (int)std::max(400.0,std::min(2600.0,v));}
int main(int argc,char**argv){
 if(argc<7){std::cerr<<"usage: phase2_process POLICY(dguard|waiter|ordered) N WORKLOAD(saturated|bursty) APP(sleep|ledger|kv) SEED MEASURE_S [EPOCH_US] [DISPATCH_JITTER_US]\n";return 2;}
 std::string pol=argv[1], wl=argv[3], app=argv[4]; int N=atoi(argv[2]); uint64_t seed=strtoull(argv[5],nullptr,10); double measure_s=atof(argv[6]); int epoch_us=argc>7?atoi(argv[7]):1200, djmax=argc>8?atoi(argv[8]):80;
 if(N<3 || (pol!="dguard"&&pol!="waiter"&&pol!="ordered")){std::cerr<<"bad args\n";return 2;}
 size_t sz=sizeof(SharedHead)+N*sizeof(pthread_mutex_t)+N*sizeof(uint8_t)+N*sizeof(Metric)+N*sizeof(long long)+N*64*sizeof(uint64_t);
 char* mem=(char*)mmap(nullptr,sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0); if(mem==MAP_FAILED){perror("mmap");return 3;} memset(mem,0,sz);
 auto* H=(SharedHead*)mem; char* p=mem+sizeof(SharedHead); auto* forks=(pthread_mutex_t*)p;p+=N*sizeof(pthread_mutex_t); auto* owed=(uint8_t*)p;p+=N; auto* M=(Metric*)p;p+=N*sizeof(Metric); auto* bal=(long long*)p;p+=N*sizeof(long long); auto* kv=(uint64_t*)p;
 H->N=N;H->epoch_us=epoch_us;H->dispatch_jitter_us=djmax;H->workload=(wl=="saturated"?0:1);H->app=(app=="sleep"?0:(app=="ledger"?1:2));H->stop=0;
 pthread_mutexattr_t ma;pthread_mutexattr_init(&ma);pthread_mutexattr_setpshared(&ma,PTHREAD_PROCESS_SHARED);
 for(int i=0;i<N;i++){pthread_mutex_init(&forks[i],&ma);bal[i]=1000000;for(int j=0;j<64;j++)kv[i*64+j]=(uint64_t)(i+1)*0x9e3779b97f4a7c15ULL+j;} pthread_mutexattr_destroy(&ma);
 sem_init(&H->room,1,(unsigned)std::max(1,N-1));
 std::vector<pid_t> kids; kids.reserve(N);
 H->start_ns=now_ns()+50000000ULL; H->measure_start_ns=H->start_ns+50000000ULL; H->measure_end_ns=H->measure_start_ns+(uint64_t)(measure_s*1e9);
 for(int i=0;i<N;i++){
  pid_t pid=fork(); if(pid<0){perror("fork");break;} if(pid==0){
   uint64_t rng=seed^(0x9e3779b97f4a7c15ULL*(uint64_t)(i+1)); int L=i,R=(i+1)%N; while(now_ns()<H->start_ns) std::this_thread::yield(); sleep_us(sample_think(rng,H->workload));
   while(now_ns()<H->measure_end_ns){
    uint64_t arr=now_ns(); uint64_t st=0;
    if(pol=="waiter"){
      sem_wait(&H->room); pthread_mutex_lock(&forks[L]); sleep_us(rnd(rng,0,H->dispatch_jitter_us)); pthread_mutex_lock(&forks[R]); st=now_ns();
    } else if(pol=="ordered"){
      int a=std::min(L,R),b=std::max(L,R); pthread_mutex_lock(&forks[a]); sleep_us(rnd(rng,0,H->dispatch_jitter_us)); pthread_mutex_lock(&forks[b]); st=now_ns();
    } else {
      while(true){uint64_t t=now_ns();uint64_t ep=(t-H->start_ns)/(uint64_t)std::max(1,H->epoch_us*1000);if(guard_member(N,i,ep)) __atomic_store_n(&owed[i],1,__ATOMIC_RELEASE);int pl=(i-1+N)%N,pr=(i+1)%N;bool mine=__atomic_load_n(&owed[i],__ATOMIC_ACQUIRE);if(!mine&&(__atomic_load_n(&owed[pl],__ATOMIC_ACQUIRE)||__atomic_load_n(&owed[pr],__ATOMIC_ACQUIRE))){sleep_us(20);continue;}int a=std::min(L,R),b=std::max(L,R);if(pthread_mutex_trylock(&forks[a])!=0){sleep_us(mine?5:20);continue;}if(pthread_mutex_trylock(&forks[b])!=0){pthread_mutex_unlock(&forks[a]);sleep_us(mine?5:20);continue;}if(!mine&&(__atomic_load_n(&owed[pl],__ATOMIC_ACQUIRE)||__atomic_load_n(&owed[pr],__ATOMIC_ACQUIRE))){pthread_mutex_unlock(&forks[b]);pthread_mutex_unlock(&forks[a]);sleep_us(20);continue;}st=now_ns();sleep_us(rnd(rng,0,H->dispatch_jitter_us));break;}
    }
    if(H->app==0) sleep_us(sample_eat(rng));
    else if(H->app==1){int reps=32+(int)(sm64(rng)%97);long long d=0;uint64_t x=rng^(uint64_t)(i+1);for(int k=0;k<reps;k++){x^=x<<13;x^=x>>7;x^=x<<17;d+=(x&7)+1;d-=((x>>3)&7)+1;} long long amt=(sm64(rng)%7)+1;bal[L]-=amt;bal[R]+=amt;bal[L]+=d;bal[R]-=d;}
    else {int reps=128+(int)(sm64(rng)%385);uint64_t x=rng^(uint64_t)(i+17);for(int k=0;k<reps;k++){x^=x<<13;x^=x>>7;x^=x<<17;auto&a=kv[L*64+((x>>1)&63)];auto&b=kv[R*64+((x>>9)&63)];a=(a^x)*0x9e3779b185ebca87ULL+(b>>1);b=(b+(x^a))*0xc2b2ae3d27d4eb4fULL;}}
    if(pol=="waiter"){pthread_mutex_unlock(&forks[R]);pthread_mutex_unlock(&forks[L]);sem_post(&H->room);} else {int a=std::min(L,R),b=std::max(L,R);pthread_mutex_unlock(&forks[b]);pthread_mutex_unlock(&forks[a]); if(pol=="dguard") __atomic_store_n(&owed[i],0,__ATOMIC_RELEASE);}
    uint64_t end=now_ns(); if(arr>=H->measure_start_ns&&arr<H->measure_end_ns){if(M[i].nw<MAXW)M[i].waits[M[i].nw++]=(st-arr)/1e6;M[i].meals++;}M[i].total++; sleep_us(sample_think(rng,H->workload));
   }
   _exit(0);
  } else kids.push_back(pid);
 }
 for(pid_t k:kids) waitpid(k,nullptr,0);
 std::vector<double>w;std::vector<double>counts(N);uint64_t meals=0;int zeros=0;for(int i=0;i<N;i++){meals+=M[i].meals;counts[i]=M[i].meals;if(M[i].meals==0)zeros++;for(int j=0;j<M[i].nw;j++)w.push_back(M[i].waits[j]);}
 sort(w.begin(),w.end());auto pct=[&](double q){if(w.empty())return 0.0;double x=q*(w.size()-1);size_t a=floor(x),b=ceil(x);double f=x-a;return w[a]*(1-f)+w[b]*f;};double mean=w.empty()?0:accumulate(w.begin(),w.end(),0.0)/w.size();double sum=accumulate(counts.begin(),counts.end(),0.0),sq=0;for(double x:counts)sq+=x*x;double jain=sq?sum*sum/(N*sq):0; long long balsum=0;for(int i=0;i<N;i++)balsum+=bal[i];
 std::cout<<"mode,policy,N,workload,app,seed,measure_s,epoch_us,meals,throughput_s,mean_wait_ms,p95_wait_ms,p99_wait_ms,max_wait_ms,jain,zero_workers,ledger_invariant\n";
 std::cout<<std::fixed<<std::setprecision(6)<<"process,"<<pol<<","<<N<<","<<wl<<","<<app<<","<<seed<<","<<measure_s<<","<<epoch_us<<","<<meals<<","<<meals/measure_s<<","<<mean<<","<<pct(.95)<<","<<pct(.99)<<","<<(w.empty()?0:w.back())<<","<<jain<<","<<zeros<<","<<(app=="ledger"?(balsum==(long long)N*1000000LL):1)<<"\n";
 for(int i=0;i<N;i++)pthread_mutex_destroy(&forks[i]);sem_destroy(&H->room);munmap(mem,sz);return 0;
}
