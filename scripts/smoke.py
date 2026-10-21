import sys, time, statistics as st
sys.path.insert(0, '.')
from figgie.engine import play
from figgie.agents.cs224r import RandomAgent, FairValue, Conservative, MarketMaker, Bayesian
t=time.time(); N=int(sys.argv[1]) if len(sys.argv)>1 else 50
names=["Bayesian","FairValue","MarketMaker","Conservative"]
P=[[] for _ in names]; W=[0]*4
for g in range(N):
    ag=[Bayesian(seed=g),FairValue(seed=g),MarketMaker(seed=g),Conservative(seed=g)]
    r=play(ag, seed=g)
    for i in range(4): P[i].append(r.profits[i])
    best=max(r.profits); 
    for i in range(4): W[i]+= r.profits[i]==best
    assert abs(sum(r.profits))<1e-6
print(f"{time.time()-t:.1f}s")
for n,p,w in zip(names,P,W): print(n, round(st.mean(p),1), round(1.96*st.stdev(p)/len(p)**.5,1), w/N)
