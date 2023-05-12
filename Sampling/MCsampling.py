import pandas as pd
import scipy.stats as st
import numpy as np
import math
import os 
import time
from MonteCarloSampling.MCsampler import MCsampling, beta82, dgmma1, Bigussain
from plotUtils.plotting import plot_xy, plot_distribution, plot2D


def SaveNp(a:np.ndarray, folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)
    p = os.path.join(folder, f"{name}.npy")
    np.save(f'{p}', np.array(a))
    print(f"saving result at : {p}")


def reject(interval, N, target, targetname=""):

    print("reject sampling")
    sampler = MCsampling(target)
    x = sampler.build_interval(m=interval[0], M=interval[1], N=N)
    method = "RejectSampling"

    k = max(target(x)/st.norm.pdf(x,loc=np.mean(interval)))
    sample_d = k*st.norm.pdf(x,loc=np.mean(interval))
    
    start = time.time()
    r = sampler.sampling(samplingN=N, interval=interval, method=method)
    end = time.time()
    
    SaveNp(
        a=r['sample'], 
        folder=os.path.join(".","sampling",method), 
        name=targetname
    )
    plot_distribution(
        x = x, sample_d=sample_d, 
        target=target, samples=r['sample'], 
        method=method, comment = targetname
    )
    
    return {'time': end-start, 'accrate': r['accept rate']}

def MCMC(interval, N, target, targetname=""):

    print("Metropolis-Hasting")
    sampler = MCsampling(target)
    x = sampler.build_interval(m=interval[0], M=interval[1], N=N)
    
    start = time.time()
    r = sampler.sampling(samplingN=N, interval=interval, method="MCMC")
    end = time.time()

    SaveNp(
        a=r, folder=os.path.join(".","sampling","MCMC"),
        name = targetname
    )

    plot_distribution(
        x = x, sample_d=None,
        target=target, samples=r['sample'], 
        method="MCMC", comment = targetname
    )

    plot_xy(
        x=list(i for i in range( math.ceil(len(r['alldata'])/100) ) ), 
        y=[r['alldata'][::100]], method="MCMC", comment = f"convergeTest_{targetname}"
    )

    return {'time': end-start, 'accrate': r['accept rate']}

def Gibbs2d(N):
    args = {"mu":[0.0, 0.0],"sigma":[[2.0, 0.3], [0.3, 0.5]]}
    sampler = MCsampling(None)
    
    start = time.time()
    r = sampler.sampling(samplingN=N, interval=None, method="Gibbs", mu = args['mu'], sigma=args['sigma'])
    end = time.time()
    print(end-start)
    
    SaveNp(
        a=r, folder=os.path.join(".","sampling","Gibbs"),
        name="2dgussian"
    )
    
    plot2D(
        r, method="Gibbs", 
        mu=args['mu'], sigma=args['sigma'],target=Bigussain, 
        comment="gussain2d"
    )

def main(Numberlist=[10000],experimenttime=1, targetF={}):
    rtime = []
    raccrate = []
    mctime = []
    mcaccrate = []

    interval= [0.0, 1.0]
    result = pd.DataFrame()
    if os.path.exists("resulttest.csv"):
        result = pd.read_csv("resulttest.csv")

    for k,v in targetF.items():
        print(k)
        for i in range(experimenttime):
            print(f"experiment: {i}")
            for N in Numberlist: 
                print(N)
                r = reject(interval, N, v, targetname=k )
                mc = MCMC(interval, N, v ,targetname= k)
                rtime.append(r['time'])
                raccrate.append(r['accrate'])
                mctime.append(mc['time'])
                mcaccrate.append(mc['accrate'])
        
                result = pd.concat([
                    result,     
                    pd.DataFrame({
                        'target': [k]*2,
                        'sampling number':[N]*2,
                        'method':['reject', 'MCMC'],
                        'time' : [np.mean(rtime), np.mean(mctime)],
                        'accept Rate':[np.mean(raccrate), np.mean(mcaccrate)]
                    })
                ])
    
    result.to_csv("resulttest.csv", index=False)


if __name__ == "__main__":
   
    targetF = {"beta82":beta82} 
    """
    "doubleG1":dgmma1
    """
    #main(targetF=targetF)
    Gibbs2d(10000)
