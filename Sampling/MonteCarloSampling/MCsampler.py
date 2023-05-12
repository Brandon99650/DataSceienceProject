import scipy.stats as st
import numpy as np
from tqdm import tqdm

class MCsampling():
    def __init__(self, target) -> None:
        self.target = target

    def sampling(self,samplingN:int, interval:list, method:str, **kwargs):
        if method == "RejectSampling":
            return self.__reject_sampling(samplingN, interval)
        elif method == "MCMC":
            return self.__MCMC(samplingN, interval)
        elif method == "Gibbs":
            #print(kwargs)
            return self.__Gibbs(samplingN, mu=kwargs['mu'],sigma=kwargs['sigma'])
    
    def build_interval(self, N , M,m):
        step= (M-m)/N
        #print(step)
        return np.arange(m,M,step)

    def __reject_sampling(self, N:int, interval):
        vmin = interval[0]
        vmax = interval[1]
        #print(f"from:{vmin} to {vmax}", end=", step=")
        x = self.build_interval(N, vmax, vmin)
        samples = []
        mean = np.mean(interval)
        k = max(self.target(x)/st.norm.pdf(x,loc=mean))
        pbar = tqdm(range(0,N))
        itera = 0
        while(len(samples)< N):
            itera += 1
            x = np.random.uniform(vmin,vmax)
            y = np.random.uniform(vmin, k*st.norm.pdf(x,loc=mean))
            if y <= self.target(x):
                samples.append(x)
                pbar.update()          
        pbar.close()
        print(f"accept rate : {N/itera}")
        return {'accept rate':N/itera, 'sample':samples}

    def __MCMC_burnin(self, interval, burnin):
        all_data = []
        previous = np.random.uniform(interval[0], interval[1])
        itera = 0
        pbar = tqdm(range(0,burnin))
        while(itera < burnin):
            x = np.random.normal(loc=previous)
            if x > interval[1] or x < interval[0]:
                continue
            itera += 1
            pbar.update()
            alpha = self.target(x)/self.target(previous)
            if alpha >= 1.0:
                previous = x
                all_data.append(x)
            else :
                markovC = np.random.choice([1,0], p=[alpha, 1-alpha])
                if markovC == 1:
                    #accept
                    all_data.append(x)
                    previous = x
        return previous, all_data
    
    def __MCMC(self, N:int, interval, burinin=False):
        #Metropolis–Hasting Algorithm
        
        previous = np.random.uniform(interval[0], interval[1])
        itera = 0
        all_data=None
        samples = []
        #burn-in stage:
        if not burinin:
            print("Burn-in")
            previous, all_data = self.__MCMC_burnin(interval, 1000)
            print("Burn-in over")
            samples.append(previous)
        
        
        pbar = tqdm(range(0,N))
        while (len(samples)< N) :
            x = np.random.normal(loc=previous)
            if x > interval[1] or x < interval[0]:
                continue 
            itera += 1
            # g is a symmetric (gusssian)
            alpha = self.target(x)/self.target(previous)
            if alpha  >= 1.0:
                samples.append(x)
                previous = x
                all_data.append(x)
                pbar.update()
            else :
                markovC = np.random.choice([1,0], p=[alpha, 1-alpha])
                if markovC == 1:
                    #accept
                    samples.append(x)
                    all_data.append(x)
                    previous = x
                    pbar.update()
    
        pbar.close()
        print(f"accept rate {N/itera}")
        return {'accept rate':N/itera, 'sample':samples, 'alldata':all_data}

    def __Gibbs(self, N:int, mu, sigma):
        #print(interval)
        print(f"condsigma : {sigma[0][0]-( (sigma[0][1]**2)/sigma[1][1] )}, {sigma[1][1]-( (sigma[0][1]**2)/sigma[0][0] )}")
        x0 = np.random.uniform(-1.0,1.0)
        x1 = np.random.uniform(-1.0, 1.0)
        ret= np.zeros((N,2))
        i = 0
        pbar = tqdm(range(0, N))
        for i in pbar:
            x0 = np.random.normal(
                loc = mu[0]+(sigma[0][1]/sigma[1][1])*(x1 - mu[1]),
                scale = sigma[0][0]-( (sigma[0][1]**2)/sigma[1][1] )
            )
            x1 = np.random.normal(
                loc = mu[1]+(sigma[0][1]/sigma[0][0])*(x0 - mu[0]),
                scale = sigma[1][1]-( (sigma[0][1]**2)/sigma[0][0] )
            )
            
            ret[i][0]=x0
            ret[i][1]=x1
            i += 1
            pbar.update()
        pbar.close()
        return ret


def beta82(x):
    return st.beta.pdf(x, a=8, b=2) # + st.beta.pdf(x, a = 5, b= 5)

def dgmma1(x):
    return st.dgamma.pdf(x, a=1)

def Bigussain(x, mu, sigma):
    return st.multivariate_normal(mu, sigma).pdf(x)
