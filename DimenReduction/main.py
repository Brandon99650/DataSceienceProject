from DataProcessingTool import iris_data
from DimReduction import LDA
from DimReduction import PCA
from DimReduction import RandomProjection
from DimReduction import get_class_info
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
from sklearn.metrics import pairwise_distances


def Plot_2D(result:dict):
    print("Plot")
    fig = plt.figure(figsize=(8,6))
    color = ['red','blue','green']
    #color = ['black','black','black']
    
    axidx = 1
    for method, r in result.items():
        for std, ret in r.items():
            ax = fig.add_subplot(2, 2, axidx)
            coloridx = 0
            class_order = []
            for c,v in ret.items():
                class_order.append(c)
                ax.scatter(v['x'],v['y'],color = color[coloridx])
                coloridx += 1
            ax.legend(class_order)
            ax.title.set_text(method+" : Data "+ std)
            axidx += 2
        axidx = 2

    fig.tight_layout()
    plt.show()

def reduction(result:dict,method:str,model,datastd=False):

    if method not in result:
        result[method] = {}

    data = iris_data(std=datastd)
    is_std = "Standardization"
    if not datastd:
        is_std = "Not_"+ is_std
    
    data_info = get_class_info(data)
    reducutor = model(outpath=(method + "_"+is_std))
    to_dim = 2
    y = reducutor.get_transition(data,to_dim,data_info,save=True)
    result[method][is_std] = {}
    for c,info in data_info.items():
        if c == 'total_mean' or c == 'total_class_num':
            continue
        result[method][is_std][c] = {} 
        class_range = info['class range']
        result[method][is_std][c]['x'] = y[class_range[0]:class_range[1]][:,0]
        result[method][is_std][c]['y'] = y[class_range[0]:class_range[1]][:,1]

def pca_and_lda_cmp():
    result = {}
    reduction(result,"PCA",PCA,datastd=True)
    reduction(result,"PCA",PCA,datastd=False)
    reduction(result,"LDA",LDA,datastd=True)
    reduction(result,"LDA",LDA,datastd=False)
    Plot_2D(result)

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def random_proj():
    data = fetch_20newsgroups_vectorized().data[:500]
    R = RandomProjection()
    testing_dim = [300,1000,10000]

    origin_pairdist = upper_tri_masking(
        pairwise_distances(data.A,n_jobs=-1)
    )

    
    proj_dim = {}

    for d in testing_dim:
        proj_dim[d] = {}
        ret = R.get_transition(data, to_dim=d)
        proj_paridist = upper_tri_masking(
            pairwise_distances(ret,n_jobs=-1)
        )
        proj_dim[d]['pairwise_distance'] = proj_paridist
        proj_dim[d]['rate'] = proj_paridist/origin_pairdist
        proj_dim[d]['mean_rate'] =np.mean(proj_dim[d]['rate'])
        print()

    fig = plt.figure(figsize=(10,4))
    for i, d in enumerate(testing_dim):
        ax1 = fig.add_subplot(1, 3, i+1)
        ax1.hist(proj_dim[d]['rate'])
        ax1.title.set_text(str(d)+", Mean = "+str(proj_dim[d]['mean_rate']))
        ax1.set_xlabel('rate of pairwise distance')
        ax1.set_ylabel('num') 
    
    fig.tight_layout()
    plt.show()

def ru():
    data = fetch_20newsgroups_vectorized().data[:500]
    R = RandomProjection()
    ret = R.get_transition(data, to_dim=1000)

    pairdist = pairwise_distances(data.A,n_jobs=-1)
    origin_pairdist = upper_tri_masking(pairdist)

    projdist = pairwise_distances(ret[:,:100],n_jobs=-1)
    proj_dist = ( upper_tri_masking(projdist) )

    rate_array = proj_dist/origin_pairdist
    print(np.mean(rate_array))


if __name__ == "__main__":
    pca_and_lda_cmp()
    #random_proj()