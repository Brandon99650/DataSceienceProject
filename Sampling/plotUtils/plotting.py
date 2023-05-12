import seaborn as sns
import matplotlib.pyplot as plt
import os 
import numpy as np
from tqdm import tqdm

def plot_xy(x, y, method, comment):
    """
    curve
    """
    plt.figure(dpi=90)
    for yi in y:
        plt.plot(x,yi)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    imgpath = os.path.join(".", "vis", f"{method}")
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)
    imgpath = os.path.join(imgpath, f"{comment}.png")
    plt.savefig(imgpath)
    print(f"img at {imgpath}")
    plt.close()

def plot_distribution(x , sample_d, samples, target, method, comment="default"):
    ax = sns.histplot(data=samples,stat="density", kde = True, bins=20, color="r", label = "samples")
    ax.plot(x, target(x), color="g", label="f(x)")
    if sample_d is not None:
        ax.plot(x, sample_d, color = "b", label="sampler")
    plt.legend()
    imgpath = os.path.join(".", "vis", f"{method}")
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)
    imgpath = os.path.join(imgpath, f"{comment}.png")
    plt.savefig(imgpath)
    print(f"img at {imgpath}")
    plt.close()

def plot2D(pos, method, mu, sigma, target, comment="default"):
    

    """
    2D Bi Guassian pdf & sampling coordinate
    """
    xlist = pos[: ,0]
    ylist = pos[:, 1] 
    xmax = np.max(xlist)
    xmin = np.min(xlist)
    ymin = np.min(ylist)
    ymax = np.max(ylist)
    print(f"{pos.shape} {xlist.shape} {ylist.shape}")
    print(f"{xmin} {xmax}, {ymin} {ymax}")
    c = input()
    f = plt.figure()
    ax1 = f.add_subplot(211)
    for i, r in enumerate(tqdm(pos)):
            ax1.scatter(r[0],r[1], color = "b")
    
    ax2 = f.add_subplot(212)
    x, y = np.mgrid[xmin:xmax:(xmax-xmin)/100, ymin:ymax:(ymax-ymin)/100]
    ax2.contourf(x, y, target(np.dstack((x, y)), mu, sigma), 20)
    
    imgpath = os.path.join(".", "vis", f"{method}")
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)
    imgpath = os.path.join(imgpath, f"{comment}.png")
    plt.savefig(imgpath)
    print(f"img at {imgpath}")
    plt.close()

