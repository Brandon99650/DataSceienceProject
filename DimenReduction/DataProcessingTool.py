#load iris Data
from sklearn.datasets import load_iris
import numpy as np

def standardization(data:np.ndarray):

    """
    ci = (ci-ave)/sigma
    """
    col_std = data.std(axis = 0)
    return (data - np.mean(data, axis=0) )/col_std


def iris_data(std=True): 
    data = load_iris()
    format_data = {
        "features":data.data, 
        "labels":data.target
    }
    if std:
        #print("std")
        format_data['features'] = standardization(format_data['features'])
    return format_data