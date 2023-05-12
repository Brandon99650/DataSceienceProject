
import numpy as np
from tqdm import tqdm
import os

def testdata():
    from DataProcessingTool import standardization as stdza
    data = {}
    data['features']=np.array([[1,2,3],[4,5,6],[7,8,9]])
    data['labels']=np.array([0,0,1])
    data['features'] = stdza(data['features'])
    return data

def get_class_info(data:dict)->dict:
    """
        Will caculate the mean vectors 
        and number of each classes
    """

    u = np.unique(data['labels'], return_index=True,return_counts=True)
    """
        u[0]: a 1*n row-ndarray of each unique value
        u[1]: a 1*n row-ndarray of idx of each unique value
        u[2]: a 1*n row-ndarray of idx count of each unique value
    """

    class_info = {}
    total_sum = np.zeros( (1,data['features'].shape[1]) )
    class_info['total_class_num'] = u[0].shape[0]
    for idx, group in enumerate(u[0]):
            
            class_info[group]={}
            class_info[group]['num']=u[2][idx]
            this_class_range = (u[1][idx],u[1][idx]+u[2][idx])
            class_info[group]['class range'] = this_class_range
           
            sum = np.sum(
                data['features'][this_class_range[0]:this_class_range[1]],
                axis=0
            )
            
            total_sum += sum
            class_info[group]['meanvec'] = sum*(1/u[2][idx])
    
    total_mean = total_sum*(1/data['features'].shape[0])
    
    class_info['total_mean'] = total_mean
    return class_info

def ordered_eigen(arr:np.ndarray,num=-1)->dict:
    (eigenvalues, eignevectors)=np.linalg.eig(arr)
    Num = num
    if Num == -1:
        """
        all values 
        """
        Num =  eigenvalues.shape[0]

    descending_eigenvalues = (-np.sort(-eigenvalues))[:num]
    descending_order_eigenvectors = eignevectors[:,np.argsort(-eigenvalues)][:, :num]
    ret = {}
    ret['eigenvalues']= descending_eigenvalues
    ret['eigenvectors'] = descending_order_eigenvectors

    return ret


class __DimReduction:
    def __init__(self) -> None:
        
        self.output_path = os.getcwd()
        self.omega = np.array([])

    def get_transition(self, data:dict,to_dim:int, class_info=None,save=False) -> np.ndarray:
        pass

    def reduce(self, features:np.ndarray) -> np.ndarray:
        
        return np.dot(features,self.omega)

    def _save_pattern(self):
        pass

class RandomProjection(__DimReduction):
    def __init__(self) -> None:
        super().__init__()
    
    def get_transition(self, data, to_dim:int, class_info=None, save=False) -> np.ndarray:
        return self.reduce( data*((3/to_dim)**(1/2)), to_dim)
    
    def reduce(self, features,to_dim) :
        r = np.zeros((features.shape[0],to_dim))
        n_nonzero_samples =  (features.shape[1]//3)
        for d in tqdm(range(to_dim)):
            nonzero_idx = (np.random.choice(features.shape[1],size=n_nonzero_samples,replace=False)).tolist()
            #print(nonzero_idx)
            onesidx = nonzero_idx[:(n_nonzero_samples//2)+1]
            ones = features[:, onesidx]
            negative_onesidx = nonzero_idx[(n_nonzero_samples//2)+1:]
            negative_ones = features[:,negative_onesidx]
            
            #c = input()
            dotresult = ones.sum(axis=1) - negative_ones.sum(axis=1)
            #print(dotresult.shape)
            r[:,d] = dotresult.reshape(features.shape[0],)
    
        return r



class PCA(__DimReduction):
    
    def __init__(self, outpath="PCA_Pattern") -> None:
        super().__init__()
        self.output_path = os.path.join(self.output_path,outpath) 
    
    def get_transition(self, data:dict,to_dim:int, class_info=None,save=False) -> np.ndarray:
        
        cov = np.cov(data['features'].T)

        eigen = ordered_eigen(arr=cov, num=to_dim)

        self.omega = eigen['eigenvectors']

        result = self.reduce(data['features'])

        if save:
            self._save_pattern(cov,result,to_dim)

        return result 
    
    def _save_pattern(self,cov,result,to_dim):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        print(f"These matrices will be saved at {self.output_path}")  
        
        np.save(os.path.join(self.output_path,"Cov.npy"),cov)
        np.savetxt(os.path.join(self.output_path,"Cov.txt"),cov)

        np.save(os.path.join(self.output_path,"Omega.npy"),self.omega)
        np.savetxt(os.path.join(self.output_path,"Omega.txt"),self.omega)
        
        np.save(os.path.join(self.output_path,"result_with_dim"+str(to_dim)+".npy"),result)
        np.savetxt(os.path.join(self.output_path,"result_with_dim"+str(to_dim)+".txt"),result)
    


class LDA(__DimReduction):

    def __init__(self, outpath="LDA_Pattern") -> None:
        super().__init__()
        print("LDA")
        self.output_path = os.path.join(self.output_path,outpath) 
        
    def get_transition(self, data:dict,to_dim:int, class_info=None,save=False) -> np.ndarray:
        print("Reduce dim")
        class_info = class_info
        max_reduction = class_info['total_class_num']-1
        
        if to_dim > max_reduction:
            print("No way")
            return np.array([])

        if class_info is None:
            class_info = get_class_info(data)
        Sb = self.__between_scattar(data,class_info)
        Sw = self.__within_scattar(data,class_info)
        eigen = ordered_eigen(
            arr = np.dot(np.linalg.inv(Sw),Sb), 
            num=to_dim
        )

        self.omega=eigen['eigenvectors']
        result = self.reduce(data['features'])
        if save:
            
            self._save_patterns(Sb,Sw,result,to_dim)
        return result

    def __within_scattar(self, data:dict, class_info:dict) -> np.ndarray:
        
        """
        SUM_{i=0}^c:    SUM_{j in ci}:  (xj-uj)(xj-uj)^T
        (diff)(diff^T)
        """

        mean_matrix = np.zeros(data['features'].shape)
        for c,info in class_info.items():
            if c == 'total_mean' or c == 'total_class_num':
                continue
            class_range = info['class range']
            mean_matrix[class_range[0]:class_range[1]] = info['meanvec']
        
        diff = data['features']-mean_matrix
        sw = np.dot(diff.T, diff)
        return sw

    def __between_scattar(self, data:dict, class_info:dict) -> np.ndarray:

        """
        SUM_{i=0}^c:    ( Ni(ui-u)(ui-u)^T)
                        (weighted_diff)(diff^T)
        """
        diff = np.zeros((class_info['total_class_num'],data['features'].shape[1]))
        weighted_diff = np.zeros((class_info['total_class_num'],data['features'].shape[1]))
        idx = 0
        total_mean = class_info['total_mean']
        
        for c,info in class_info.items():
            if c == 'total_mean' or c == 'total_class_num':
                continue
            
            diff[idx]=info['meanvec']-total_mean
            weighted_diff[idx] = info['num']*diff[idx]
            idx += 1
        sb = np.dot(weighted_diff.T,diff)
        return sb

    def _save_patterns(self, Sb, Sw, result,to_dim):
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        print(f"These matrices will be saved at {self.output_path}")    
        np.save(os.path.join(self.output_path,"Sb.npy"),Sb)
        np.savetxt(os.path.join(self.output_path,"Sb.txt"),Sb)

        np.save(os.path.join(self.output_path,"Sw.npy"),Sw)
        np.savetxt(os.path.join(self.output_path,"Sw.txt"),Sw)

        np.save(os.path.join(self.output_path,"Omega.npy"),self.omega)
        np.savetxt(os.path.join(self.output_path,"Omega.txt"),self.omega)
        
        np.save(os.path.join(self.output_path,"result_with_dim"+str(to_dim)+".npy"),result)
        np.savetxt(os.path.join(self.output_path,"result_with_dim"+str(to_dim)+".txt"),result)