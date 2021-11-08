import numpy as np

def RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
 
    return np.sqrt(np.mean((a-b)**2))
