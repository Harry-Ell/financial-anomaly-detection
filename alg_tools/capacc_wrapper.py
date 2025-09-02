'''
Storage area of script which allows us to interface between r and python, to 
make use of Martin Tvetens implementation of capa cc.
'''

import os
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter

os.environ["R_HOME"] = r"C:/Users/ellinghh/AppData/Local/Programs/R/R-4.5.1"
os.environ["PATH"] += os.pathsep + r"C:/Users/ellinghh/AppData/Local/Programs/R/R-4.5.1/bin/x64"

# Load packages
utils = rpackages.importr('utils')
capacc = rpackages.importr('capacc')
robjects.r('library(Matrix)')


def capa_cc(X, Q, b=0.05, b_point=0.1, min_seg_len=4):
    
    with localconverter(numpy2ri.converter):
        r_X = robjects.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
        robjects.globalenv['x'] = r_X

        r_Q = robjects.r.matrix(Q, nrow=Q.shape[0], ncol=Q.shape[1])
        robjects.globalenv['Q_dense'] = r_Q

    robjects.r('Q <- Matrix(Q_dense, sparse = TRUE)')
    robjects.r(f'res <- capa.cc(x, Q, b={b}, b_point={b_point}, min_seg_len={min_seg_len})$anoms')
    result = robjects.r['res']

    with localconverter(pandas2ri.converter):
        df_result = robjects.conversion.rpy2py(result)
    return df_result
