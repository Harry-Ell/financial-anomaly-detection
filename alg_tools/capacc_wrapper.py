'''
Storage area of script which allows us to interface between r and python, to 
make use of Martin Tvetens implementation of capa cc.
'''

import os
import configparser
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter


def _configure_r():
    '''
    Set up environment variables so that rpy2 can locate R.
    '''

    r_home = os.environ.get("R_HOME")
    if not r_home:
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(__file__), "r_config.ini")
        if os.path.exists(config_path):
            config.read(config_path)
            r_home = config.get("R", "home", fallback=None)

    if not r_home:
        raise EnvironmentError(
            "R installation directory not found. Set the R_HOME environment "
            "variable or provide it in alg_tools/r_config.ini."
        )

    bin_path = os.path.join(r_home, "bin")
    if os.path.exists(os.path.join(bin_path, "x64")):
        bin_path = os.path.join(bin_path, "x64")

    os.environ["R_HOME"] = r_home
    os.environ["PATH"] += os.pathsep + bin_path


_configure_r()

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
