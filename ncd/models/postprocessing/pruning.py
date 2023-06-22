import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import os

def cam_pruning(x: np.ndarray, adj_start: np.ndarray, cutoff: float = 0.001, verbose: bool = False):
    """CAM Pruning to remove spurious edges

    :param x: Dataset as of shape (n_samples, n_variables).
    :param adj_start: Adjacency matrix to be pruned of shape (n_variables, n_variables).
    :param cutoff: cutOffPVal in selGam method.
    :param verbose: Print output of R script.
    :return: Pruned adjacency matrix.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    np_cv_rules = robjects.default_converter + numpy2ri.converter
    with np_cv_rules:
        robjects.globalenv['dataset'] = x
        robjects.globalenv['dag'] = adj_start
        robjects.globalenv['cutoff'] = cutoff
        robjects.globalenv['verbose'] = verbose
        robjects.r(f'source("{root}/R/cam_pruning.R")')
        pruned_dag = np.array(robjects.globalenv['pruned_dag'])
    return pruned_dag
