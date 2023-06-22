import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import os

def pc(x: np.ndarray, method: str, undirected: bool = True):
    """PC Algorithm
    Wrapper around bnlearn functions.

    :param x: Dataset as of shape (n_samples, n_variables).
    :param method: Variation of the PC Algorithm
        - stable: PC-stable from "Order-independent constraint-based causal structure learning" [1]
        - hiton: HITON-PC from "Local Causal and Markov Blanket Induction for Causal Discovery and Feature Selection for Classification Part I: Algorithms and Empirical Evaluation" [2]
        - maxmin: max-min-PC from "Time and Sample Efficient Discovery of Markov Blankets and Direct Causal Relations" [3]
        - hybrid: Hybrid-PC from "A Hybrid Algorithm for Bayesian Network Structure Learning with Application to Multi-Label Learning" [4]
    :param undirected: Adjacency matrix to be pruned of shape (n_variables, n_variables).
    :return: Adjacency matrix.

    References:
    [1] https://dl.acm.org/doi/10.5555/2627435.2750365
    [2] https://dl.acm.org/doi/10.5555/1756006.1756013
    [3] https://doi.org/10.1145/956750.956838
    [4] https://doi.org/10.1016/j.eswa.2014.04.032

    For original references see: https://www.bnlearn.com/documentation/man/structure.learning.html
    """
    root = os.path.dirname(os.path.abspath(__file__))
    np_cv_rules = robjects.default_converter + numpy2ri.converter
    with np_cv_rules:
        robjects.globalenv['x'] = x
        robjects.globalenv['method'] = method
        robjects.globalenv['undirected'] = undirected
        robjects.r(f'source("{root}/R/pc.R")')
        result = np.array(robjects.globalenv['result'])
    return result
