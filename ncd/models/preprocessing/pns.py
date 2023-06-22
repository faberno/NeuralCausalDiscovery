import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
from typing import Optional


def pns(x: np.ndarray, adj_start: Optional[np.ndarray] = None, num_neighbors: Optional[int] = None,
        thresh: float = 0.75, n_estimators: int =500):
    """Preliminary neighborhood selection
    Selects a set of potential parents for each variable. Descriptions can be found at:
    - Bühlmann et al. (2014) [1] Section 3.1
    - Lachapelle et al. (2019) [2] Section A.3

    :param x: Dataset as of shape (n_samples, n_variables).
    :param adj_start: Prior known adjacency matrix of shape (n_variables, n_variables).
    :param num_neighbors: The maximum number of features to select. If None, then all features are kept.
    :param thresh: The threshold value to use for feature selection [3].
    :param n_estimators: Number of estimators in slearn.ExtraTreesRegressor
    :return: Adjacency matrix of possible parents for each node.

    Implementation from https://github.com/kurowasan/GraN-DAG/blob/master/gran_dag/train.py

    References:
    [1] https://doi.org/10.1214/14-AOS1260
    [2] https://doi.org/10.48550/arXiv.1906.02226
    [3] https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
    """
    num_nodes = x.shape[1]

    if adj_start is None:
        adj_start = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    for node in tqdm(range(num_nodes), desc="PNS"):
        x_other = np.copy(x)
        x_other[:, node] = 0
        reg = ExtraTreesRegressor(n_estimators=n_estimators)
        reg = reg.fit(x_other, x[:, node])
        selected_reg = SelectFromModel(reg, threshold=f"{thresh}*mean", prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False).astype(np.float)

        adj_start[:, node] *= mask_selected

    return adj_start
