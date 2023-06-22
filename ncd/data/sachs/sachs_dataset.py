import os
import pandas as pd
from typing import Union, Tuple
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))


def sachs(obs: bool = True, as_df: bool = False):
    """Protein-Signaling Network by Sachs et al. (https://doi.org/10.1126/science.1105809)
    - 11 nodes / 17 edges
    - 853 (obs) / 7466 (obs + interventions) samples

    :param obs: If true, the purely observational dataset is loaded, if not, the observational+interventional one.
    :param as_df: If true return pd.Dataframes, if not return np.array.
    :return: Adjacency matrix (11x11) and dataset (853/7466x11) as dataframe or array.
    """
    G = pd.read_csv(
        os.path.join(root, 'sachs_graph.csv'),
        index_col=0
    ).astype(np.float32)

    data_name = 'obs' if obs else 'int'
    X = pd.read_csv(
        os.path.join(root, f'sachs_{data_name}_data.csv')
    ).astype(np.float32)

    if as_df:
        return G, X
    return G.to_numpy(), X.to_numpy()
