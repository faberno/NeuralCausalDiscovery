import glob
import os
import urllib.request
import json
import random
from typing import Optional

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

BN_REPOS_DISCRETE = [
    "asia", "cancer", "earthquake", "sachs", "survey",  # small
    "alarm", "barley", "child", "insurance", "mildew", "water"  # medium
                                                       "hailfinder", "hepar2", "win95pts",  # large
    "andes", "diabetes", "link", "munin", "pathfinder", "pigs"  # very large
]

BN_REPOS_GAUSSIAN = [
    "ecoli70", "magic-niab",  # medium
    "magic-irri",  # large
    "arth150"  # very large
]

BN_REPOS_MIXED = [
    "healthcare", "sangiovese",  # small
    "mehra-original", "mehra-complete"  # medium
]

BN_URL = "https://www.bnlearn.com/bnrepository/"
root = os.path.dirname(os.path.abspath(__file__))
resources = os.path.join(root, "resources")


def find_dataset_index(dir: str):
    i = 0
    while len(glob.glob(os.path.join(dir, f"{i}_*"))) != 0:
        i += 1
    return i


def create_bnlearn_dataset(name: str, n_samples: int, seed: Optional[int] = None, save: bool = False):
    """Create a dataset from a bayesian network
    All possible BNs can be found at https://www.bnlearn.com/bnrepository/
    By specifying the name of the network, the needed .rda file is downloaded and
    a dataset randomly sampled.
    Three types of networks exist:
    - Discrete: only contains categorical data
    - Gaussian: only contains continuous data
    - Conditional Linear Gaussian: contains mixed data (categorical+continuous)

    :param name: Name of the BN
    :param n_samples: Number of samples the dataset should contain
    :param seed: Seed that should be specified in R
    :param save: Save the dataset as csv. If True, the path will be ncd/data/bnlearn/{name}/syn_datasets/{index}_{name}_{n_samples}_{seed}.csv
    :return: Adjacency matrix (pd), dataset (pd), states of all nodes in BN (dict: if discrete [states] else None)
    """
    name = name.lower()
    assert any([(name in REPO) for REPO in
                [BN_REPOS_DISCRETE, BN_REPOS_GAUSSIAN, BN_REPOS_MIXED]]), f"{name} was not found or is not supported"
    if not os.path.exists(resources):
        os.makedirs(resources)

    repo_path = os.path.join(resources, name)
    file_path = os.path.join(repo_path, f'{name}.rda')
    if not os.path.exists(repo_path):
        os.makedirs(repo_path)
    if not os.path.isfile(file_path):
        urllib.request.urlretrieve(f"{BN_URL}{name}/{name}.rda", file_path)

    if seed is None:
        seed = random.randint(0, 10000)

    np_cv_rules = robjects.default_converter + pandas2ri.converter
    with np_cv_rules:
        robjects.globalenv['seed'] = seed
        robjects.globalenv['n_samples'] = n_samples
        robjects.globalenv['file_path'] = file_path
        robjects.r(f'source("{root}/create_bn.R")')
        adj = pd.DataFrame(robjects.globalenv['adj'])
        x = pd.DataFrame(robjects.globalenv['X'])
    bn = robjects.globalenv['bn']
    is_categorical = x.dtypes == 'category'
    variable_states = {name: (list(entry[3].names[0]) if categ else None) for (name, entry, categ) in
                       zip(bn.names, bn, is_categorical)}
    graph_path = os.path.join(repo_path, f"{name}_graph.csv")
    if not os.path.isfile(graph_path):
        adj.to_csv(graph_path)

    state_path = os.path.join(repo_path, f"{name}_states.json")
    if not os.path.isfile(state_path):
        with open(state_path, 'w') as fp:
            json.dump(variable_states, fp)

    if save:
        dataset_path = os.path.join(repo_path, "syn_datasets")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            i = 0
        else:
            i = find_dataset_index(dataset_path)
        x.to_csv(os.path.join(dataset_path, f"{i}_{name}_{n_samples}_{seed}.csv"), index=False)

    return adj, x, variable_states


def load_bnlearn_dataset(name: str, index: int):
    """Load a previously created bnlearn dataset

    :param name: name of the BN
    :param index: index of the dataset. The index can be found in the file name: {index}_{name}_{n_samples}_{seed}.csv
    :return: Adjacency matrix (pd), dataset (pd), states of all nodes in BN (dict: if discrete [states] else None)
    """
    name = name.lower()
    repo_path = os.path.join(resources, name)
    graph_path = os.path.join(repo_path, f"{name}_graph.csv")
    state_path = os.path.join(repo_path, f"{name}_states.json")
    dataset_path = os.path.join(repo_path, "syn_datasets", f"{index}_{name}_*.csv")
    candidates = glob.glob(dataset_path)
    assert (len(candidates) > 0, f"No dataset with index {index} was found.")
    dataset_path = candidates[0]
    assert (
        all((os.path.isfile(graph_path), os.path.isfile(state_path), os.path.isfile(dataset_path)))
    )
    adj = pd.read_csv(graph_path, index_col=0)
    x = pd.read_csv(dataset_path)
    with open(state_path, 'r') as fp:
        states = json.load(fp)

    return adj, x, states
