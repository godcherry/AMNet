import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import pickle
import torch_geometric
from config import *

data_dir = project_path / "dataset"


def aucPerformance(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    roc_auc = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    return roc_auc, auc_pr


def train_val_test_split(*arrays, train_size=0.1, val_size=0.1, test_size=0.8, stratify=None, random_state=None):

    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result



def load_raw_dataset(dataset):
    """
    :param dataset: dataset_name
    :return:   graph: nx.Digraph
               x: torch.tensor
               y: torch.tensor
    """
    with open(data_dir / (dataset + '.graph'), 'rb') as f:
        graph = pickle.load(f)
    with open(data_dir / (dataset + '.x'), 'rb') as f:
        x = pickle.load(f)
    with open(data_dir / (dataset + '.y'), 'rb') as f:
        y = pickle.load(f)
    return graph, x, y


def load_pyg_data(dataset, split='0.5_0.2_0.3', to_undirected=True, fixed_split=True, cached=True):
    """
    :param dataset:
    :param split:
    :param to_undirected:
    :param fixed_split:
    :return:
    """
    graph, x, y = load_raw_dataset(dataset)
    if to_undirected:
        graph = graph.to_undirected()

    # create pyg data object
    data = torch_geometric.utils.from_networkx(graph)
    data.x = x
    data.y = y

    split_pert = [float(i) for i in split.split("_")]
    has_val = (len(split_pert) == 3)

    if fixed_split:
        with open(data_dir / (dataset + split + '.split'), 'rb') as f:
            split_dict = pickle.load(f)
            if has_val:
                idx_train, idx_val, idx_test = split_dict['train'], split_dict['val'], split_dict['test']
            else:
                idx_train, idx_test = split_dict['train'], split_dict['test']
    else:
        train_size, val_size, test_size = split_pert
        idx_train, idx_val, idx_test = train_val_test_split(np.arange(len(y)), train_size=train_size,
                                 val_size=val_size, test_size=test_size,
                                 stratify=y, random_state=None)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if cached:
            with open(data_dir / (dataset + split + '.split'), 'wb') as f:
                pickle.dump({'train': idx_train,
                             'val': idx_val,
                             'test': idx_test}, f)


    if has_val:
        data.train_mask = idx_train
        data.val_mask = idx_val
        data.test_mask = idx_test
    else:
        data.train_mask = idx_train
        data.test_mask = idx_test

    return data