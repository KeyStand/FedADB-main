import torch
from torch.utils.data import DataLoader
from collections import Counter

import random
import numpy as np
import os

from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10


__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "cifar10": get_all_targets_cifar10,
}
DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
}


import os
import pickle
import numpy as np

def get_partition_str(partition):
    """Generate a string representation of the partition configuration."""
    if partition.method == "centralized":
        return "centralized"
    elif partition.method == "iid":
        return "iid"
    elif partition.method == "lda":
        return f"lda_{partition.alpha:.4f}"
    else:
        raise ValueError("Unknown partition method")

def data_distributer(
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    oracle_size=0,
    oracle_batch_size=None,
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    Saves the data distribution for the current parameters and loads it if previously saved.
    """
    root = os.path.join(root, dataset_name)
    all_targets = DATA_INSTANCES[dataset_name](root)
    num_classes = len(np.unique(all_targets))

    # Generate unique identifier for the data distribution
    partition_str = get_partition_str(partition)
    identifier = f"{dataset_name}_{n_clients}_{partition_str}"
    save_dir = os.path.join(root, 'distributions')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, identifier + '.pkl')

    # Check if saved distribution exists
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
        net_dataidx_map = saved_data['net_dataidx_map']
        net_dataidx_map_test = saved_data.get('net_dataidx_map_test', None)
        oracle_idxs = saved_data.get('oracle_idxs', None)
        print("loading saved data_distributer")
    else:
        # Compute partitions if no saved distribution is found
        net_dataidx_map_test = None
        if partition.method == "centralized":
            net_dataidx_map = centralized_partition(all_targets)
        elif partition.method == "iid":
            net_dataidx_map = iid_partition(all_targets, n_clients)
        elif partition.method == "lda":
            net_dataidx_map = lda_partition(all_targets, n_clients, partition.alpha)
        else:
            raise NotImplementedError

        # Compute oracle indices
        oracle_idxs = oracle_partition(all_targets, oracle_size=oracle_size)

        # Save the computed distribution
        saved_data = {
            'net_dataidx_map': net_dataidx_map,
            'net_dataidx_map_test': net_dataidx_map_test,
            'oracle_idxs': oracle_idxs,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(saved_data, f)

    # Initialize local loaders
    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None} for i in range(n_clients)
    }

    # Distribute client train data
    print(">>> Distributing client train data...")
    for client_idx, dataidxs in net_dataidx_map.items():
        local_loaders[client_idx]["datasize"] = len(dataidxs)
        local_loaders[client_idx]["train"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=batch_size, dataidxs=dataidxs,
        )

    # Distribute client test data if applicable
    print(">>> Distributing client test data...")
    if net_dataidx_map_test is not None:
        for client_idx, dataidxs in net_dataidx_map_test.items():
            local_testloader = DATA_LOADERS[dataset_name](
                root, train=False, batch_size=batch_size, dataidxs=dataidxs,
            )
            local_loaders[client_idx]["test"] = local_testloader
            local_loaders[client_idx]["dist"] = get_dist_vec(
                local_testloader, num_classes
            )

    # Create global loaders
    global_loaders = {
        "train": DATA_LOADERS[dataset_name](root, train=True, batch_size=batch_size),
        "test": DATA_LOADERS[dataset_name](root, train=False, batch_size=batch_size),
    }

    # Compute data map
    data_map = net_dataidx_map_counter(net_dataidx_map, all_targets)

    # Assemble data distribution
    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "data_map": data_map,
        "num_classes": num_classes,
    }

    # Set oracle loader if applicable
    obs = batch_size if oracle_batch_size is None else oracle_batch_size
    if oracle_idxs is not None:
        data_distributed["oracle"] = DATA_LOADERS[dataset_name](
            root, train=True, batch_size=obs, dataidxs=oracle_idxs
        )

    print(data_distributed)
    return data_distributed
def centralized_partition(all_targets):
    labels = all_targets
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    tot_idx = np.array(tot_idx)
    np.random.shuffle(tot_idx)
    net_dataidx_map[0] = tot_idx

    return net_dataidx_map


def iid_partition(all_targets, n_clients):
    labels = all_targets
    length = int(len(labels) / n_clients)
    tot_idx = np.arange(len(labels))
    net_dataidx_map = {}

    for client_idx in range(n_clients):
        np.random.shuffle(tot_idx)
        data_idxs = tot_idx[:length]
        tot_idx = tot_idx[length:]
        net_dataidx_map[client_idx] = np.array(data_idxs)

    return net_dataidx_map



def lda_partition(all_targets, n_clients, alpha):
    labels = all_targets
    length = int(len(labels) / n_clients)
    net_dataidx_map = {}

    unique_classes = np.unique(labels)

    tot_idx_by_label = []
    for i in unique_classes:
        idx_by_label = np.where(labels == i)[0]
        tot_idx_by_label.append(idx_by_label)

    min_size = 0

    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        N, K = len(all_targets), len(np.unique(all_targets))

        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(all_targets == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(
                N, alpha, n_clients, idx_batch, idx_k
            )

    for i in range(n_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def oracle_partition(all_targets, oracle_size=0):
    oracle_idxs = None

    if oracle_size != 0:
        idxs_dict = {}

        for i in range(len(all_targets)):
            label = torch.tensor(all_targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

            oracle_idxs = []

        for value in idxs_dict.values():

            oracle_idxs += value[0:oracle_size]

    return oracle_idxs


def get_dist_vec(dataloader, num_classes):
    """Calculate distribution vector for local set"""
    targets = dataloader.dataset.targets
    dist_vec = torch.zeros(num_classes)
    counter = Counter(targets)

    for class_idx, count in counter.items():
        dist_vec[class_idx] = count

    dist_vec /= len(targets)

    return dist_vec


def net_dataidx_map_counter(net_dataidx_map, all_targets):
    data_map = [[] for _ in range(len(net_dataidx_map.keys()))]
    num_classes = len(np.unique(all_targets))

    prev_key = -1
    for key, item in net_dataidx_map.items():
        client_class_count = [0 for _ in range(num_classes)]
        class_elems = all_targets[item]
        for elem in class_elems:
            client_class_count[elem] += 1

        data_map[key] = client_class_count

    return np.array(data_map)
