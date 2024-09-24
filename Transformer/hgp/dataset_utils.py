import os
import random
import numpy as np
from torch.utils.data import random_split
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from datasets import SiameseGraph
import copy
def msle_loss(output, target):
    output = torch.log(output + 1)
    target = torch.log(target + 1)
    return torch.mean(torch.square(output - target))


def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target))


def mae_loss(output, target):
    return torch.mean(torch.abs(target - output))



import torch
def generate_dataset_siamese(dataset_dir, dataset_name_list, print_info=False):
    dataset_list = list()
    for ds in dataset_name_list:

        ds_path = os.path.join(dataset_dir, ds)

        if os.path.isfile(ds_path):
            tem_data = torch.load(ds_path)
            X = np.empty((0,15))
            H = np.empty((0,6))
            for i in range(len(tem_data)):
                X = np.vstack((X, tem_data[i].x))
                H = np.vstack((H,tem_data[i].hls_attr[0]))
            mean1, std1 = np.mean(X, axis=0), np.std(X, axis=0)
            std1 = np.where(std1 != 0, std1, 1e+4)
            mean2, std2 = np.mean(H, axis=0), np.std(H, axis=0)
            std2 = np.where(std2 != 0, std2, 1e+4)

            for i in range(len(tem_data)):
                tem_data[i].x = (tem_data[i].x - mean1) / std1
                tem_data[i].hls_attr[0] = (tem_data[i].hls_attr[0] - mean2) / std2


            dataset_list.append(tem_data)
            if print_info:
                print(ds_path)


    return dataset_list

def generate_dataset(dataset_dir, dataset_name_list, print_info=False):
    dataset_list = list()
    for ds in dataset_name_list:
        ds_path = os.path.join(dataset_dir, ds)
        if os.path.isfile(ds_path):
            tem_data = torch.load(ds_path)
            # dataset_list = dataset_list + tem_data
            dataset_list.append(tem_data)
            if print_info:
                print(ds_path)
    return dataset_list







def split_dataset(all_list, shuffle=True, seed=6666):
    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    # print("first ten train graphs Y before shuffle:", first_10_y)

    if shuffle and seed is not None:
        np.random.RandomState(seed=seed).shuffle(all_list)
        # print("seed number:", seed)
    elif shuffle and seed is None:
        random.shuffle(all_list)
        # print("seed number:", seed)

    first_10_y = []
    for i in all_list[0:10]:
        first_10_y.append(i.y)
    print("first ten train graphs Y after shuffle:", first_10_y)


    train_ds, test_ds = random_split(all_list, [round(0.8 * len(all_list)), round(0.2 * len(all_list))],
                                     generator=torch.Generator().manual_seed(42))

    return train_ds, test_ds
def divide_by_benchmark_siamese(ds,k):
    train_ds = [list() for _ in range(k)]
    test_ds = [list() for _ in range(k)]
    kfold = KFold(k, shuffle=True, random_state=42)
    for i in range(len(ds)):
        j = 0
        for train_idx, valid_idx in kfold.split(ds[i]):
            train_ds[j].append(Subset(ds[i], train_idx))
            test_ds[j].append(Subset(ds[i], valid_idx))
            j += 1
    k_test = []
    for i in range(k):
        siamese_designs = SiameseGraph(copy.deepcopy(test_ds[i][0]))
        for j in range(1,len(test_ds[i])):
            temp = SiameseGraph(copy.deepcopy(test_ds[i][j]))
            L1 = len(siamese_designs.graph_dataset.dataset)
            L2 = siamese_designs.graph_dataset.indices.shape[0]
            siamese_designs.graph_dataset.indices = np.append(siamese_designs.graph_dataset.indices,temp.graph_dataset.indices+L1)
            siamese_designs.test_pairs.extend((np.array(temp.test_pairs)+[L2,L2]).tolist())
            siamese_designs.test_targets.extend(temp.test_targets)
            siamese_designs.graph_dataset.dataset.extend(temp.graph_dataset.dataset)
        k_test.append(siamese_designs)
    return train_ds,k_test

def divide_by_benchmark(ds,k,ds_cp):
    train_ds = [list() for _ in range(k)]
    test_ds = [list() for _ in range(k)]
    train_ds_cp = [list() for _ in range(k)]
    test_ds_cp = [list() for _ in range(k)]

    kfold = KFold(k, shuffle=True, random_state=42)
    for i in range(len(ds)):
        j = 0
        for train_idx, valid_idx in kfold.split(ds[i]):
            train_ds[j].append(Subset(ds[i], train_idx))
            test_ds[j].append(Subset(ds[i], valid_idx))
            train_ds_cp[j].append(Subset(ds_cp[i], train_idx))
            test_ds_cp[j].append(Subset(ds_cp[i],valid_idx))
            j += 1
    siamese_test = []
    for i in range(k):
        siamese_designs = SiameseGraph(copy.deepcopy(test_ds[i][0]),cp=copy.deepcopy(test_ds_cp[i][0]))
        for j in range(1,len(test_ds[i])):
            temp = SiameseGraph(copy.deepcopy(test_ds[i][j]),cp=copy.deepcopy(test_ds_cp[i][j]))
            L1 = len(siamese_designs.graph_dataset.dataset)
            L2 = siamese_designs.graph_dataset.indices.shape[0]
            siamese_designs.graph_dataset.indices = np.append(siamese_designs.graph_dataset.indices,temp.graph_dataset.indices+L1)
            siamese_designs.cp.indices = np.append(siamese_designs.cp.indices,temp.cp.indices+L1)
            siamese_designs.test_pairs.extend((np.array(temp.test_pairs)+[L2,L2]).tolist())
            siamese_designs.test_targets.extend(temp.test_targets)
            siamese_designs.graph_dataset.dataset.extend(temp.graph_dataset.dataset)
            siamese_designs.cp.dataset.extend(temp.cp.dataset)
        siamese_test.append(siamese_designs)

    k_train = []
    k_train_cp = []
    k_test = []
    k_test_cp = []
    for i in range(k):
        kth_train = copy.deepcopy(train_ds[i][0])
        kth_train_cp = copy.deepcopy(train_ds_cp[i][0])
        kth_test = copy.deepcopy(test_ds[i][0])
        kth_test_cp = copy.deepcopy(test_ds_cp[i][0])

        for j in range(1, len(train_ds[i])):
            L = len(kth_train.dataset)
            kth_train.dataset.extend(train_ds[i][j].dataset)
            kth_train_cp.dataset.extend(train_ds_cp[i][j].dataset)
            kth_train.indices = np.append(kth_train.indices,
                                          train_ds[i][j].indices + L)
            kth_train_cp.indices = np.append(kth_train_cp.indices,
                                             train_ds_cp[i][j].indices + L)

        for j in range(1, len(test_ds[i])):
            L = len(kth_test.dataset)
            kth_test.dataset.extend(test_ds[i][j].dataset)
            kth_test_cp.dataset.extend(test_ds_cp[i][j].dataset)
            kth_test.indices = np.append(kth_test.indices,
                                          test_ds[i][j].indices + L)
            kth_test_cp.indices = np.append(kth_test_cp.indices,
                                             test_ds_cp[i][j].indices + L)
        k_train.append(kth_train)
        k_train_cp.append(kth_train_cp)

        k_test.append(kth_test)
        k_test_cp.append(kth_test_cp)

    return k_train,k_train_cp,k_test,k_test_cp,siamese_test
