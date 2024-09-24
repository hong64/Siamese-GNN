import numpy as np
import torch
import clhs as cl
from torch.utils.data import Dataset


class SiameseGraph(Dataset):
    def __init__(self, graph_dataset, cp=None, train=False):
        self.graph_dataset = graph_dataset
        self.cp = cp
        self.train = train

        ys = np.array([self._get(graph).numpy() for graph in graph_dataset])
        ys = np.squeeze(ys)
        n = len(graph_dataset)
        X = torch.empty(0, self.graph_dataset[0].x.shape[1])
        for i in range(len(self.graph_dataset)):
            feature = torch.mean(self.graph_dataset[i].x, dim=0).unsqueeze(0)
            X = torch.cat([X, feature], dim=0)
        if self.train:
            diffs = np.empty((0, 18))
            for i in range(len(self.graph_dataset)):
                flag = [True] * len(graph_dataset)
                flag[i] = False
                diff = np.zeros((n - 1, 18))
                diff[:, 0:15] = X[i, :] - X[flag, :]
                diff[:, 15] = i
                diff[:, 16] = list(range(0, i)) + list(range(i + 1, n))
                dominances = np.ones((n - 1, 1))
                neg = np.where((ys[i, :] >= ys[flag, :]).all(1))[0]
                dominances[neg, :] = 0
                diff[:, -1] = np.squeeze(dominances, axis=1)
                diffs = np.vstack((diffs, diff))
            sample_indices = cl.clhs(diffs[:, 0:15], num_samples=10000, max_iterations=100, progress=False,random_state=128)[
                "sample_indices"]
            samples = diffs[sample_indices, :]
            samples[0, 15:].astype(int)
            self.train_pairs = samples[:,15:17].tolist()
            self.train_targets = samples[:,-1].tolist()
        else:
            self.test_pairs = []
            self.test_targets = []
            diffs = np.empty((0, 18))
            for i in range(len(self.graph_dataset)):
                flag = [True] * len(graph_dataset)
                flag[i] = False
                diff = np.zeros((n - 1, 18))
                diff[:, 0:15] = X[i, :] - X[flag, :]
                diff[:, 15] = i
                diff[:, 16] = list(range(0, i)) + list(range(i + 1, n))
                dominances = np.ones((n - 1, 1))
                neg = np.where((ys[i, :] >= ys[flag, :]).all(1))[0]
                dominances[neg, :] = 0
                diff[:, -1] = np.squeeze(dominances, axis=1)
                diffs = np.vstack((diffs, diff))
            sample_indices = cl.clhs(diffs[:, 0:15], num_samples=10000, max_iterations=100, progress=False,random_state=128)[
                "sample_indices"]
            samples = diffs[sample_indices, :]
            samples[0, 15:].astype(int)
            self.test_pairs = samples[:, 15:17].tolist()
            self.test_targets = samples[:, -1].tolist()
    def _get(self, data):
        obj1 = (data.y[0][0] + data.y[0][1] + data.y[0][2] + data.y[0][3]) / 4
        obj2 = data.y[0][6]
        obj3 = data.y[0][7]
        return torch.Tensor([[obj1, obj2, obj3]])

    def __getitem__(self, index):
        if self.train:
            x1 = self.graph_dataset[int(self.train_pairs[index][0])]
            x2 = self.graph_dataset[int(self.train_pairs[index][1])]
            relation = int(self.train_targets[index])
        else:
            x1 = self.graph_dataset[int(self.test_pairs[index][0])]
            x2 = self.graph_dataset[int(self.test_pairs[index][1])]
            if self.cp:
                x1.cp_x = self.cp[int(self.test_pairs[index][0])].x
                x1.cp_hls_attr = self.cp[int(self.test_pairs[index][0])].hls_attr
                x1.cp_bench_name = self.cp[int(self.test_pairs[index][0])].bench_name
                x1.cp_prj_name = self.cp[int(self.test_pairs[index][0])].prj_name
                x1.cp_edge_index = self.cp[int(self.test_pairs[index][0])].edge_index
                x2.cp_x = self.cp[int(self.test_pairs[index][1])].x
                x2.cp_hls_attr = self.cp[int(self.test_pairs[index][1])].hls_attr
                x2.cp_bench_name = self.cp[int(self.test_pairs[index][1])].bench_name
                x2.cp_prj_name = self.cp[int(self.test_pairs[index][1])].prj_name
                x2.cp_edge_index = self.cp[int(self.test_pairs[index][1])].edge_index

            # append cp's hls_attr and cp's x
            relation = int(self.test_targets[index])

        return (x1, x2), relation

    def __len__(self):
        if self.train:
            return len(self.train_pairs)
        else:
            return len(self.test_pairs)

def sort_dominance(X,i,ids,label):
    dis = X[i,:] - X[ids,:]
    norm_ids = torch.norm(dis, dim=1, p=2)
    if label != "neg":
        indices = torch.argsort(norm_ids,descending=True)
    else:
        indices = torch.argsort(norm_ids)
    ids = [ids[index] for index in indices]
    return ids


class SiameseGraph_DSE(Dataset):
    def __init__(self, designs,anchor_id, remain_ids):
        self.designs = designs
        self.anchor_id = anchor_id
        self.remain_ids = remain_ids
    def __getitem__(self, index):
        compare_id = self.remain_ids[index]
        return self.designs[self.anchor_id],self.designs[compare_id]
    def __len__(self):
        return len(self.remain_ids)


