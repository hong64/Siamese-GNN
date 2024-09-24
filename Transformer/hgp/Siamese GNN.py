

from dataset_utils import *
from torch.utils.data import DataLoader
from Contrastive_Model import  train_siamese_gnn
from datasets import SiameseGraph
import  torch
torch.manual_seed(128)
#将数据集划分威训练集和测试集
dataset_dir = os.path.abspath('train_ds/std')
dataset = os.listdir(dataset_dir)
ds = generate_dataset_siamese(dataset_dir, dataset, print_info=False)
k=5
train_ds,k_test = divide_by_benchmark_siamese(ds,k)
k_train = []
for i in range(k):
    siamese_designs = SiameseGraph(copy.deepcopy(train_ds[i][0]),train=True)
    for j in range(1,len(train_ds[i])):
        temp = SiameseGraph(copy.deepcopy(train_ds[i][j]),train=True)
        L1 = len(siamese_designs.graph_dataset.dataset)
        L2 = siamese_designs.graph_dataset.indices.shape[0]
        siamese_designs.graph_dataset.indices = np.append(siamese_designs.graph_dataset.indices,temp.graph_dataset.indices+L1)
        siamese_designs.train_pairs.extend((np.array(temp.train_pairs)+[L2,L2]).tolist())
        siamese_designs.train_targets.extend(temp.train_targets)
        siamese_designs.graph_dataset.dataset.extend(temp.graph_dataset.dataset)
    k_train.append(siamese_designs)

fold = 1
logs = {"max_acc":[],"losses":[]}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(k):

    print(len(k_train[i]),len(k_test[i]))
    train_loader = DataLoader(k_train[i], shuffle=True,batch_size=32, drop_last=True, pin_memory=False)
    test_loader = DataLoader(k_test[i], batch_size=32, shuffle=True,drop_last=True, pin_memory=False)
    print("\033[32m Type: %s\033[0m" % "Siamese-GNN")
    max_train_acc, max_test_acc = train_siamese_gnn(train_loader, test_loader, fold, device)
    logs["max_acc"].append([max_train_acc, max_test_acc])
    fold+=1
    print("\033[32m ACC: %f\033[0m" % max_test_acc)

np.save("./siamese_gnn/train_data.npy", np.array(logs))







