from hier_models.hier_bram_model import *
from hier_models.hier_cp_model import *
from hier_models.hier_dsp_model import *
from hier_models.hier_ff_model import *
from hier_models.hier_lut_model import *
from hier_models.hier_pwr_model import *
import torch
torch.manual_seed(128)
def pred_obj(models,data,device):
    preds = np.array([]).reshape(batch_size, 0)
    for obj_type, model in models.items():
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            if obj_type == "cp":
                out = model(data.cp_x, data.cp_edge_index, data.batch, data.cp_hls_attr)
            else:
                out = model(data.x, data.edge_index, data.batch, data.hls_attr)
            preds = np.hstack((preds, out.view(-1).to("cpu").reshape(batch_size, 1)))
    return preds
def valid_acc(models, siamese_valid_dataset, device):

    siamese_loader = DataLoader(siamese_valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_acc = 0
    count = 0
    for _, data in enumerate(siamese_loader):
        preds_0 = pred_obj(models, data[0][0], device)
        preds_1 = pred_obj(models, data[0][1], device)
        area_0 = ((preds_0[:, 0] + preds_0[:, 1] + preds_0[:, 2] + preds_0[:, 3]) / 4).reshape(batch_size, 1)
        area_1 = ((preds_1[:, 0] + preds_1[:, 1] + preds_1[:, 2] + preds_1[:, 3]) / 4).reshape(batch_size, 1)
        preds_0 = np.hstack((preds_0, area_0))
        preds_1 = np.hstack((preds_1, area_1))
        target = data[1].numpy()
        dominance_pred = np.where(((preds_0[:, 4:] - preds_1[:, 4:]) >= [0, 0, 0]).all(1), 0, 1)
        acc = (target == dominance_pred).sum()
        total_acc += acc
        count += 1
    total_acc /= ( len(siamese_loader)*batch_size)
    return total_acc
def save(reg,epoch,model_dir,type):
    checkpoint = {
                'model': reg["best_valid_model"].state_dict(),
                'optimizer': reg["optimizer"].state_dict(),
                'epoch': epoch,
                'min_test_mae': reg["min_test_mae"]
            }
    torch.save(checkpoint, os.path.join(model_dir, 'hier_' + type + '_test.pt'))
def train_hier_gnns(train_ds,valid_ds,train_ds_cp,valid_ds_cp,siamese_test,fold):
    epoches = 500
    bram = bram_model(device,train_ds,valid_ds)
    cp = cp_model(device, train_ds_cp, valid_ds_cp)
    dsp = dsp_model(device, train_ds, valid_ds)
    ff = ff_model(device, train_ds, valid_ds)
    lut = lut_model(device, train_ds, valid_ds)
    pwr = pwr_model(device, train_ds, valid_ds)
    max_acc = -100000
    siamese_valid_dataset = siamese_test
    model_dir = "./hier/"+str(fold)
    for epoch in range(epoches):

        if not bram["nan"]:
            train_bram_model(bram,epoch,device)
        if not cp["nan"]:
            train_cp_model(cp,epoch,device)
        if not dsp["nan"]:
            train_dsp_model(dsp,epoch,device)
        if not ff["nan"]:
            train_ff_model(bram,epoch,device)
        if not lut["nan"]:
            train_lut_model(lut,epoch,device)
        if not bram["nan"]:
            train_pwr_model(pwr,epoch,device)
        models = {"bram": bram["best_valid_model"], "ff": ff["best_valid_model"],"lut": lut["best_valid_model"],
                  "dsp": dsp["best_valid_model"], "cp": cp["best_valid_model"], "pwr": pwr["best_valid_model"]}
        acc = valid_acc(models, siamese_valid_dataset, device)
        # print(acc)
        if acc > max_acc:
            save(bram,epoch,model_dir,"bram")
            save(cp, epoch, model_dir, "cp")
            save(dsp, epoch, model_dir, "dsp")
            save(ff, epoch, model_dir, "ff")
            save(lut, epoch, model_dir, "lut")
            save(pwr, epoch, model_dir, "pwr")
            max_acc = acc
    print(max_acc)
    return max_acc


model_name = "hier"
#将数据集划分威训练集和测试集
batch_size = 1024
dataset_dir = os.path.abspath('train_ds/std')
dataset = os.listdir(dataset_dir)
ds = generate_dataset(dataset_dir, dataset, print_info=False)

dataset_dir_cp = os.path.abspath('train_ds/rdc')
dataset_cp = os.listdir(dataset_dir_cp)
ds_cp = generate_dataset(dataset_dir_cp, dataset_cp, print_info=False)
k = 5
k_train,k_train_cp,k_test,k_test_cp,siamese_test = divide_by_benchmark(ds,k,ds_cp)


logs = {"acc":[],"losses":[],"max_acc":[]}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(k):
    if i == 3 or i==4:
        continue
    print("\033[32m Type: %s\033[0m" % "Siamese-GNN")
    max_valid_acc = train_hier_gnns(k_train[i], k_test[i],k_train_cp[i],k_test_cp[i],siamese_test[i],i+1)
    logs["acc"].append( [max_valid_acc])
    print("\033[32m ACC: %f\033[0m" % max_valid_acc)
np.save("./k_result/"+model_name+"_1_2_3.npy", np.array(logs))








