from dataset_utils import *
from Contrastive_Architecture import ModelConfig
import time
from hier_models.hier_bram_model import HierNet
from hgp.ironman_pro.graph_model_cp import GCNNet


device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
def best_fold(model_type):
    if model_type == "hgp":
        data = np.load("./k_result/hier.npy", allow_pickle=True).item()
    elif model_type == "ironman":
        data = np.load("./k_result/iron.npy", allow_pickle=True).item()
    accs = data["acc"]
    return accs.index(max(accs))+1

def single_obj_model(fold,obj_type,model_type):
    if model_type == "hgp":
        path = './hier/'
        if obj_type in ["bram","dsp"]:
            model_path = os.path.abspath(path+str(fold)+'/hier_'+obj_type+'_test.pt')
            model = HierNet(in_channels=15, hidden_channels=64, num_layers=3, conv_type='sage',
                                    hls_dim=6, drop_out=0.0)
        elif obj_type in ["ff","lut"]:
            model_path = os.path.abspath(path + str(fold) + '/hier_' + obj_type + '_test.pt')
            model = HierNet(in_channels=15, hidden_channels=64, num_layers=3, conv_type='sage',
                            hls_dim=6, drop_out=0.0)
        elif obj_type == "power":
            model_path = os.path.abspath(path + str(fold) + '/hier_' + 'pwr' + '_test.pt')
            model = HierNet(in_channels=15, hidden_channels=64, num_layers=3, conv_type='sage',
                            hls_dim=6, drop_out=0.0)
        else:
            model_path = os.path.abspath(path + str(fold) + '/hier_' + obj_type + '_test.pt')
            model = HierNet(in_channels=11, hidden_channels=64, num_layers=3, conv_type='sage',
                            hls_dim=1, drop_out=0.0)
        params = torch.load(model_path, map_location=device)
        model.load_state_dict(params['model'])
    elif model_type == "ironman":
        path = './iron/'
        if obj_type in ["bram", "dsp"]:
            model_path = os.path.abspath(path + str(fold) + '/iron_' + obj_type + '_test.pt')
            model = GCNNet(in_channels=15)
        elif obj_type in ["ff", "lut"]:
            model_path = os.path.abspath(path + str(fold) + '/iron_' + obj_type + '_test.pt')
            model = GCNNet(in_channels=15)
        elif obj_type == "power":
            model_path = os.path.abspath(path + str(fold) + '/iron_pwr_test.pt')
            model = GCNNet(in_channels=15)
        else:
            model_path = os.path.abspath(path + str(fold) + '/iron_cp_test.pt')
            model = GCNNet(in_channels=11)
        params = torch.load(model_path, map_location=device)
        model.load_state_dict(params['model'])
    return model

def load(model_type):
    if model_type == "siamese":
        # model_path = os.path.abspath('./model_comparison/sage/2/cls_loss_h64_d0_checkpoint_test.pt')
        model_path = os.path.abspath('./siamese_gnn/4/cls_test.pt')
        params = torch.load(model_path, map_location=device)
        model = ModelConfig(15, device)
        model.load_state_dict(params['model'])
        return model
    else:
        obj_models = {"lut": None, "ff": None, "dsp": None, "bram": None, "cp": None, "power": None}
        fold = best_fold(model_type)
        for obj_type in list(obj_models.keys()):
            obj_models[obj_type] = single_obj_model(fold, obj_type,model_type)
        return obj_models

def siamese_gnn(config1, config2, model,nodes,inference_time):
    config1.to(device)
    config2.to(device)
    model.eval()
    with torch.no_grad():
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        batch1, batch2 = torch.tensor([0 for _ in range(config1.x.shape[0])]).to(device), torch.tensor(
            [0 for _ in range(config2.x.shape[0])]).to(device)
        start_time = time.perf_counter()
        out = model(config1.x, config2.x, config1.edge_index,
                    config2.edge_index, batch1, batch2,
                    config1.hls_attr, config2.hls_attr)
        end_time = time.perf_counter()
        curr_time = end_time - start_time
        inference_time.append(curr_time)# 计算时间
        nodes.append([config1.x.shape[0],config2.x.shape[0],config1.x.shape[0]+config2.x.shape[0]])
    return [out.max(1)[1].item(),out[0][1].item()]
def post_synthesis(config):
    obj1 = (config.y[0][0] + config.y[0][1]+config.y[0][2] + config.y[0][3])/4
    obj2 = config.y[0][6].cpu()
    obj3 = config.y[0][7].cpu()
    return [obj1.cpu(),obj2,obj3]

def ADRS(test_ds,ref_configs):
    #generate the synthesized pareto front
    testPoints = [post_synthesis(config) for config in test_ds]
    paretoPoints, dominatedPoints_pareto = simple_cull(testPoints, dominates)

    refPoints = []
    ADRSes = []

    for config in ref_configs:
        refPoints.append(post_synthesis(config))
        paretoPoints_estimated, dominatedPoints_estimated= simple_cull(copy.deepcopy(refPoints), dominates)
        diff = 0.0
        for r in paretoPoints:
            dist = []
            for s in paretoPoints_estimated:
                s = np.array(s)
                r = np.array(r)
                tmp = np.linalg.norm((s - r) / r, np.inf)
                dist.append(tmp)
            f = min(dist)
            diff += f
        adrs = diff / len(paretoPoints)
        ADRSes.append(adrs)
    return ADRSes


def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)

def reg_pred_objs(reg_models, design, design_cp,model_type):
    objs_pred = []
    design.to(device)
    design_cp.to(device)
    # ['lut', 'ff', 'dsp', 'bram', 'cp', 'power']
    for obj_type in list(reg_models.keys()):
        model = reg_models[obj_type].to(device)
        model.eval()
        with torch.no_grad():
            batch = torch.tensor([0 for _ in range(design.x.shape[0])]).to(device)
            if obj_type == "cp":
                if model_type == "hgp":
                    objs_pred.append(model(design_cp.x, design_cp.edge_index, batch, design_cp.hls_attr).item())
                elif model_type == "ironman":
                    objs_pred.append(model(design_cp.x, design_cp.edge_index, batch).item())

            else:
                if model_type == "hgp":
                    objs_pred.append(model(design.x, design.edge_index, batch, design.hls_attr).item())
                elif model_type == "ironman":
                    objs_pred.append(model(design.x, design.edge_index, batch).item())
    objs = [(objs_pred[0] + objs_pred[1] + objs_pred[2] + objs_pred[3]) / 4, objs_pred[4], objs_pred[5]]
    return objs


def infer_dominances(model_type,model,design1,design2,design1_cp, design2_cp,nodes,inference_time):
    if model_type == "siamese":
        pred1, prob1 = siamese_gnn(design1, design2,model,nodes,inference_time)
        pred2, prob2 = siamese_gnn(design2, design1,model,nodes,inference_time)
        if pred1 == 0 and pred2 == 0:
            return 1,1
        return pred1,pred2
    else:
        if not design1.batch:
            objs1 = reg_pred_objs(model, design1, design1_cp,model_type)
            design1.batch = objs1
        else:
            objs1 = design1.batch
        if not design2.batch:
            objs2 = reg_pred_objs(model, design2, design2_cp,model_type)
            design2.batch = objs2
        else:
            objs2 = design2.batch
        if (np.array(objs1) <= np.array(objs2)).all() and (np.array(objs2) != np.array(objs1)).any():
            return 1,0
        elif (np.array(objs1) >= np.array(objs2)).all():
            return 0,1
        else:
            return 1,1

def DSE(model_type,benchmark,nodes,inference_time):
    ds_path = os.path.abspath('./DSE_ds/std/'+benchmark)
    test_ds = torch.load(ds_path)


    if model_type == "siamese":
        X = np.empty((0, 15))
        H = np.empty((0, 6))
        for i in range(len(test_ds)):
            X = np.vstack((X, test_ds[i].x))
            H = np.vstack((H, test_ds[i].hls_attr[0]))
        mean1, std1 = np.mean(X, axis=0), np.std(X, axis=0)
        std1 = np.where(std1 != 0, std1, 1e+4)
        mean2, std2 = np.mean(H, axis=0), np.std(H, axis=0)
        std2 = np.where(std2 != 0, std2, 1e+4)
        for i in range(len(test_ds)):
            test_ds[i].x = (test_ds[i].x - mean1) / std1
            test_ds[i].hls_attr[0] = (test_ds[i].hls_attr[0] - mean2) / std2

    ds_path = os.path.abspath('./DSE_ds/rdc/' + benchmark)
    test_ds_cp = torch.load(ds_path)
    np.random.RandomState(seed=128).shuffle(test_ds)
    np.random.RandomState(seed=128).shuffle(test_ds_cp)

    model = load(model_type)
    pareto_front = []
    start = time.perf_counter()
    design_ds = copy.deepcopy(test_ds)
    design_ds_cp = copy.deepcopy(test_ds_cp)

    while True:
        candidate = copy.deepcopy(design_ds[0])
        candidate_cp = copy.deepcopy(design_ds_cp[0])

        temp = []
        temp_cp = []
        is_dominated = False
        del design_ds[0]
        del design_ds_cp[0]

        for  i in range(len(design_ds)):
            pred1,pred2 = infer_dominances(model_type,model,candidate,design_ds[i],candidate_cp,design_ds_cp[i],nodes,inference_time)
            if pred1 == 0:
                is_dominated = True
            if pred2 != 0:
                temp.append(design_ds[i])
                temp_cp.append(design_ds_cp[i])
        if not is_dominated:
            pareto_front.append(candidate)
        design_ds = copy.deepcopy(temp)
        design_ds_cp = copy.deepcopy(temp_cp)
        if len(design_ds) == 0:
            break
    end = time.perf_counter()
    adrs_evolution = ADRS(test_ds,pareto_front)
    return adrs_evolution,end-start


