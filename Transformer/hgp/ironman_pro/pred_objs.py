import os
import torch
jknFlag = 0
from hgp.ironman_pro.graph_model_cp import GCNNet

def objs_pred(data,device,type):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type == "dsp" or type == "bram":
        model_path = os.path.abspath('../baseline/ironman_pro/model/' + type + '_mae_checkpoint_test.pt')
    else:
        model_path = os.path.abspath('../baseline/ironman_pro/model/'+type+'_checkpoint_test.pt')
    params = torch.load(model_path, map_location=device)
    model = GCNNet(in_channels=data.num_features)
    model = model.to(device)
    model.load_state_dict(params['model'])
    model.eval()
    with torch.no_grad():
        num = data.x.shape[0]
        batch = torch.tensor([0 for i in range(num)],device=device)
        out = model(data.x, data.edge_index, batch)
    obj = out.view(-1).item()
    obj = round(obj, 3)
    return obj
