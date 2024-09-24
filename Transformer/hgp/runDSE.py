import numpy as np
import os
import pandas as pd
from DSE import DSE
k=0
dataset_dir = os.path.abspath('DSE_ds/std')
dataset = os.listdir(dataset_dir)
nodes  = []
inference_time = []
for ds in dataset:
    model_types = ["hgp"]
    res = []
    df = pd.DataFrame(columns=model_types, dtype=float)
    for model_type in model_types:
        print("\033[31m %s GNN \033[0m" % model_type)
        adrses,runtime = DSE(model_type,ds,nodes,inference_time)
        nodes = np.array(nodes)
        inference_time = np.array(inference_time)
        np.save("./extension/nodes_size_dse"+str(k)+".npy", nodes)
        np.save("./extension/inference_time_dse"+str(k)+".npy", inference_time)
        k += 1
        nodes = []
        inference_time = []
        print("\033[31m %s \033[0m" % str(ds[0:ds.find(".pt")]))
        print("\033[34m adrs evolution\033[0m")
        print(adrses)
        print("\033[34m final adrs:\033[0m %f , \033[34m # of synthesis:\033[0m %d, \033[34m runtime:\033[0m %f" % (adrses[-1], len(adrses), runtime))
        res.append([adrses,runtime])
        df[model_type] = [adrses[-1], len(adrses), runtime]
















