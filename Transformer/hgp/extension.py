import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# sns.set_theme(style="whitegrid",font='New Roman',font_scale=1.4)
sns.set_theme(font='Times New Roman',font_scale=1.4)
path = "extension"
# sns.set_theme(style = "whitegrid",font='Times New Roman')
benchmark = ['md knn', 'stencil3d', 'fft stride', 'bfs bulk', 'viterbi']
all = pd.DataFrame(columns = ["Size","Inference", "benchmark"])
for i,name in enumerate(benchmark):
    feature_sizes = np.load("./"+path+"/nodes_size_dse"+str(i)+".npy",allow_pickle=True)
    inference_time = np.load("./"+path+"/inference_time_dse"+str(i)+".npy",allow_pickle=True)
    joint_size = [size[-1] for size in feature_sizes ]
    df = pd.DataFrame({"Size1 plus Size2":joint_size,"Inference Time":inference_time})
    df = df.groupby("Size1 plus Size2")["Inference Time"].mean().reset_index()
    df["benchmark"] = [name] * df.shape[0]
    df = df[::5]
    print(df.shape)
    all = pd.concat([all,df])
axis = []
plt.tight_layout()
# plt.tick_params(axis="both", which="major", direction="in", width=1, length=5, pad=5)

fig = sns.lmplot(x="Size1 plus Size2", y="Inference Time", hue="benchmark", col="benchmark", col_wrap=5,palette="Set1", data=all,sharey=False)

for i,name in  enumerate(benchmark):
    # if i!=1:
    #     continue
    X = all[all["benchmark"] == name]["Size1 plus Size2"]
    Y = all[all["benchmark"] == name]["Inference Time"]
    XX = sm.add_constant(X)
    model = sm.OLS(Y, XX).fit()
    # 查看模型结果
    print(model.rsquared)
    fig.axes[i].lines[0].set_data(X,model.fittedvalues)
    title = fig.axes[i].get_title()
    fig.axes[i].set_title(title + " (R2 = "+str(round(model.rsquared,2))+")")
fig.set_axis_labels('Input Size', 'Inference Time (s)')
bwith = 1
for i in range(len(benchmark)):
    fig.axes[i].tick_params(axis="both", which="major", direction="out", width=1.5, length=5, pad=5)
    fig.axes[i].spines['bottom'].set_linewidth(bwith)  # 图框下边
    fig.axes[i].spines['left'].set_linewidth(bwith)  # 图框左边
    fig.axes[i].spines['top'].set_linewidth(bwith)  # 图框上边
    fig.axes[i].spines['right'].set_linewidth(bwith)  # 图框右边

    fig.axes[i].spines['top'].set_color('black')
    fig.axes[i].spines['left'].set_color('black')

    fig.axes[i].spines['bottom'].set_color('black')
    fig.axes[i].spines['right'].set_color('black')

    fig.axes[i].spines['bottom'].set_visible(True)  # 图框下边
    fig.axes[i].spines['left'].set_visible(True) # 图框左边
    fig.axes[i].spines['top'].set_visible(True)  # 图框上边
    fig.axes[i].spines['right'].set_visible(True)
    fig.axes[i].xaxis.set_tick_params(which='both', bottom=True, top=False, direction='out')
    fig.axes[i].yaxis.set_tick_params(which='both', bottom=True, top=False, direction='out')


fig_path = "./"+path+"/extension.svg"

fig.savefig(fig_path, dpi = 500)


