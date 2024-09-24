import matplotlib.pyplot as plt
from dataset_utils import *
import torch
torch.cuda.device_count()
from torch.optim import lr_scheduler
from Contrastive_Architecture import ModelConfig

target = ['lut', 'ff', 'dsp', 'bram', 'uram', 'srl', 'cp', 'power']
tar_idx = 3
jknFlag = 0



def train(model, train_loader,optimizer,device):
    model.train()
    total_cls_loss = 0
    total_acc = 0
    criterion  = torch.nn.CrossEntropyLoss()
    for _, data in enumerate(train_loader):
        input1,input2,target = data[0][0].to(device),data[0][1].to(device),data[1].to(device)
        optimizer.zero_grad()
        out = model(input1.x, input2.x, input1.edge_index, input2.edge_index,input1.batch,input2.batch,input1.hls_attr,input2.hls_attr,target)
        L = criterion(out,target)
        total_acc += ((target.float() == out.max(1)[1]).sum()).item()
        loss = L
        loss.backward()
        optimizer.step()
        total_cls_loss += L.item()

    total_cls_loss /= len(train_loader)
    total_acc /= (len(train_loader)*train_loader.batch_size)
    return total_cls_loss,total_acc


def test(model, loader, epoch,device):
    # print("------------------test---------------------------")
    criterion  = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        total_cls_loss = 0
        total_acc = 0
        for _, data in enumerate(loader):
            input1, input2, target = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            out = model(input1.x, input2.x, input1.edge_index, input2.edge_index, input1.batch, input2.batch,input1.hls_attr,input2.hls_attr)
            L = criterion(out,target)
            total_cls_loss += L.item()
            total_acc += ((target.float() == out.max(1)[1]).sum()).item()
        total_cls_loss /= len(loader)
        total_acc /= (len(loader)*loader.batch_size)
    return total_cls_loss,total_acc


def train_siamese_gnn(train_loader, valid_loader, fold,device):
    model_dir = os.path.abspath('./siamese_gnn/'+str(fold))


    data_ini = None
    for step, data in enumerate(train_loader):
        if step == 0:
            data_ini = data
            break
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelConfig(data_ini[0][0].num_features,device)

    LR = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=0.001)
    min_train_loss = 100
    min_test_loss = 100
    max_train_acc = -1
    max_test_acc = -1
    train_losses,test_losses = [],[]
    train_acces,test_acces = [],[]
    epoches = 100

    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    for epoch in range(epoches):
        train_loss,train_acc = train(model, train_loader,optimizer,device)
        test_loss,test_acc = test(model, valid_loader, epoch,device)
        print(train_acc,test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acces.append(train_acc)
        test_acces.append(test_acc)
        scheduler.step()

        save_train = False
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            save_train = True

        save_test = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_test = True


        checkpoint_1 = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_train_acc': max_train_acc
        }

        checkpoint_2 = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_train:
            torch.save(checkpoint_1, os.path.join(model_dir, 'cls_train.pt'))

        if save_test:
            torch.save(checkpoint_2, os.path.join(model_dir, 'cls_test.pt'))

    print('Min Train Loss: ' + str(min_train_loss))
    print('Min Test Loss: ' + str(min_test_loss))
    x=list(range(0,epoches))

    #plot
    plt.plot(x,train_losses,label="train")
    plt.plot(x,test_losses,label="test")
    plt.legend()
    plt.savefig("./loss.png")
    plt.show()
    plt.plot(x, train_acces, label="train")
    plt.plot(x, test_acces, label="test")
    plt.legend()
    plt.savefig("./acc.png")
    plt.show()
    return max_train_acc,max_test_acc





