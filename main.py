# pip install torch_snippets

from torch_snippets import *
from utils.utils import trn_tfms, val_tfms, save_model, load_model
from dataset import SiameseNetworkDataset
from model import SiameseNetwork, ContrastiveLoss
from training import device, train_batch, validate_batch
import sys
# import wandb
# wandb.init()

# dataset creation
trn_ds=SiameseNetworkDataset(folder="./data/faces/training/", transform=trn_tfms)
val_ds=SiameseNetworkDataset(folder="./data/faces/testing/",transform=val_tfms)

# Creating data loader
trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=8)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=8)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(),lr = 3e-4)
try:
    PATH = sys.argv[2]
    load_model(model,optimizer,PATH=PATH)
except Exception as e:
    print("No pretrained model provided or model failed to load")


def train(n_epochs=200):
    log = Report(n_epochs)
    for epoch in range(n_epochs):
        N = len(trn_dl)
        for i, data in enumerate(trn_dl):
            loss, acc = train_batch(model, data, optimizer, \
                                    criterion)
            log.record(epoch+(1+i)/N,trn_loss=loss,trn_acc=acc, \
                       end='\r')
            N = len(val_dl)
            for i, data in enumerate(val_dl):
                loss, acc = validate_batch(model, data, \
                                           criterion)
                log.record(epoch+(1+i)/N,val_loss=loss,val_acc=acc, \
                           end='\r')
                #if (epoch+1)%20==0: log.report_avgs(epoch+1)
        if epoch%10==0 and epoch !=0:
            save_model(loss=loss,model=model,epoch=epoch,optimizer=optimizer)

    print(f"This is type of loss---------------------->{loss}")
    print(f"This is type of model---------------------->{model}")
    print(f"This is type of optimizer---------------------->{optimizer}")
    save_model(loss=loss,model=model,epoch=epoch,optimizer=optimizer)
    return log

if __name__== '__main__':
    log = train(200)
    # log.plot_epochs(['trn_loss','val_loss'])
    # log.plot_epochs(['trn_acc','val_acc'])
    print("Done")