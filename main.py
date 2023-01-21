# pip install torch_snippets

from torch_snippets import *
from utils.utils import trn_tfms, val_tfms
from dataset import SiameseNetworkDataset

import wandb
wandb.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset creation
trn_ds=SiameseNetworkDataset(folder="./data/faces/training/", transform=trn_tfms)
val_ds=SiameseNetworkDataset(folder="./data/faces/testing/",transform=val_tfms)

# Creating data loader
trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=8)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=8)

