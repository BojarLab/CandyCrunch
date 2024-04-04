import pickle
import pandas as pd

from CandyCrunch.model import (SimpleDataset,CandyCrunch_CNN,transform_mz,transform_prec,transform_rt)
from glycowork.motif.annotate import annotate_dataset,get_k_saccharides
from glycowork.motif.processing import get_lib
from glycowork.motif.tokenization import get_stem_lib,glycan_to_composition
from training_utils import *
from sklearn.metrics import pairwise_distances

print("Reading data")
#Train and test data can be found on zenodo at https://doi.org/10.5281/zenodo.7940046
#Please modify the filepaths below to point to your downloaded files
with open("X_train.pkl", "rb") as file:
  X_train = pickle.load(file)
with open("X_test.pkl", "rb") as file:
  X_test = pickle.load(file)
with open("y_train.pkl", "rb") as file:
  y_train = pickle.load(file)
with open("y_test.pkl", "rb") as file:
  y_test = pickle.load(file)
with open("glycans.pkl", "rb") as file:
  glycans = pickle.load(file) 

print("Preparing dataloaders")
trainset = SimpleDataset(X_train, y_train, transform_mz=transform_mz, transform_prec=transform_prec, transform_rt=transform_rt)
valset = SimpleDataset(X_test, y_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 256, shuffle = True, drop_last = True, pin_memory = True,
                                          num_workers = 4)
valloader = torch.utils.data.DataLoader(valset, batch_size = 256, shuffle = False, drop_last = True, pin_memory = True,
                                        num_workers = 4)
dataloaders = {'train':trainloader, 'val':valloader}
lib = get_lib(glycans)
stem_lib = get_stem_lib(lib)

print("Calculating composition/structure distance for loss")
embs = annotate_dataset(glycans, feature_set = ['exhaustive'], condense = True)
embs2 = get_k_saccharides(glycans,size = 3, libr=lib)
embs2.index = glycans
embs = pd.concat([embs, embs2], axis=1)
cols = embs.columns.values.tolist()
embs[cols] = embs[cols].applymap(float)
dist = pairwise_distances(embs, metric='cosine')
dist = dist*1000*20
dist2 = torch.tensor(dist, requires_grad=True).to(device)
comps = [glycan_to_composition(k, stem_libr = stem_lib) for k in glycans]
comp_df = pd.DataFrame.from_dict(comps).fillna(0)
dist = pairwise_distances(comp_df, metric='cosine')
dist = dist*1000*50
dist3 = torch.tensor(dist, requires_grad=True).to(device)

print("Preparing the model")
model = CandyCrunch_CNN(2048, num_classes = len(glycans))
model = model.apply(lambda module: init_weights(module, mode = 'kaiming'))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model = model.to(device)

optimizer_ft, scheduler, criterion = training_setup(model, 0.0001, weight_decay = 2e-05)
primary_loss = Poly1CrossEntropyLoss(num_classes = len(glycans), epsilon = 1, reduction = 'mean').to(device)
criterion = custom_loss(primary_loss,dist2,dist3).to(device)

print("Start training")
model_ft = train_model(model, 'CandyCrunch_example_script',dataloaders, criterion, optimizer_ft, scheduler, glycans, num_epochs = 200,
                       patience = 12)
                       
torch.save(model_ft.state_dict(), f'./CandyCrunch_example_script.pt')
