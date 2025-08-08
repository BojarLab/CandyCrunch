import pickle
import pandas as pd

from candycrunch.model import (SimpleDataset,CandyCrunch_CNN,transform_mz,transform_prec,transform_rt)
from glycowork.motif.annotate import annotate_dataset,get_k_saccharides
from glycowork.motif.processing import get_lib
from glycowork.motif.tokenization import get_stem_lib,glycan_to_composition
from training_utils import *
from sklearn.metrics import pairwise_distances

def filter_data_exceptions(features, labels, valid_labels):
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] not in valid_labels:
            del labels[i]
            del features[i]

print("Reading data")
#Train and test data can be found on zenodo at https://doi.org/10.5281/zenodo.7940046
#Please modify the filepaths below to point to your downloaded files

with open("../../../Downloads/X_train_CC2_240110.pkl", "rb") as file:
  X_train = pickle.load(file)
with open("../../../Downloads/X_test_CC2_240110.pkl", "rb") as file:
  X_test = pickle.load(file)
with open("../../../Downloads/y_train_CC2_240110.pkl", "rb") as file:
  y_train = pickle.load(file)
with open("../../../Downloads/y_test_CC2_240110.pkl", "rb") as file:
  y_test = pickle.load(file)
with open("../../../Downloads/glycans_240110.pkl", "rb") as file:
  glycans = pickle.load(file) 

print("Preprocessing data")
filter_data_exceptions(X_train, y_train, glycans)
filter_data_exceptions(X_test, y_test, glycans)

disallowed_glycans = []
allowed_glycan_comps = {}
for glyc in glycans:
    try:
        glycomp = glycan_to_composition(glyc)
        allowed_glycan_comps[glyc] = glycomp
    except KeyError:
        disallowed_glycans.append(glyc)
comp_vector_order = list(set(x for y in allowed_glycan_comps.values() for x in y))
comp_vector_order = sorted(comp_vector_order,key=lambda x:x.lower())

glycan_comp_vect_map = {}
for glyc,glycomp in allowed_glycan_comps.items():
    comp_vect = np.zeros(12)
    for mono,counts in glycomp.items():
        comp_vect[comp_vector_order.index(mono)] = counts
    glycan_comp_vect_map[glyc] = comp_vect

X_train = [t[:2] + (glycan_comp_vect_map[gt],) + t[3:]
           for t, gt in zip(X_train, y_train)]
X_test = [t[:2] + (glycan_comp_vect_map[gt],) + t[3:]
          for t, gt in zip(X_test, y_test)]

y_train = [glycans.index(c) for c in y_train]
y_test = [glycans.index(c) for c in y_test]


print("Preparing dataloaders")
trainset = SimpleDataset(X_train, y_train, transform_mz=transform_mz, transform_prec=None, transform_rt=transform_rt)
valset = SimpleDataset(X_test, y_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 256, shuffle = True, drop_last = True, pin_memory = True,
                                          num_workers = 0)
valloader = torch.utils.data.DataLoader(valset, batch_size = 256, shuffle = False, drop_last = True, pin_memory = True,
                                        num_workers = 0)
dataloaders = {'train':trainloader, 'val':valloader}


print("Calculating composition/structure distance for loss")
embs = annotate_dataset(glycans, feature_set = ['exhaustive'], condense = True)
embs2 = get_k_saccharides(glycans,size = 3)
embs2.index = glycans
embs = pd.concat([embs, embs2], axis=1)
cols = embs.columns.values.tolist()
embs[cols] = embs[cols].map(float)
dist = pairwise_distances(embs, metric='cosine')
dist = dist*1000*20
dist2 = torch.tensor(dist, requires_grad=True).to(device)
comps = [glycan_to_composition(k) for k in glycans]
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

optimizer_ft, scheduler, criterion = training_setup(model, 0.0001, weight_decay = 2e-05,num_classes=len(set(glycans)))
primary_loss = Poly1CrossEntropyLoss(num_classes = len(glycans), epsilon = 1, reduction = 'mean').to(device)
criterion = custom_loss(primary_loss,dist2,dist3).to(device)

print("Start training")
model_ft = train_model(model, 'CandyCrunch_example_script',dataloaders, criterion, optimizer_ft, scheduler, glycans, num_epochs = 200,
                       patience = 12)
                       
torch.save(model_ft.state_dict(), f'./CandyCrunch_example_script.pt')
