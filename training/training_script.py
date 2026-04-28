import pickle
import pandas as pd

from candycrunch.model import (SimpleDataset, CandyCrunch_CNN, transform_mz, transform_rt)
from glycowork.motif.annotate import annotate_dataset, get_k_saccharides
from glycowork.motif.processing import get_lib
from glycowork.motif.tokenization import get_stem_lib, glycan_to_composition
from training_utils import *
from sklearn.metrics import pairwise_distances
import warnings
# Suppress the specific sklearn runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")

def filter_data_exceptions(features, labels, valid_labels):
    """
    Filter out data points with:
    1. Labels not in valid_labels
    2. Glycans that cannot be processed by glycan_to_composition or other glycan functions

    Returns filtered features, labels, and the updated valid_labels list
    """
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] not in valid_labels:
            del labels[i]
            del features[i]
    problematic_glycans = set()
    for i in range(len(labels) - 1, -1, -1):
        glycan = labels[i]
        try:
            glycan_to_composition(glycan)
            from glycowork.motif.graph import glycan_to_nxGraph
            glycan_to_nxGraph(glycan)
        except (KeyError, IndexError, ValueError, AttributeError, Exception) as e:
            problematic_glycans.add(glycan)
            del labels[i]
            del features[i]
            print(f"  - Removed problematic glycan '{glycan}' due to error: {type(e).__name__}")
    if problematic_glycans:
        print(f"Removed {len(problematic_glycans)} unprocessable glycans: {problematic_glycans}")

    # Update valid_labels to exclude problematic glycans
    valid_labels = [g for g in valid_labels if g not in problematic_glycans]

    return features, labels, valid_labels

print("Reading data")
#Train and test data can be found on zenodo at https://doi.org/10.5281/zenodo.7940046
#Please modify the filepaths below to point to your downloaded files

with open("./prepared_datasets/X_train.pkl", "rb") as file:
  X_train = pickle.load(file)
with open("./prepared_datasets/X_test.pkl", "rb") as file:
  X_test = pickle.load(file)
with open("./prepared_datasets/y_train.pkl", "rb") as file:
  y_train = pickle.load(file)
with open("./prepared_datasets/y_test.pkl", "rb") as file:
  y_test = pickle.load(file)
with open("./prepared_datasets/glycans.pkl", "rb") as file:
  glycans = pickle.load(file)

print("Preprocessing data")
X_train, y_train, glycans = filter_data_exceptions(X_train, y_train, glycans)
X_test, y_test, glycans = filter_data_exceptions(X_test, y_test, glycans)


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
print(f"This is comp_vector_order for Zenodo dataset {comp_vector_order}")
glycan_comp_vect_map = {}
for glyc,glycomp in allowed_glycan_comps.items():
    comp_vect = np.zeros(len(comp_vector_order))
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
trainset = SimpleDataset(X_train, y_train, transform_mz=transform_mz, transform_rt=transform_rt)
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
embs = embs.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
dist = pairwise_distances(embs, metric='cosine')
dist = dist*1000*20
dist2 = torch.tensor(dist, requires_grad=True).to(device)
comps = [glycan_to_composition(k) for k in glycans]
comp_df = pd.DataFrame.from_dict(comps).fillna(0)
dist = pairwise_distances(comp_df, metric='cosine')
dist = dist*1000*50
dist3 = torch.tensor(dist, requires_grad=True).to(device)

print("Preparing the model")
model = CandyCrunch_CNN(2048, num_classes = len(glycans), input_precursor_dim = len(comp_vector_order))
model = model.apply(lambda module: init_weights(module, mode = 'kaiming'))
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model = model.to(device)

optimizer_ft, scheduler, criterion = training_setup(model, 0.0001, weight_decay = 2e-05,num_classes=len(set(glycans)))
primary_loss = Poly1CrossEntropyLoss(num_classes = len(glycans), epsilon = 1, reduction = 'mean').to(device)
criterion = custom_loss(primary_loss,dist2,dist3).to(device)

print("Start training")
model_ft = train_model(model, 'CandyCrunch_example_script',dataloaders, criterion, optimizer_ft, scheduler, glycans, num_epochs = 20,
                       patience = 12)
                       
torch.save(model_ft.state_dict(), f'./CandyCrunch_example_script.pt')
