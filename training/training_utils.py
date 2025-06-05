import copy
import json
import numpy as np
import time
import torch

import matplotlib.pyplot as plt
from candycrunch.analysis import glycan_to_graph_monos,mono_graph_to_nx,enumerate_k_graphs,mono_frag_to_string
from glycowork.ml.model_training import EarlyStopping, disable_running_stats, enable_running_stats, training_setup
import torch.nn.functional as F
from torchmetrics.functional import accuracy, matthews_corrcoef, mean_squared_error

device = "cpu"
if torch.cuda.is_available():
  device = "cuda:0"

   
class custom_loss(torch.nn.Module):
  def __init__(self, primary_loss, dist_sim, dist_comp, logit_norm = False, t = 1.0):
    super(custom_loss, self).__init__()
    self.primary_loss = primary_loss
    self.dist_sim = dist_sim
    self.dist_comp = dist_comp
    self.logit_norm = logit_norm
    self.t = t
  def forward(self, output, target):
    if self.logit_norm:
      norms = torch.norm(output, p=2, dim=-1, keepdim=True) + 1e-7
      output = torch.div(output, norms) / self.t
    loss2 = self.primary_loss(output, target)
    output = torch.nn.functional.softmax(output, dim=1)
    target_sim = self.dist_sim[target]
    loss_sim = output*target_sim
    target_comp = self.dist_comp[target]
    loss_comp = output*target_comp
    loss = loss_comp.mean() + loss_sim.mean() + loss2
    return loss
    
def init_weights(model, mode = 'sparse', sparsity = 0.1):
  """initializes linear layers of PyTorch model with a weight initialization\n
  | Arguments:
  | :-
  | model (Pytorch object): neural network (such as SweetNet) for analyzing glycans
  | mode (string): which initialization algorithm; choices are 'sparse','kaiming','xavier';default:'sparse'
  | sparsity (float): proportion of sparsity after initialization; default:0.1 / 10%
  """
  if type(model) == torch.nn.Linear:
    if mode == 'sparse':
      torch.nn.init.sparse_(model.weight, sparsity = sparsity)
    elif mode == 'kaiming':
      torch.nn.init.kaiming_uniform_(model.weight)
    elif mode == 'xavier':
      torch.nn.init.xavier_uniform_(model.weight)
    else:
      print("This initialization option is not supported.")

class Poly1CrossEntropyLoss(torch.nn.Module):
  def __init__(self,
               num_classes: int,
               epsilon: float = 1.0,
               reduction: str = "none",
               weight: torch.Tensor = None):
    """
    Create instance of Poly1CrossEntropyLoss
    :param num_classes:
    :param epsilon:
    :param reduction: one of none|sum|mean, apply reduction to final loss tensor
    :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
    """
    super(Poly1CrossEntropyLoss, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.reduction = reduction
    self.weight = weight
    return

  def forward(self, logits, labels):
    """
    Forward pass
    :param logits: tensor of shape [N, num_classes]
    :param labels: tensor of shape [N]
    :return: poly cross-entropy loss
    """
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                       dtype=logits.dtype)
    pt = torch.sum(labels_onehot * torch.nn.functional.softmax(logits, dim=-1), dim=-1)
    CE = torch.nn.functional.cross_entropy(input=logits,
                         target=labels,
                         reduction='none',
                         weight=self.weight,
                         label_smoothing = 0.1)
    poly1 = CE + self.epsilon * (1 - pt)
    if self.reduction == "mean":
      poly1 = poly1.mean()
    elif self.reduction == "sum":
      poly1 = poly1.sum()
    return poly1
        
        
def train_model(model, model_name, dataloaders, criterion, optimizer,
                scheduler, glycans, num_epochs = 25, patience = 50):
  """trains a deep learning model on predicting glycan properties\n
  | Arguments:
  | :-
  | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
  | dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
  | criterion (PyTorch object): PyTorch loss function
  | optimizer (PyTorch object): PyTorch optimizer
  | scheduler (PyTorch object): PyTorch learning rate decay
  | num_epochs (int): number of epochs for training; default:25
  | patience (int): number of epochs without improvement until early stop; default:50\n
  | Returns:
  | :-
  | Returns the best model seen during training
  """
  since = time.time()
  early_stopping = EarlyStopping(patience = patience, verbose = True)
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0
  epoch_topk = 0
  epoch_topk10 = 0
  best_acc = 0.0
  val_losses = []
  val_acc = []
  train_losses = []
  train_acc = []
  losses_dict = {}
  losses_dict['train'] = {}
  losses_dict['val'] = {}
  losses_dict['time'] = []
  start = time.time_ns()
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-'*10)
    
    for phase in ['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()
        
      running_loss = []
      running_acc = []
      running_mcc = []
      running_topk = []
      running_topk10 = []
      for data in dataloaders[phase]:
        mz_list, mz_remainder, precursor, glycan_type, rt, mode_in, lc, modification, trap, y = data
        mz_list = torch.stack([mz_list, mz_remainder], dim=1)
        mz_list = mz_list.to(device)
        precursor = precursor.to(device)
        glycan_type = glycan_type.to(device)
        rt = rt.to(device)
        mode_in = mode_in.to(device)
        lc = lc.to(device)
        modification = modification.to(device)
        trap = trap.to(device)
        y = y.squeeze().to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(phase == 'train'):
          #first forward pass
          enable_running_stats(model)
          pred = model(mz_list, precursor, glycan_type, rt, mode_in, lc, modification, trap)
          loss = criterion(pred, y)

          if phase == 'train':
            loss.backward()
            optimizer.first_step(zero_grad = True)
            #second forward pass
            disable_running_stats(model)
            criterion(model(mz_list, precursor, glycan_type, rt, mode_in, lc, modification, trap), y).backward()
            optimizer.second_step(zero_grad = True)

        #collecting relevant metrics            
        running_loss.append(loss.item())
        running_acc.append(accuracy(pred, y, task="multiclass", num_classes=len(glycans)))
        running_topk.append(accuracy(pred, y, task="multiclass", num_classes=len(glycans),top_k=5))
        running_topk10.append(accuracy(pred, y, task="multiclass", num_classes=len(glycans), top_k=10))

      #averaging metrics at end of epoch  
      epoch_loss = np.mean(running_loss)
      epoch_acc = torch.mean(torch.stack(running_acc))
      epoch_topk = torch.mean(torch.stack(running_topk))
      epoch_topk10 = torch.mean(torch.stack(running_topk10))
      print('{} Loss: {:.4f} Accuracy: {:.4f} Top-5 Accuracy: {:.4f} Top-10 Accuracy: {:.4f}'.format(
          phase, epoch_loss, epoch_acc, epoch_topk, epoch_topk10))
      losses_dict[phase][epoch] = epoch_loss

      #keep best model state_dict
      if phase == 'val' and epoch_loss <= best_loss:
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())
      if phase == 'val' and epoch_acc > best_acc:
          best_acc = epoch_acc
      if phase == 'val':
        val_losses.append(epoch_loss)
        val_acc.append(epoch_acc.item())
        #check Early Stopping & adjust learning rate if needed
        early_stopping(epoch_loss, model)
        scheduler.step(epoch_loss)
      if phase == 'train':
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc.item())
      torch.cuda.empty_cache()
        
    if early_stopping.early_stop:
      print("Early stopping")
      break
    print()
    print(time.time_ns()-start)
    losses_dict['time'].append(time.time_ns()-start)  
  losses_json = json.dumps(losses_dict)
  f = open(f'{model_name}_metrics.json',"w")
  f.write(losses_json)
  f.close()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))

  ## plot loss & score over the course of training 
  fig, ax = plt.subplots(nrows = 2, ncols = 1) 
  plt.subplot(2, 1, 1)
  plt.plot(range(epoch+1), val_losses, label='Validation')  
  plt.plot(range(epoch+1), train_losses, label='Training')
  plt.title('Model Training')
  plt.ylabel('Validation Loss')
  #plt.legend(['Validation Loss'],loc = 'best')
  plt.legend() 

  plt.subplot(2, 1, 2)
  plt.plot(range(epoch+1), val_acc, label='Validation')
  plt.plot(range(epoch+1), train_acc, label='Training')
  plt.xlabel('Number of Epochs')
  plt.ylabel('Validation Accuracy')
 #plt.legend(['Validation Accuracy'], loc = 'best')
  plt.legend()
  plt.savefig(f'{model_name}_metric_plots.png')
  return model
