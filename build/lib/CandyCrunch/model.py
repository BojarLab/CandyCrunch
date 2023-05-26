import torch
import numpy as np
import random
from torch import flatten
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


def remove_low_intensity_peaks(array, removal_threshold, removal_percentage):
  candidate_indices = np.where(np.logical_and(array > 0.0001, array <= removal_threshold))[0]
  indices_to_remove = np.random.choice(candidate_indices, round(removal_percentage*len(candidate_indices)))
  array_copy = np.copy(array)
  array_copy[indices_to_remove] = 0
  return array_copy


def peak_intensity_jitter(array, augment_intensity):
  return array * np.random.uniform(1-augment_intensity, 1+augment_intensity, len(array)).astype(np.float32)


def new_peak_addition(array, n_noise_peaks, max_noise_intensity):
  idx_noise_peaks = np.random.choice(np.argwhere(array == 0)[0], n_noise_peaks)
  new_values = max_noise_intensity * np.random.random(len(idx_noise_peaks))
  noisy_array = np.copy(array)
  noisy_array[idx_noise_peaks] = new_values
  return noisy_array


transform_mz = transforms.Compose([
  lambda x: remove_low_intensity_peaks(x, removal_threshold = 0.008, removal_percentage = 0.1),
  lambda x: peak_intensity_jitter(x, augment_intensity = 0.25),
  lambda x: new_peak_addition(x, n_noise_peaks = 10, max_noise_intensity = 0.005)
])


def precursor_jitter(prec):
  return prec + random.uniform(-0.5, 0.5)


transform_prec = transforms.Compose([
    lambda x: precursor_jitter(x)
])


def rt_jitter(RT):
  return max(0, RT + random.uniform(-0.1, 0.1))


transform_rt = transforms.Compose([
    lambda x: rt_jitter(x)
])


class SimpleDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, transform_mz = None, transform_prec = None, transform_rt = None,
               transform_prec_neg = None, transform_prec_pos = None):
    self.x = x
    self.y = y
    self.transform_mz = transform_mz
    self.transform_prec = transform_prec
    self.transform_prec_neg = transform_prec_neg
    self.transform_prec_pos = transform_prec_pos
    self.transform_rt = transform_rt

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    mz = self.x[index][0]
    if self.transform_mz:
      mz = self.transform_mz(mz)
    mz_r = self.x[index][1]
    prec = self.x[index][2]
    if self.transform_prec:
      prec = self.transform_prec(prec)
    glycan_type = self.x[index][3]
    RT = self.x[index][4]
    if self.transform_rt:
      RT = self.transform_rt(RT)
    mode = self.x[index][5]
    if any([self.transform_prec_neg, self.transform_prec_pos]):
      if mode == 0:
        prec = self.transform_prec_neg(prec)
      else:
        prec = self.transform_prec_pos(prec)
    lc = self.x[index][6]
    modification = self.x[index][7]
    trap = self.x[index][8]
    out = self.y[index]
    return torch.FloatTensor(mz), torch.FloatTensor(mz_r), torch.FloatTensor([prec]), torch.LongTensor([glycan_type]), torch.FloatTensor([RT]), torch.LongTensor([mode]), torch.LongTensor([lc]), torch.LongTensor([modification]), torch.LongTensor([trap]), torch.LongTensor([out])


class ResUnit(nn.Module):

    def __init__(self, in_channels, size = 3, dilation = 1, causal = False, in_ln = True):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine = True)
            self.ln1.weight.data.fill_(1.0)
        self.conv_in = nn.Conv1d(in_channels, in_channels//2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels//2, affine = True)
        self.ln2.weight.data.fill_(1.0)
        self.conv_dilated = nn.Conv1d(in_channels//2, in_channels//2, size, dilation = self.dilation,
                                      padding = ((dilation*(size-1)) if causal else (dilation*(size-1)//2)))
        self.ln3 = nn.InstanceNorm1d(in_channels//2, affine = True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels//2, in_channels, 1)

    def forward(self, inp):
        x = inp
        if self.in_ln:
            x = self.ln1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.functional.leaky_relu(self.ln2(self.conv_in(x)))  
        x = self.conv_dilated(x)
        if self.causal and self.size > 1:
            x = x[:, :, :-self.dilation*(self.size-1)]
        x = nn.functional.leaky_relu(self.ln3(x))
        x = self.conv_out(x)
        return x+inp


class CandyCrunch_CNN(torch.nn.Module):

  def __init__(self, input_dim, num_classes = 1, hidden_dim = 512):
    super(CandyCrunch_CNN, self).__init__()

    self.input_dim = input_dim

    self.mz_lin1 = torch.nn.Linear(input_dim, 2*hidden_dim)
    self.prec_lin1 = torch.nn.Linear(1, 24)
    self.rt_lin1 = torch.nn.Linear(1, 24)
    self.comb_lin1 = torch.nn.Linear(2*hidden_dim+24+24+24+24+24+24+24, 2*512)
    self.comb_lin2 = torch.nn.Linear(2*512, 2*256)
    self.comb_lin3 = torch.nn.Linear(2*256, num_classes)

    self.type_emb = torch.nn.Embedding(5, 24)
    self.mode_emb = torch.nn.Embedding(3, 24)
    self.lc_emb = torch.nn.Embedding(4, 24)
    self.modification_emb = torch.nn.Embedding(4, 24)
    self.trap_emb = torch.nn.Embedding(5, 24)

    self.conv1 = torch.nn.Conv1d(in_channels = 2, out_channels = 64, kernel_size = 1)
    self.res1 = ResUnit(64, size = 2, dilation = 1, causal = True)
    self.res2 = ResUnit(64, size = 2, dilation = 2, causal = True)
    self.res3 = ResUnit(64, size = 2, dilation = 4, causal = True)
    self.res4 = ResUnit(64, size = 2, dilation = 8, causal = True)
    self.res5 = ResUnit(64, size = 2, dilation = 16, causal = True)
    self.res6 = ResUnit(64, size = 2, dilation = 32, causal = True)
    self.maxpool1 = torch.nn.MaxPool1d(kernel_size = 20)
    self.fc1 = torch.nn.Linear(in_features = 6528, out_features = 1024)

    self.mz_bn1 = torch.nn.LayerNorm(2*hidden_dim)
    self.prec_bn1 = torch.nn.LayerNorm(24)
    self.rt_bn1 = torch.nn.LayerNorm(24)
    self.comb_bn1 = torch.nn.LayerNorm(2*512)
    self.comb_bn2 = torch.nn.LayerNorm(2*256)
    self.mz_act1 = torch.nn.LeakyReLU()
    self.prec_act1 = torch.nn.LeakyReLU()
    self.rt_act1 = torch.nn.LeakyReLU()
    self.comb_act1 = torch.nn.LeakyReLU()
    self.comb_act2 = torch.nn.LeakyReLU()
    self.comb_dp1 = torch.nn.Dropout(0.2)
    self.comb_dp2 = torch.nn.Dropout(0.2)

  def forward(self, mz_list, precursor, glycan_type, rt, mode, lc, modification, trap, rep = False):
    glycan_type = self.type_emb(glycan_type).squeeze()
    mode = self.mode_emb(mode).squeeze()
    lc = self.lc_emb(lc).squeeze()
    modification = self.modification_emb(modification).squeeze()
    trap = self.trap_emb(trap).squeeze()
    precursor = self.prec_act1(self.prec_bn1(self.prec_lin1(precursor)))
    rt = self.rt_act1(self.rt_bn1(self.rt_lin1(rt)))
    mz = F.leaky_relu(self.conv1(mz_list))
    mz = self.res1(mz)
    mz = self.res2(mz)
    mz = self.res3(mz)
    mz = self.res4(mz)
    mz = self.res5(mz)
    mz = self.res6(mz)

    mz = self.maxpool1(mz)
    mz = F.leaky_relu(self.fc1(flatten(mz, start_dim = 1)))

    comb = torch.cat([mz, precursor, glycan_type, rt, mode, lc, modification, trap], dim = 1)
    comb = self.comb_dp1(self.comb_act1(self.comb_bn1(self.comb_lin1(comb))))
    comb_rep = self.comb_lin2(comb)
    comb = self.comb_dp2(self.comb_act2(self.comb_bn2(comb_rep)))
    comb = self.comb_lin3(comb)
    if rep:
      return comb, comb_rep
    else:
      return comb
