import torch
from torch import flatten
from torch import nn
import torch.nn.functional as F
# Training augments with the functions test-time augmentation uses, which live with the inference code (the built app predicts without PyTorch)
from candycrunch.inference import transform_mz, transform_rt


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform_mz = None, transform_rt = None):
        self.x = x
        self.y = y
        self.transform_mz = transform_mz
        self.transform_rt = transform_rt

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        mz = self.x[index][0]
        if self.transform_mz:
            mz = self.transform_mz(mz)
        mz_r = self.x[index][1]
        prec = self.x[index][2]
        glycan_type = self.x[index][3]
        RT = self.x[index][4]
        if self.transform_rt:
            RT = self.transform_rt(RT)
        mode = self.x[index][5]
        lc = self.x[index][6]
        modification = self.x[index][7]
        trap = self.x[index][8]
        out = self.y[index]
        return torch.FloatTensor(mz), torch.FloatTensor(mz_r), torch.FloatTensor(prec), torch.LongTensor(
            [glycan_type]), torch.FloatTensor([RT]), torch.LongTensor([mode]), torch.LongTensor([lc]), torch.LongTensor(
            [modification]), torch.LongTensor([trap]), torch.LongTensor([out])


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
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln2.weight.data.fill_(1.0)
        self.conv_dilated = nn.Conv1d(in_channels // 2, in_channels // 2, size, dilation = self.dilation,
                                      padding = ((dilation * (size - 1)) if causal else (dilation * (size - 1) // 2)))
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)

    def forward(self, inp):
        # Instance norms run as group norms with one channel per group (the same normalization, ~4x faster on CPU) and the convolutions as
        # batched matrix products, the causal one as one product per tap on the shifted input (conv1d is slow for dilated kernel-2 convolutions)
        n = inp.shape[0]
        x = F.group_norm(inp, inp.shape[1], self.ln1.weight, self.ln1.bias, self.ln1.eps) if self.in_ln else inp
        x = F.leaky_relu(x, inplace = self.in_ln)
        x = torch.baddbmm(self.conv_in.bias[None, :, None], self.conv_in.weight[:, :, 0].expand(n, -1, -1), x)
        x = F.leaky_relu(F.group_norm(x, x.shape[1], self.ln2.weight, self.ln2.bias, self.ln2.eps), inplace = True)
        if self.causal:
            w = self.conv_dilated.weight
            y = torch.baddbmm(self.conv_dilated.bias[None, :, None], w[:, :, -1].expand(n, -1, -1), x)
            for tap in range(self.size - 1):
                shift = (self.size - 1 - tap) * self.dilation
                y[:, :, shift:].baddbmm_(w[:, :, tap].expand(n, -1, -1), x[:, :, :-shift])
        else:
            y = self.conv_dilated(x)
        x = F.leaky_relu(F.group_norm(y, y.shape[1], self.ln3.weight, self.ln3.bias, self.ln3.eps), inplace = True)
        return torch.baddbmm(self.conv_out.bias[None, :, None], self.conv_out.weight[:, :, 0].expand(n, -1, -1), x).add_(inp)


class CandyCrunch_CNN(torch.nn.Module):

    def __init__(self, input_dim, num_classes = 1, hidden_dim = 512, input_precursor_dim = None):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim

        self.mz_lin1 = torch.nn.Linear(input_dim, 2 * hidden_dim)  # not used
        self.prec_lin1 = torch.nn.Linear(input_precursor_dim, 24)
        self.rt_lin1 = torch.nn.Linear(1, 24)
        self.comb_lin1 = torch.nn.Linear(2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512)
        self.comb_lin2 = torch.nn.Linear(2 * 512, 2 * 256)
        self.comb_lin3 = torch.nn.Linear(2 * 256, num_classes)

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

        self.mz_bn1 = torch.nn.LayerNorm(2 * hidden_dim)  # not used
        self.prec_bn1 = torch.nn.LayerNorm(24)
        self.rt_bn1 = torch.nn.LayerNorm(24)
        self.comb_bn1 = torch.nn.LayerNorm(2 * 512)
        self.comb_bn2 = torch.nn.LayerNorm(2 * 256)
        self.mz_act1 = torch.nn.LeakyReLU()  # not used
        self.prec_act1 = torch.nn.LeakyReLU()
        self.rt_act1 = torch.nn.LeakyReLU()
        self.comb_act1 = torch.nn.LeakyReLU()
        self.comb_act2 = torch.nn.LeakyReLU()
        self.comb_dp1 = torch.nn.Dropout(0.2)
        self.comb_dp2 = torch.nn.Dropout(0.2)

    def forward(self, mz_list, precursor, glycan_type, rt, mode, lc, modification, trap, rep = False):
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_act1(self.prec_bn1(self.prec_lin1(precursor)))
        rt = self.rt_act1(self.rt_bn1(self.rt_lin1(rt)))
        # Convolutional trunk runs per 16 spectra (per-spectrum results are identical): full-batch activations are
        # >100 MB each and get freshly page-faulted on every layer, whereas chunk-sized ones are recycled by the allocator
        mz_chunks = []
        for chunk in mz_list.split(16):
            mz = F.leaky_relu(self.conv1(chunk), inplace = True)
            mz = self.res1(mz)
            mz = self.res2(mz)
            mz = self.res3(mz)
            mz = self.res4(mz)
            mz = self.res5(mz)
            mz = self.res6(mz)
            mz_chunks.append(self.maxpool1(mz))
        mz = F.leaky_relu(self.fc1(flatten(torch.cat(mz_chunks), start_dim = 1)))

        comb = torch.cat([mz, precursor, glycan_type, rt, mode, lc, modification, trap], dim = 1)
        comb = self.comb_dp1(self.comb_act1(self.comb_bn1(self.comb_lin1(comb))))
        comb_rep = self.comb_lin2(comb)
        comb = self.comb_dp2(self.comb_act2(self.comb_bn2(comb_rep)))
        comb = self.comb_lin3(comb)
        if rep:
            return comb, comb_rep
        return comb

class RMSNorm(torch.nn.Module):

    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * x.pow(2).mean(dim = -1, keepdim = True).add(self.eps).rsqrt() * self.weight


def make_norm(norm_type, dim):
    if norm_type == 'layer':
        return nn.LayerNorm(dim)
    if norm_type == 'rms':
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type={norm_type!r}. Use 'layer' or 'rms'.")


ACTIVATIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leaky_relu': nn.LeakyReLU, 'silu': nn.SiLU, 'elu': nn.ELU}


def make_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f'Unknown activation={name!r}.')
    return ACTIVATIONS[name]()


# Glycan class codes, as the models take them: O 0, N 1, free/lipid 2 (prediction.CLASS_CODES); pretraining also uses 3 for spectra whose candidate
# compositions span classes. Fixed routing by glycan class has one classifier expert per class code
GLYCAN_CLASS_EXPERTS = 3
# The values of each instrument code, in code order, as build_training_pickles.py and process_for_inference code them (the last name of each: any other
# value). Fixed routing by instrument has one classifier expert per instrument of the training data
INSTRUMENT_CODES = {'mode': ('positive', 'negative'), 'lc': ('PGC', 'C18', 'other'), 'modification': ('reduced', 'permethylated', 'other'),
                    'trap': ('linear', 'orbitrap', 'amazon', 'other')}


class MoEClassifier(nn.Module):
    """Mixture-of-experts classifier, routed once per spectrum: logits = shared(h) (only with shared_expert) + the sum over the router's top_k experts
    of gate_e * expert_e(h), the gates the router's softmax renormalized over those top_k. Each expert is a linear layer, or with hidden_dim an MLP
    (activation a name, as make_activation takes it). top_k = num_experts is dense gating (the balancing loss is then constant at 1). shared_expert adds
    an expert every spectrum uses (DeepSeekMoE's shared experts), so each class's weights there learn from all of its spectra. fixed_routing replaces
    the router by expert indices given to forward (e.g. the glycan class): one expert per spectrum and no balancing loss; a spectrum whose index names
    no expert gets the mean of all experts"""

    def __init__(self, input_dim, num_classes, num_experts = 4, top_k = 2, hidden_dim = None, dropout = 0.0, activation = 'gelu', shared_expert = False,
                 fixed_routing = False):
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError('top_k must be between 1 and num_experts.')
        if fixed_routing and top_k != 1:
            raise ValueError('Fixed routing sends each spectrum to one expert; use top_k=1.')
        self.input_dim, self.num_classes, self.num_experts, self.top_k, self.fixed_routing = input_dim, num_classes, num_experts, top_k, fixed_routing
        self.last_aux_loss = self.last_gate_probs = self.last_top_indices = self.last_top_weights = None
        self.router = None if fixed_routing else nn.Linear(input_dim, num_experts)
        make_expert = lambda: nn.Linear(input_dim, num_classes) if hidden_dim is None else nn.Sequential(
            nn.Linear(input_dim, hidden_dim), make_activation(activation), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.experts = nn.ModuleList([make_expert() for _ in range(num_experts)])
        self.shared_expert = make_expert() if shared_expert else None

    def load_balancing_loss(self, gate_probs, top_indices):
        """Switch-style balancing loss from the gate probabilities [B, E] and the chosen experts [B, K]"""
        selected = F.one_hot(top_indices, num_classes = self.num_experts).float().sum(dim = 1)
        return self.num_experts * torch.sum(gate_probs.mean(dim = 0) * selected.mean(dim = 0) / self.top_k)

    def fixed_gates(self, x, expert_indices):
        """Fixed routing: gate probabilities [B, E], expert index [B, 1] and weight [B, 1] of each spectrum. A spectrum whose index names no expert gets
        index -1, weight 0 and uniform gate probabilities (forward gives it the mean of all experts)"""
        expert_indices = expert_indices.reshape(-1).to(device = x.device, dtype = torch.long)
        if expert_indices.numel() != x.size(0):
            raise ValueError('expert_indices needs one expert index per spectrum.')
        known = (expert_indices >= 0) & (expert_indices < self.num_experts)
        one_hot = F.one_hot(expert_indices.clamp(0, self.num_experts - 1), num_classes = self.num_experts).to(x.dtype)
        gate_probs = torch.where(known.unsqueeze(1), one_hot, torch.full_like(one_hot, 1.0 / self.num_experts))
        top_indices = torch.where(known, expert_indices, torch.full_like(expert_indices, -1)).unsqueeze(1)
        return gate_probs, top_indices, known.to(x.dtype).unsqueeze(1)

    def forward(self, x, return_routing = False, return_selected_logits = True, expert_indices = None):
        """Logits [B, num_classes] of x [B, input_dim] (expert_indices [B]: the expert of each spectrum, needed with fixed routing, ignored otherwise);
        with return_routing also a dictionary of the routing"""
        if self.fixed_routing:
            if expert_indices is None:
                raise ValueError('A classifier with fixed routing needs expert_indices.')
            gate_probs, top_indices, top_weights = self.fixed_gates(x, expert_indices)
            self.last_aux_loss = x.new_zeros(())
        else:
            gate_probs = torch.softmax(self.router(x), dim = -1)
            top_weights, top_indices = torch.topk(gate_probs, k = self.top_k, dim = -1)
            top_weights = top_weights / top_weights.sum(dim = -1, keepdim = True).clamp_min(1e-9)
            self.last_aux_loss = self.load_balancing_loss(gate_probs, top_indices)
        self.last_gate_probs, self.last_top_indices, self.last_top_weights = gate_probs.detach(), top_indices.detach(), top_weights.detach()
        # Each expert runs only on the spectra routed to it, so no [B, E, C] logits are made
        logits = x.new_zeros(x.size(0), self.num_classes)
        selected_logits = x.new_zeros(x.size(0), self.top_k, self.num_classes) if return_routing and return_selected_logits else None
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = top_indices == expert_idx
            if not expert_mask.any():
                continue
            batch_indices, slot_indices = expert_mask.nonzero(as_tuple = True)
            expert_output = expert(x.index_select(0, batch_indices))
            logits.index_add_(0, batch_indices, expert_output * top_weights[batch_indices, slot_indices].unsqueeze(-1))
            if selected_logits is not None:
                selected_logits[batch_indices, slot_indices] = expert_output
        if self.fixed_routing:
            # Spectra whose index names no expert (e.g. the mixed glycan class of pretraining spectra) get the mean of all experts
            unrouted = (top_indices[:, 0] < 0).nonzero(as_tuple = True)[0]
            if unrouted.numel():
                expert_input = x.index_select(0, unrouted)
                mean_output = torch.stack([expert(expert_input) for expert in self.experts]).mean(dim = 0)
                logits.index_add_(0, unrouted, mean_output)
                if selected_logits is not None:
                    selected_logits[unrouted, 0] = mean_output
        if self.shared_expert is not None:
            logits = logits + self.shared_expert(x)
        if not return_routing:
            return logits
        routing = {'gate_probs': gate_probs, 'top_indices': top_indices, 'top_weights': top_weights,
                   'expert_batch_sizes': torch.bincount(top_indices[top_indices >= 0].reshape(-1), minlength = self.num_experts)}
        if return_selected_logits:
            routing['selected_logits'] = selected_logits
        return logits, routing

    def get_aux_loss(self):
        return self.last_aux_loss


class ResUnit2(nn.Module):
    """ResUnit with a choice of activation and an optional squeeze-and-excitation of its output channels"""

    def __init__(self, in_channels, size = 3, dilation = 1, causal = False, in_ln = True, se = False, se_reduction = 8, encoder_activation = 'leaky_relu'):
        super().__init__()
        self.size, self.dilation, self.causal, self.in_ln, self.se = size, dilation, causal, in_ln, se
        self.activation = make_activation(encoder_activation)
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine = True)
            self.ln1.weight.data.fill_(1.0)
        # Bottleneck: 1x1 convolution to half the channels, dilated convolution, 1x1 convolution back
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln2.weight.data.fill_(1.0)
        padding = dilation * (size - 1) if causal else dilation * (size - 1) // 2
        self.conv_dilated = nn.Conv1d(in_channels // 2, in_channels // 2, size, dilation = dilation, padding = padding)
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine = True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)
        if self.se:
            se_hidden = max(in_channels // se_reduction, 1)
            self.se_fc1 = nn.Conv1d(in_channels, se_hidden, 1)
            self.se_fc2 = nn.Conv1d(se_hidden, in_channels, 1)

    def forward(self, inp):
        x = self.activation(self.ln1(inp) if self.in_ln else inp)
        x = self.conv_dilated(self.activation(self.ln2(self.conv_in(x))))
        if self.causal and self.size > 1:
            x = x[:, :, :-self.dilation * (self.size - 1)]
        x = self.conv_out(self.activation(self.ln3(x)))
        if self.se:
            x = x * torch.sigmoid(self.se_fc2(self.activation(self.se_fc1(x.mean(dim = 2, keepdim = True)))))
        return x + inp


class CandyCrunch_CNN2(nn.Module):
    """CandyCrunch_CNN with ResUnit2 blocks, normalized fully connected layers and an optional mixture-of-experts classifier (MoEClassifier).
    Fixed routing (classifier_fixed_routing) sends each spectrum to one classifier expert, chosen by classifier_routing_key: 'glycan_class', the expert of
    its glycan class code (O 0, N 1, free/lipid 2), with classifier_num_experts = 3, or 'instrument', the expert of its (mode, lc, modification, trap)
    codes, looked up in a table built from classifier_instrument_experts, the codes of each expert's instrument in expert order (training_script.py
    takes the instruments of the training data; checkpoints keep the list in model_kwargs). A spectrum whose glycan class or instrument has no expert
    gets the mean of all experts (and the shared expert, if any)"""

    def __init__(self, input_dim, num_classes = 1, hidden_dim = 1024, input_precursor_dim = None, dropout = 0.2, se = None, se_reduction = None,
                 norm_type = None, activation = None, encoder_activation = None, classifier_moe = False, classifier_num_experts = 4, classifier_top_k = 2,
                 classifier_expert_hidden_dim = None, classifier_shared_expert = False, classifier_fixed_routing = False,
                 classifier_routing_key = 'glycan_class', classifier_instrument_experts = None):
        super().__init__()
        if (classifier_shared_expert or classifier_fixed_routing) and not classifier_moe:
            raise ValueError('classifier_shared_expert and classifier_fixed_routing need classifier_moe=True.')
        if classifier_routing_key not in ('glycan_class', 'instrument'):
            raise ValueError(f"classifier_routing_key is 'glycan_class' or 'instrument', not {classifier_routing_key!r}.")
        if classifier_fixed_routing and classifier_routing_key == 'glycan_class' and classifier_num_experts != GLYCAN_CLASS_EXPERTS:
            # With more experts, the ones past the class codes would never be trained
            raise ValueError(f'Fixed routing sends each spectrum to the expert of its glycan class (O, N, free/lipid), so it needs '
                             f'classifier_num_experts={GLYCAN_CLASS_EXPERTS}.')
        self.input_dim, self.classifier_moe = input_dim, classifier_moe
        self.classifier_fixed_routing, self.classifier_routing_key = bool(classifier_fixed_routing), classifier_routing_key
        self.type_emb = nn.Embedding(5, 24)
        self.mode_emb = nn.Embedding(3, 24)
        self.lc_emb = nn.Embedding(4, 24)
        self.modification_emb = nn.Embedding(4, 24)
        self.trap_emb = nn.Embedding(5, 24)
        if self.classifier_fixed_routing and classifier_routing_key == 'instrument':
            # The checkpoint keeps the instruments in model_kwargs, so the state dict does not need the table
            self.register_buffer('instrument_experts', self.instrument_lookup_table(classifier_instrument_experts, classifier_num_experts), persistent = False)
        self.prec_block = nn.Sequential(nn.Linear(input_precursor_dim, 24), make_norm(norm_type, 24), make_activation(activation))
        self.rt_block = nn.Sequential(nn.Linear(1, 24), make_norm(norm_type, 24), make_activation(activation))
        self.res_block = nn.Sequential(nn.Conv1d(in_channels = 2, out_channels = 64, kernel_size = 1), nn.LeakyReLU(),
                                       *[ResUnit2(64, size = 3, dilation = 2 ** i, se = se, se_reduction = se_reduction, encoder_activation = encoder_activation)
                                         for i in range(6)], nn.AdaptiveMaxPool1d(102))
        self.fc1 = nn.Sequential(nn.Linear(in_features = 6528, out_features = hidden_dim), make_norm(norm_type, hidden_dim), make_activation(activation),
                                 nn.Dropout(dropout))
        self.comb_block1 = nn.Sequential(nn.Linear(hidden_dim + 24 * 7, hidden_dim // 2), make_norm(norm_type, hidden_dim // 2), make_activation(activation),
                                         nn.Dropout(dropout))
        if classifier_moe:
            self.classifier_out = MoEClassifier(input_dim = hidden_dim // 2, num_classes = num_classes, num_experts = classifier_num_experts,
                                                top_k = classifier_top_k, hidden_dim = classifier_expert_hidden_dim, dropout = dropout, activation = activation,
                                                shared_expert = classifier_shared_expert, fixed_routing = classifier_fixed_routing)
        else:
            self.comb_lin2 = nn.Linear(hidden_dim // 2, num_classes)

    def instrument_lookup_table(self, instruments, num_experts):
        """Table [mode, lc, modification, trap] -> expert index (-1: no expert) over every code the embeddings take, from the instrument codes of each
        expert in expert order"""
        sizes = (self.mode_emb.num_embeddings, self.lc_emb.num_embeddings, self.modification_emb.num_embeddings, self.trap_emb.num_embeddings)
        if not instruments:
            raise ValueError('Fixed routing by instrument needs classifier_instrument_experts: the (mode, lc, modification, trap) codes of each '
                             "expert's instrument.")
        if len(instruments) != num_experts:
            raise ValueError(f'classifier_instrument_experts lists {len(instruments)} instruments for {num_experts} experts.')
        table = torch.full(sizes, -1, dtype = torch.long)
        for expert, codes in enumerate(instruments):
            codes = tuple(int(code) for code in codes)
            if len(codes) != len(sizes) or not all(0 <= code < size for code, size in zip(codes, sizes)):
                raise ValueError(f'Instrument {codes} of expert {expert} is not four (mode, lc, modification, trap) codes the embeddings take.')
            if table[codes] >= 0:
                raise ValueError(f'Instrument {codes} is listed for experts {int(table[codes])} and {expert}.')
            table[codes] = expert
        return table

    def get_classifier_aux_loss(self):
        aux_loss = self.classifier_out.get_aux_loss() if self.classifier_moe else None
        return self.fc1[0].weight.new_zeros(()) if aux_loss is None else aux_loss

    def encode(self, mz_features):
        """The spectra's features: what the convolutional trunk and fc1 make of the binned intensities and m/z remainders alone, the same for every
        composition, RT and setting a spectrum is tested with"""
        # The convolutional trunk runs per 16 spectra, as CandyCrunch_CNN's (per-spectrum results are identical)
        return self.fc1(flatten(torch.cat([self.res_block(chunk) for chunk in mz_features.split(16)]), start_dim = 1))

    def forward(self, mz_features, precursor, glycan_type, rt, mode, lc, modification, trap, rep = False, return_routing = False,
                return_aux_losses = False, return_selected_logits = True, features = None):
        # features: the spectra's features from encode, if already made (mz_features is then not read)
        # Fixed routing takes the expert from the glycan class code or the instrument's codes, before the codes become their embeddings
        if not self.classifier_fixed_routing:
            expert_indices = None
        elif self.classifier_routing_key == 'instrument':
            expert_indices = self.instrument_experts[mode.reshape(-1), lc.reshape(-1), modification.reshape(-1), trap.reshape(-1)]
        else:
            expert_indices = glycan_type
        embedded = [emb(x).squeeze(1) for emb, x in zip([self.type_emb, self.mode_emb, self.lc_emb, self.modification_emb, self.trap_emb],
                                                         [glycan_type, mode, lc, modification, trap])]
        mz = self.encode(mz_features) if features is None else features
        comb = self.comb_block1(torch.cat([mz, self.prec_block(precursor), embedded[0], self.rt_block(rt), *embedded[1:]], dim = 1))
        routing = None
        if not self.classifier_moe:
            logits = self.comb_lin2(comb)
        elif return_routing:
            logits, routing = self.classifier_out(comb, return_routing = True, return_selected_logits = return_selected_logits, expert_indices = expert_indices)
        else:
            logits = self.classifier_out(comb, expert_indices = expert_indices)
        out = [logits] + ([comb] if rep else []) + ([routing] if return_routing else []) + ([self.get_classifier_aux_loss().reshape(1)] if return_aux_losses else [])
        return out[0] if len(out) == 1 else tuple(out)