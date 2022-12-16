import numpy as np
import pandas as pd
import networkx as nx
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt
import math
import copy
import re
from collections import Counter
from itertools import product
from operator import neg
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from glycowork.motif.analysis import cohen_d
from glycowork.motif.graph import bracket_removal, min_process_glycans
from glycowork.glycan_data.loader import lib

mono_attributes = {'Gal':{'mass':{'A':{'13A':60,'24A':60,'04A':60,'35A':74,'25A':104,'02A':120,'Gal':162.0528},'X':{'Gal':162.0528}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Gal':[1,2,3,4,5,6]}},
                  'Glc':{'mass':{'A':{'13A':60,'24A':60,'04A':60,'35A':74,'25A':104,'02A':120,'Glc':162.0528},'X':{'Glc':162.0528}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Glc':[1,2,3,4,5,6]}},
                  'Man':{'mass':{'A':{'13A':60,'24A':60,'04A':60,'35A':74,'25A':104,'02A':120,'Man':162.0528},'X':{'Man':162.0528}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Man':[1,2,3,4,5,6]}},
                  'Hex':{'mass':{'A':{'13A':60,'24A':60,'04A':60,'35A':74,'25A':104,'02A':120,'Hex':162.0528},'X':{'Hex':162.0528}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Hex':[1,2,3,4,5,6]}},
                  'GalNAc':{'mass':{'A':{'04A':60,'24A':60,'35A':74,'03A':90,'25A':104,'02A':120,'GalNAc': 203.0794},'X':{'GalNAc': 203.0794}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GalNAc':[1,2,3,4,5,6]}},
                  'GlcNAc':{'mass':{'A':{'04A':60,'24A':60,'35A':74,'03A':90,'25A':104,'02A':120,'GlcNAc': 203.0794},'X':{'GlcNAc': 203.0794}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GlcNAc':[1,2,3,4,5,6]}},
                  'HexNAc':{'mass':{'A':{'04A':60,'24A':60,'35A':74,'03A':90,'25A':104,'02A':120,'HexNAc': 203.0794},'X':{'HexNAc': 203.0794}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'HexNAc':[1,2,3,4,5,6]}},
                  'Neu5Ac':{'mass':{'A':{'Neu5Ac':291.0954},'X':{'02X':70,'04X':169.8,'Neu5Ac':291.0954}},'atoms':{'02X':[1,2,3],'04X':[1,2,3,4,5],'Neu5Ac':[1,2,3,4,5,6,7,8,9]}},
                  'Neu5Gc':{'mass':{'A':{'Neu5Gc':307.0903},'X':{'02X':70,'04X':185.9,'Neu5Gc':307.0903}},'atoms':{'02X':[1,2,3],'04X':[1,2,3,4,5],'Neu5Gc':[1,2,3,4,5,6,7,8,9]}},
                  'GlcA':{'mass':{'A':{'GlcA':176.03209},'X':{'GlcA':176.03209}},'atoms':{'GlcA':[1,2,3,4,5,6]}},
                  'HexA':{'mass':{'A':{'HexA':176.03209},'X':{'HexA':176.03209}},'atoms':{'HexA':[1,2,3,4,5,6]}},
                  'Fuc':{'mass':{'A':{'Fuc':146.0579},'X':{'Fuc':146.0579}},'atoms':{'Fuc':[1,2,3,4,5,6]}},
                  'dHex':{'mass':{'A':{'dHex':146.0579},'X':{'dHex':146.0579}},'atoms':{'dHex':[1,2,3,4,5,6]}},
                  'GlcNAc6S':{'mass':{'A':{'04A':139.9568,'24A':139.9568,'35A':153.9568,'03A':169.9568,'25A':183.9568,'02A': 199.9568,'GlcNAc6S':283.0362},'X':{'GlcNAc6S':283.0362}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GlcNAc6S':[1,2,3,4,5,6]}},
                  'GlcNAcOS':{'mass':{'A':{'04A':139.9568,'24A':139.9568,'35A':153.9568,'03A':169.9568,'25A':183.9568,'02A': 199.9568,'GlcNAcOS':283.0362},'X':{'GlcNAcOS':283.0362}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GlcNAcOS':[1,2,3,4,5,6]}},
                  'GalNAc6S':{'mass':{'A':{'04A':139.9568,'24A':139.9568,'35A':153.9568,'03A':169.9568,'25A':183.9568,'02A': 199.9568,'GalNAc6S':283.0362},'X':{'GalNAc6S':283.0362}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GalNAc6S':[1,2,3,4,5,6]}},
                  'GalNAcOS':{'mass':{'A':{'04A':139.9568,'24A':139.9568,'35A':153.9568,'03A':169.9568,'25A':183.9568,'02A': 199.9568,'GalNAcOS':283.0362},'X':{'GalNAcOS':283.0362}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GalNAcOS':[1,2,3,4,5,6]}},
                  'HexNAcOS':{'mass':{'A':{'04A':139.9568,'24A':139.9568,'35A':153.9568,'03A':169.9568,'25A':183.9568,'02A': 199.9568,'HexNAcOS':283.0362},'X':{'HexNAcOS':283.0362}},'atoms':{'04A':[5,6],'24A':[3,4],'35A':[4,5,6],'03A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'HexNAcOS':[1,2,3,4,5,6]}},
                  'Gal6S':{'mass':{'A':{'13A': 60,'24A': 139.9568,'04A': 139.9568,'35A': 153.9568,'25A': 183.9568,'02A': 199.9568,'Gal6S': 242.0096},'X':{'Gal6S': 242.0096}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Gal6S':[1,2,3,4,5,6]}},
                  'GalOS':{'mass':{'A':{'13A': 60,'24A': 139.9568,'04A': 139.9568,'35A': 153.9568,'25A': 183.9568,'02A': 199.9568,'GalOS': 242.0096},'X':{'GalOS': 242.0096}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GalOS':[1,2,3,4,5,6]}},
                  'Glc6S':{'mass':{'A':{'13A': 60,'24A': 139.9568,'04A': 139.9568,'35A': 153.9568,'25A': 183.9568,'02A': 199.9568,'Glc6S': 242.0096},'X':{'Glc6S': 242.0096}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'Glc6S':[1,2,3,4,5,6]}},
                  'GlcOS':{'mass':{'A':{'13A': 60,'24A': 139.9568,'04A': 139.9568,'35A': 153.9568,'25A': 183.9568,'02A': 199.9568,'GlcOS': 242.0096},'X':{'GalOS': 242.0096}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'GlcOS':[1,2,3,4,5,6]}},
                  'HexOS':{'mass':{'A':{'13A': 60,'24A': 139.9568,'04A': 139.9568,'35A': 153.9568,'25A': 183.9568,'02A': 199.9568,'HexOS': 242.0096},'X':{'HexOS': 242.0096}},'atoms':{'13A':[2,3],'24A':[3,4],'04A':[5,6],'35A':[4,5,6],'25A':[3,4,5,6],'02A':[3,4,5,6],'HexOS':[1,2,3,4,5,6]}},
                  'Global':{'mass':{'H2O':-18,'CH2O':-30,'C2H2O':-42.04,'C2H4O2':-60.04,'C3H8O4':-108.04}}
                  }

bond_type_helper = {1:['bond','no_bond'],2:['red_bond','red_no_bond']}

cut_type_dict = {'bond':'Y','no_bond':'Z','red_bond':'C','red_no_bond':'B','13A':'13A','24A':'24A','04A':'04A','35A':'35A','03A':'03A','25A':'25A','02A':'02A','02X':'02X','04X':'04X'}

def evaluate_adjacency_monos(glycan_part, adjustment):
  """Modified version of evaluate_adjacency to check glycoletter adjacency for monosaccharide only strings\n
  | Arguments:
  | :-
  | glycan_part (string): residual part of a glycan from within glycan_to_graph
  | adjustment (int): number of characters to allow for extra length (consequence of tokenizing glycoletters)\n
  | Returns:
  | :-
  | Returns True if adjacent and False if not
  """
  #check whether glycoletters are adjacent in the main chain
  if len(glycan_part) < 1+adjustment:
    return True
  #check whether glycoletters are connected but separated by a branch delimiter
  elif glycan_part[-1] == ']':
    if len(glycan_part[:-1]) < 1+adjustment:
      return True
    else:
      return False
  return False

def glycan_to_graph_monos(glycan):
  """Modified version of glycan_to_graph taking every other node, i.e., the monosaccharides\n
  | Arguments:
  | :-
  | glycan (string): IUPAC-condensed glycan sequence\n
  | Returns:
  | :-
  | (1) a dictionary of node : monosaccharide
  | (2) an adjacency matrix of size monosaccharide X monosaccharide
  | (3) a dictionary of node : monosaccharide/linkage
  """
  bond_proc = min_process_glycans([glycan])[0]
  mono_proc = bond_proc[::2]

  all_mask_dic = {k:bond_proc[k] for k in range(len(bond_proc))}
  mono_mask_dic = {k:mono_proc[k] for k in range(len(mono_proc))}
  for k,j in mono_mask_dic.items():
    glycan = glycan.replace(j, str(k), 1)
  glycan = ''.join(re.split(r'[()]', glycan)[::2])
  adj_matrix = np.zeros((len(mono_proc), len(mono_proc)), dtype = int)

  for k in range(len(mono_mask_dic)):
    for j in range(len(mono_mask_dic)):
      if k < j:
        if k >= 100:
          adjustment = 2
        elif k >= 10:
          adjustment = 1
        else:
          adjustment = 0
        k_idx,j_idx = glycan.index(str(k)),glycan.index(str(j))
        glycan_part = glycan[k_idx+1:j_idx]

        if evaluate_adjacency_monos(glycan_part, adjustment):
          adj_matrix[k,j] = 1
          continue

        if len(bracket_removal(glycan_part)) <= 1+adjustment:
          glycan_part = bracket_removal(glycan_part)
          if evaluate_adjacency_monos(glycan_part, adjustment):
            adj_matrix[k,j] = 1
            continue

  return mono_mask_dic,adj_matrix,all_mask_dic

def create_edge_labels(gr,all_dict):
  """Helper to create a dictionary linking graph edges with bond labels\n
  | Arguments:
  | :-
  | gr (networkx_object): graph to be modified
  | all_dict (dict): dictionary mapping original node format to bonds and monos\n
  | Returns:
  | :-
  | Returns a dict mapping each gr edge to its bond label
  """
  edge_dict = {}
  for e in gr.edges:
    edge_dict[e] = {'bond_label':all_dict[(e[0]*2)+1]}

  return edge_dict

def mono_graph_to_nx(mono_graph, directed = True, libr = None):
  """Modified version of glycan_to_nxGraph, converts a mono adjacency matrix into a networkx graph, adds bonds as edge labels, and terminal,reducing end labels\n
  | Arguments:
  | :-
  | mono_graph (string): output of glycan_to_graph_monos
  | directed (bool): if True, creates a directed graph with bonds pointing from leaf to reducing end ; default:True
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used\n
  | Returns:
  | :-
  | Returns networkx graph object of a glycan made up of only monosaccharides
  """
  if libr is None:
    libr = lib
  if directed:
    template = nx.DiGraph
  else:
    template = nx.Graph
  node_dict_mono = mono_graph[0]
  adj_matrix = mono_graph[1]
  all_dict = mono_graph[2]

  if len(node_dict_mono) > 1:
    gr = nx.from_numpy_matrix(mono_graph[1],create_using=template)
    for n1, n2, d in gr.edges(data = True):
      del d['weight']
  else:
    gr = nx.Graph()
    gr.add_node(0)

  nx.set_node_attributes(gr, {k:libr.index(node_dict_mono[k]) for k in range(len(node_dict_mono))}, 'labels')
  nx.set_node_attributes(gr, {k:node_dict_mono[k] for k in range(len(node_dict_mono))}, 'string_labels')
  nx.set_node_attributes(gr, {k:'terminal' if gr.degree[k] == 1 else 'internal' for k in gr.nodes()}, 'termini')
  nx.set_node_attributes(gr, {max(gr.nodes):2},  'reducing_end')
  bond_dict = create_edge_labels(gr,all_dict)
  nx.set_edge_attributes(gr, bond_dict)

  return gr

def enumerate_subgraphs(nx_mono):
  """Returns all connected induced subgraphs of a graph\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): monosaccharide only graph\n
  | Returns:
  | :-
  | Returns a list of all networkx subgraphs
  """
  all_subgraphs = []
  for i in range(1,nx_mono.number_of_nodes()):
    k_subgraphs = enumerate_k_graphs(nx_mono,i)
    all_subgraphs.append(k_subgraphs)

  return [subg for k in all_subgraphs for subg in k]

def enumerate_k_graphs(nx_mono,k):
  """Finds all connected induced subgraphs of size k, implementation of Wernicke, S. (2005). A Faster Algorithm for Detecting Network Motifs. In: Casadio, R., Myers, G. (eds) Algorithms in Bioinformatics\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): monosaccharide only graph
  | k (int): size of subgraphs to be enumerated\n
  | Returns:
  | :-
  | Returns a list of all networkx subgraphs of size k
  """
  neighbor_dict = {v:set(nx_mono.predecessors(v))|set(nx_mono.successors(v)) for v in nx_mono.nodes}
  k_subgraphs = []
  for node in nx_mono.nodes():
    node_neighbors = {x for x in neighbor_dict[node] if x > node}
    subgraph = {node}
    extend_subgraph(subgraph,node_neighbors,node,k,k_subgraphs,neighbor_dict,nx_mono)

  return k_subgraphs

def extend_subgraph(subgraph,extension,node,k,k_subgraphs,neighbor_dict,nx_mono):
  """Main recursive feature of enumerate_k_graphs, calls itself to grow subgraph via an arbitrary path until it reaches size k\n
  | Arguments:
  | :-
  | subgraph (set): nodes making up a connected induced subgraph
  | extension (set): nodes neighbouring the subgraph with a higher node label that the start node
  | node (set): single node used to start the subgraph and take neighbours higher than
  | k (int): size of subgraph at which to stop extending and add it to the list of subgraphs
  | k_subgraphs (list): list used to accumulate all subgraphs already found of size k
  | neighbor_dict (dict): mapping of all nodes and their neighbours in the original graph
  | nx_mono (networkx_object): the original monosaccharide only graph being searched\n
  | Returns:
  | :-
  | Returns None
  """
  if len(subgraph) == k:
    graph_obj = nx_mono.subgraph(subgraph)
    k_subgraphs.append(graph_obj)
    return None
  while extension:
    w = extension.pop()
    exclusive_neighbors = get_exclusive_neighbors(w,subgraph,neighbor_dict)
    new_extension = extension|{x for x in exclusive_neighbors if x > node}
    extend_subgraph(subgraph|{w},new_extension,node,k,k_subgraphs,neighbor_dict,nx_mono)

def get_exclusive_neighbors(w,subgraph,neighbor_dict):
  """Returns the neighbors of w in the induced subgraph not neighbouring the other subgraph nodes\n
  | Arguments:
  | :-
  | w (int): node label to get the exclusive neighbours of
  | subgraph (set): nodes currently in the subgraph
  | neighbor_dict (dict): mapping of all nodes and their neighbours in the original graph\n
  | Returns:
  | :-
  | Returns a set of node labels
  """
  all_neighbors =  {x for n in subgraph for x in neighbor_dict[n]}
  w_neighbors = {x for x in neighbor_dict[w]}
  #all_neighbors = set.intersection(*[neighbor_dict[n] for n in subgraph])
  #w_neighbors = neighbor_dict[w]
  exclusive_neighbors = w_neighbors - all_neighbors

  return exclusive_neighbors

def get_broken_bonds(subg,nx_mono,nx_edge_dict):
  """Determines bonds which are floating on the subgraph nodes\n
  | Arguments:
  | :-
  | subg (networkx_object): a subgraph of nx_mono
  | nx_mono (networkx_object): the original monosaccharide only graph
  | nx_edge_dict (dict): a mapping of each edge in the original graph to its bond label\n
  | Returns:
  | :-
  | Returns a dict of bonds no longer in the subgraph and their bond label
  """
  subg_linkages = [nx_mono.out_edges(node) for node in subg.nodes()] + [nx_mono.in_edges(node) for node in subg.nodes()] #Unfortunately this is necessary due to networkx considering neighbours in directed graphs as only the successors (they even mention neighbors() and successors() are the same)
  subg_linkages = [x for v in subg_linkages for x in v]
  internal_linkages = [subg.out_edges(node) for node in subg.nodes()] + [subg.in_edges(node) for node in subg.nodes()]
  internal_linkages = [x for v in internal_linkages for x in v]
  present_breakages = [x for x in subg_linkages if x not in internal_linkages]
  present_breakages = {bond:label['bond_label'] for bond,label in nx_edge_dict.items() if bond in present_breakages}

  return present_breakages

def get_terminals(subg,present_breakages,root_node):
  """Determines all of the monosaccharides with floating bonds or those that are leaves (including the root node) in a subgraph\n
  | Arguments:
  | :-
  | subg (networkx_object): a subgraph
  | present_breakages (dict): floating bonds and their bond label
  | root_node (int): node label which is the root of the input subg\n
  | Returns:
  | :-
  | Returns a list of node labels
  """
  terminals = list({x for x in [v for w in present_breakages for v in w] if x in subg.nodes()} |
                   {node for node in subg.nodes() if subg.degree()[node] < 2 or node == root_node})
  return terminals

def calculate_mass(nx_mono):
  """Calculates the mass of a networkx glycan\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): the original monosaccharide only graph\n
  | Returns:
  | :-
  | Returns a float of the monoisotopic mass of the glycan or fragment in negative mode
  """
  comp = nx.get_node_attributes(nx_mono,'string_labels')
  mono_mods = nx.get_node_attributes(nx_mono, 'mod_labels')
  atom_dict = nx.get_node_attributes(nx_mono, 'atomic_mod_dict')
  all_atom_mods = Counter([m for d in [v.values() for v in atom_dict.values()] for m in d if m != 0])
  reducing_end = nx.get_node_attributes(nx_mono,'reducing_end')

  mass = sum([mono_attributes[v]['mass']['A'][v] for k,v in comp.items() if k not in mono_mods])-1
  mass += sum([mono_attributes[comp[k]]['mass']['A'][v] for k,v in mono_mods.items() if v in mono_attributes[comp[k]]['mass']['A']])
  mass += sum([mono_attributes[comp[k]]['mass']['X'][v] for k,v in mono_mods.items() if v not in mono_attributes[comp[k]]['mass']['A']])
  mass += -18.0105546*all_atom_mods['no_bond']
  mass += +18.0105546*all_atom_mods['red_bond']

  if reducing_end:
    if mono_mods:
      reducing_end_node = next(iter(reducing_end))
      if mono_mods[reducing_end_node] ==  comp[reducing_end_node]:
        mass += 18.0105546+2
      else:
        mass += 18.0105546+2
  return mass


def apply_global_mods(nx_mono):
  """Returns copies and the respective masses of the input graph each modified with one global modification\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): the original monosaccharide only graph\n
  | Returns:
  | :-
  | Returns two lists, one of modified graphs and one of masses
  """
  modded_subgs,modded_masses = [],[]
  for mod,mod_mass in mono_attributes['Global']['mass'].items():
    mod_label = [mod]
    mod_subg = nx_mono.copy()
    nx.set_node_attributes(mod_subg, mod_label, 'global_mod')
    total_mass = calculate_mass(nx_mono)
    mass = round(total_mass+mod_mass,6)
    modded_subgs.append(mod_subg)
    modded_masses.append(mass)
  return modded_subgs,modded_masses

def atom_mods_init(subg,present_breakages,terminals,terminal_labels):
  """Creates the initial nested dict of each terminal node with floating bonds labelled 1 and the reducing end floating bond labelled 2\n
  | Arguments:
  | :-
  | subg (networkx_object): a subgraph
  | present_breakages (dict): floating bonds and their bond label
  | terminals (list): node labels of nodes with floating bonds
  | terminal_labels (list): string labels of nodes in terminals\n
  | Returns:
  | :-
  | Returns a dict of each node label keying a dict of atoms in that node
  """
  atomic_mod_dict = {}
  for terminal,terminal_label in zip(terminals,terminal_labels):
    atomic_mod_dict[terminal] = {y:0 for y in mono_attributes[terminal_label]['atoms'][terminal_label]}

  for bond,bond_label in present_breakages.items():
    if bond[0] in subg.nodes():
      red_breakage = int(bond_label[1])
      atomic_mod_dict[bond[0]][red_breakage] = 2
    else:
      breakage = int(bond_label[-1])
      atomic_mod_dict[bond[1]][breakage] = 1
  return atomic_mod_dict

def get_mono_mods_list(root_node,subg,terminals,terminal_labels,atomic_mods,nx_edge_dict):
  """Determines all possible cross-ring modifications for each node label in terminals\n
  | Arguments:
  | :-
  | root_node (int): node label which is the root of the directed nx_mono the subgraph comes from
  | subg (networkx_object): a subgraph
  | terminals (list): node labels of nodes with floating bonds
  | terminal_labels (list): string labels of nodes in terminals
  | atomic_mods (dict): nested dict of each terminal node with floating bonds labelled at each atom
  | nx_edge_dict (dict): a mapping of each edge in the original graph to its bond label\n
  | Returns:
  | :-
  | Returns a nested list with one list of modifications per terminal node
  """
  terminal_mods = []
  for node,label in zip(terminals,terminal_labels):
    if node == root_node:
      valid_A_frags = get_valid_A_frags(subg,node,label,nx_edge_dict)
      terminal_mods.append(valid_A_frags)
    elif subg.degree()[node] > 1:
      terminal_mods.append([label])
    else:
      terminal_mods.append([x for x in mono_attributes[label]['mass']['X']])
  return terminal_mods


def get_valid_A_frags(subg,node,label,nx_edge_dict):
  """Checks which A cross-ring fragmentation is possible for the input node\n
  | Arguments:
  | :-
  | subg (networkx_object): a subgraph
  | node (int): label of node to be checked
  | label (string): string labels of nodes in terminals
  | nx_edge_dict (dict): a mapping of each edge in the original graph to its bond label\n
  | Returns:
  | :-
  | Returns a list of names of possible modifications
  """
  valid_A_mods_list = []
  A_mods_list = [x for x in mono_attributes[label]['mass']['A']]
  node_in_edges = [x for x in subg.in_edges(node)]
  for mod in A_mods_list:
    if not set([int(nx_edge_dict[bond]['bond_label'].split('-')[1][0]) for bond in node_in_edges]) <= set(mono_attributes[label]['atoms'][mod]):
      pass
    else:
      valid_A_mods_list.append(mod)
  return valid_A_mods_list

def create_dict_perms(dicty):
  """Returns all bond permutations of an atom level dict with string labels describing the floating bonds\n
  | Arguments:
  | :-
  | dicty (dict): indicates which atoms on a monosaccharide have floating bonds\n
  | Returns:
  | :-
  | Returns a list of dicts corresponsing to possible fragmentations on one node
  """
  dict_perms = []
  modded_atoms = [k for k,v in dicty.items() if v in bond_type_helper]
  for mod_prod in product(*[bond_type_helper[dicty[y]] for y in modded_atoms]):
    dict_perms.append({**dicty,**dict(zip(modded_atoms,mod_prod))})
  return dict_perms

def generate_mod_permutations(terminals,terminal_labels,mono_mods_list,atomic_mod_dict_subg):
  """Determines all possible monosaccharide modification combinations and their respective atom level representations\n
  | Arguments:
  | :-
  | terminals (list): node labels of nodes with floating bonds
  | terminal_labels (list): string labels of nodes in terminals
  | mono_mods_list (list): a nested list with one list of modifications per terminal node
  | atomic_mod_dict_subg (dict): nested dict of each terminal node with floating bonds labelled at each atom\n
  | Returns:
  | :-
  | (1) a list of all cross ring level fragmentation combinations at the node level for all of the terminal nodes
  | (2) a list of all bond fragmentation combinations at the atom dict level for all of the terminal nodes
  """
  all_terminal_perms,all_mono_mods = [],[]
  for node,label,mono_mods in zip(terminals,terminal_labels,mono_mods_list):
    possible_node_atoms = [{k:v for k,v in atomic_mod_dict_subg[node].items() if k in mono_attributes[label]['atoms'][mod]} for mod in mono_mods]
    all_atom_dict_perms,all_mono_mod_perms = [],[]
    for i,atom_dict in enumerate(possible_node_atoms):
      dict_perms = create_dict_perms(atom_dict)
      all_atom_dict_perms.extend(dict_perms)
      all_mono_mod_perms.extend(len(dict_perms)*[mono_mods[i]])
    all_terminal_perms.append(all_atom_dict_perms)
    all_mono_mods.append(all_mono_mod_perms)
  return product(*all_mono_mods),product(*all_terminal_perms)

def add_to_subgraph_fragments(subgraph_fragments,nx_mono_list,mass_list):
  """Helper to add lists of subgraphs and their respective masses to a dict\n
  | Arguments:
  | :-
  | subgraph_fragments (dict): stores lists of subgraphs indexed by their mass
  | nx_mono_list (list): list of networkx objects to be added to subgraph_fragments
  | mass_list (list): respective masses of the networkx objects to be added to subgraph_fragments\n
  | Returns:
  | :-
  | Returns an updated subgraph_fragments dict
  """
  for nx_mono,mass in zip(nx_mono_list,mass_list):
    if mass not in subgraph_fragments:
      subgraph_fragments[mass] = [nx_mono]
    else:
      subgraph_fragments[mass].append(nx_mono)

  return subgraph_fragments

def generate_atomic_frags(nx_mono, mass_mode = False, fragment_masses = None):
  """Calculates the graph and mass of all possible fragments of the input\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): the original monosaccharide only graph
  | mass_mode (bool): whether to constrain subgraph generation by observed masses
  | fragment_masses (list): all masses which are to be annotated with a fragment name\n
  | Returns:
  | :-
  | Returns a dict of lists of networkx subgraphs
  """
  # These general features are calculated here to help performance
  true_leaves = [node for node in nx_mono.nodes() if nx_mono.degree()[node] < 2]
  true_root_node = [v for v, d in nx_mono.out_degree() if d == 0][0]
  nx_edge_dict = {(node[0],node[1]):node[2] for node in nx_mono.edges(data = True)}
  node_dict = nx.get_node_attributes(nx_mono,'string_labels')
  # The first graphs to be added to the output are the global mods of the original input
  subgraph_fragments = {}
  #The subgraphs are calculated and the entire graph is also added to the list of subgraphs
  subgraphs = enumerate_subgraphs(nx_mono)
  if mass_mode:
    subgraphs = [subg for subg in subgraphs if calculate_mass(subg) > min(fragment_masses)]
  subgraphs.append(nx_mono)

  for subg in subgraphs:
    # For a subgraph we find all possible node and atom level modification
    present_breakages = get_broken_bonds(subg,nx_mono,nx_edge_dict)
    root_node = [v for v, d in subg.out_degree() if d == 0][0]
    terminals = get_terminals(subg,present_breakages,root_node)
    terminal_labels = [node_dict[x] for x in terminals]

    atomic_mod_dict_subg = atom_mods_init(subg,present_breakages,terminals,terminal_labels)
    mono_mods_list = get_mono_mods_list(root_node,subg,terminals,terminal_labels,atomic_mod_dict_subg,nx_edge_dict)
    mono_mod_perms,atom_dict_perms = generate_mod_permutations(terminals,terminal_labels,mono_mods_list,atomic_mod_dict_subg)

    for mono_mods,atom_dicts in zip(mono_mod_perms,atom_dict_perms):
      # For every modification combination we copy and label the subgraph as such, before calculating its mass and adding it to the output
      mod_subg = subg.copy() #Consider subgraph instead
      mono_mods_dict = dict(zip(terminals,mono_mods))
      nx.set_node_attributes(mod_subg, mono_mods_dict, 'mod_labels')
      atoms_mods_dict = dict(zip(terminals,atom_dicts))
      nx.set_node_attributes(mod_subg, atoms_mods_dict, 'atomic_mod_dict')
      mod_subg_mass = calculate_mass(mod_subg)
      mod_subg_mass = round(mod_subg_mass,6)
      # All global modifications are also added to each combination of modifications below
      # if [x for x in set(mono_mods_dict.values()) ^ set(nx.get_node_attributes(mod_subg,'string_labels').values()) if x[-1] == 'A']: # Comment in if criteria are desired to add global mods to subgraphs (this line adds them only if there is an A cross ring)
      modded_graphs,modded_masses = apply_global_mods(mod_subg)
      subgraph_fragments = add_to_subgraph_fragments(subgraph_fragments,modded_graphs,modded_masses)

      subgraph_fragments = add_to_subgraph_fragments(subgraph_fragments,[mod_subg],[mod_subg_mass])

  return subgraph_fragments

def subgraphs_to_domon_costello(nx_mono,subgs, max_frags = 3):
  """Converts the subgraphs of a given graph object into their canonical Domon & Costello fragment names\n
  | Arguments:
  | :-
  | nx_mono (networkx_object): the original monosaccharide only graph
  | subg (list): a list of modified networkx subgraphs
  | max_frags (int): maximum number of allowed concurrent fragmentations per mass; default:3\n
  | Returns:
  | :-
  | Returns a nested list with one list of fragment labels for each subgraph
  """
  ranks = ['Alpha','Beta','Gamma','Delta','Epsilon','Zeta','Eta'] #Possible ranks (more can be added depending on branching)
  ion_names = []
  leaf_chains = {}
  chain_rank = []

  node_dict = nx.get_node_attributes(nx_mono,'string_labels')
  og_root = [v for v, d in nx_mono.out_degree() if d == 0][0]
  og_leaves = set(v for v, d in nx_mono.in_degree() if d == 0)

  leaf_chains = {}
  for og_leaf in og_leaves:
    leaf_chains[og_leaf] = nx.shortest_path(nx_mono,source = og_leaf) # Calculates all of the paths in the original glycan from each of the original leaf nodes
  main_chains = sorted([branch_path[og_root] for branch_path in leaf_chains.values()],key=lambda x:sum([mono_attributes[node_dict[n]]['mass']['A'][node_dict[n]] for n in x]),reverse=True) #Sorts the main paths by mass as described in the nomenclature
  chain_rank = list(zip(ranks,main_chains))

  for subg in subgs:
    cuts = []
    cut_labels = []
    # Adds any global mods to the fragment labels
    global_mods = nx.get_node_attributes(subg, 'global_mod')
    if global_mods:
      global_mods = list(global_mods.values())
      cut_labels.append('_'.join(('M',global_mods[0][0])))

    root_node = [v for v, d in subg.out_degree() if d == 0][0]
    mono_mod_dict = nx.get_node_attributes(subg,'mod_labels')
    atomic_mod_dict = nx.get_node_attributes(subg,'atomic_mod_dict')
    # Adds different atoms to use for calculation depending on mod type
    for node,atom_mods in atomic_mod_dict.items():
      for atom,atom_mod in atom_mods.items():
        if atom_mod in ['bond','no_bond']:
          cut_node = [bonding_node for bonding_node,bonded_node,atts in nx_mono.edges(data=True) if bonded_node==node if atts['bond_label'][-1] == str(atom)] #Finds the node which was cleaved to produce the floating bond
          if cut_node:
            cuts.append((atom_mod,cut_node[0],node))
        if atom_mod in ['red_bond','red_no_bond']:
          cuts.append((atom_mod,node))
    cross_rings = [(k,v) for k,v in mono_mod_dict.items() if (k,v) not in node_dict.items()]
    for node,mod in cross_rings:
        cuts.append((mod,node))
    #Assigns the rank of the chain on which the breakage occured and the type and number
    for cut in cuts:
      cut_type = cut_type_dict[cut[0]]
      if cut_type[-1] in ['B','C','A']:
        cut_rank,cut_chain = [(rank,chain) for rank,chain in chain_rank if cut[1] in chain][0]
        cut_number = cut_chain.index(cut[1]) + 1
      elif cut_type[-1] in ['Y','Z']:
        cut_rank,cut_chain = [(rank,chain) for rank,chain in chain_rank if cut[1] in chain][0]
        cut_number = cut_chain[::-1].index(cut[2])+1
      elif cut_type[-1] in ['X']:
        cut_rank,cut_chain = [(rank,chain) for rank,chain in chain_rank if cut[1] in chain][0]
        cut_number = cut_chain[::-1].index(cut[1])+1
      cut_labels.append('_'.join((cut_type,str(cut_number),cut_rank)))

    if len(cut_labels) <= max_frags:
      ion_names.append((cut_labels))

  return sorted(ion_names, key = len)[:5]

def get_most_likely_fragments(out_in):
  """uses Occam's razor to determine most likely fragment at a peak\n
  | Arguments:
  | :-
  | out_in (list): list of tuples of the form m/z : fragment names, generated within CandyCrumbs\n
  | Returns:
  | :-
  | Returns format similar to out_in but condensed
  """
  out = copy.deepcopy(out_in)
  out_list = []
  single_list = []
  for t in out[::-1]:
    out_len = len(out_list)
    if len(t) > 1 and isinstance(t[1], list) and len(t[1][:1]) == 1:
      out_list.append((t[0], t[1][0]))
      single_list.append(t[1][0][0])
    elif len(t) > 1 and isinstance(t[1], list) and len(t[1][:1]) == 2:
      for tt in t[1]:
        if len(tt) == 2 and any([k in tt for k in single_list]):
          out_list.append((t[0], tt))
          break
      if len(out_list) == out_len:
        out_list.append(t)
    else:
      out_list.append(t)
  return out_list[::-1]

def CandyCrumbs(glycan_string,fragment_masses,threshold, libr = None,
                max_frags = 3, simplify = False):
  """Basic wrapper for the annotation of observed masses with correct nomenclature given a glycan\n
  | Arguments:
  | :-
  | glycan_string (string): glycan in IUPAC-condensed format
  | fragment_masses (list): all masses which are to be annotated with a fragment name
  | threshold (float): the tolerated mass difference around each observed mass at which to include fragments
  | libr (list): library of monosaccharides; if you have one use it, otherwise a comprehensive lib will be used
  | max_frags (int): maximum number of allowed concurrent fragmentations per mass; default:3
  | simplify (bool): whether to try condensing fragment options to the most likely option; default:False\n
  | Returns:
  | :-
  | Returns a list of tuples containing the observed mass and all of the possible fragment names within the threshold
  """
  if libr is None:
    libr = lib
  hit_list = []
  fragment_masses = sorted(fragment_masses)
  mono_graph = glycan_to_graph_monos(glycan_string)
  nx_mono = mono_graph_to_nx(mono_graph, directed = True, libr = libr)
  subg_frags = generate_atomic_frags(nx_mono, mass_mode = True, fragment_masses = fragment_masses)
  for mass in fragment_masses:
    hits = [k for k in subg_frags if abs(mass-k) < threshold]
    if hits:
      graphs = [g for m in hits for g in subg_frags[m]]
      ion_names = subgraphs_to_domon_costello(nx_mono,graphs, max_frags = max_frags)
      hit_list.append((mass,ion_names))
    else:
      hit_list.append((mass,None))
  if simplify:
    hit_list = get_most_likely_fragments(hit_list)

  return hit_list

def get_unique_subgraphs(nx_mono1,nx_mono2):
  """Gets the subgraphs unique to each of two input graphs\n
  | Arguments:
  | :-
  | nx_mono1 (networkx object): a monosaccharide only graph
  | nx_mono2 (networkx object): a different monosaccharide only graph\n
  | Returns:
  | :-
  | Returns two lists of networkx subgraphs of the inputs
  """
  nm = iso.categorical_node_match("string_labels",None) #This is the criterion used to match nodes (it can also be something more general i.e Hex, HexNAc etc)
  all_unique_graphs1 = []
  all_unique_graphs2 = []
  # Only compare subgraphs of the same size
  for i in range(1,min(len(nx_mono1.nodes()),len(nx_mono2.nodes()))):
    graphs1 = set()
    graphs2 = set()
    first_graphs = enumerate_k_graphs(nx_mono1,i)
    second_graphs = enumerate_k_graphs(nx_mono2,i)
    for subg_1 in first_graphs:
      undir_subg_1 = subg_1.to_undirected()
      for subg_2 in second_graphs:
        undir_subg_2 = subg_2.to_undirected()
        if nx.is_isomorphic(undir_subg_1,undir_subg_2,node_match=nm):
          graphs1.add(first_graphs.index(subg_1))
          graphs2.add(second_graphs.index(subg_2))
        else:
          pass
    #Take only the subgraphs from each graph which are not isomorphic
    kunique_graphs1 = [first_graphs[x] for x in range(len(first_graphs)) if x not in graphs1]
    all_unique_graphs1.extend(kunique_graphs1)
    kunique_graphs2 = [second_graphs[x] for x in range(len(second_graphs)) if x not in graphs2]
    all_unique_graphs2.extend(kunique_graphs2)

  return all_unique_graphs1,all_unique_graphs2

def get_plots(df_sub, glycan_list, num_bins_plot, mz_range):
  """averages and plots spectra for two glycans
  | Arguments:
  | :-
  | df_sub (dataframe): dataframe containing spectra of the two glycans and prediction confidences
  | glycan_list (list): list of two glycans in IUPAC-condensed nomenclature
  | num_bins_plot (int): number of bins to use for plotting the averaged spectra
  | mz_range (list): m/z values demarking bin edges across the whole m/z range\n
  | Returns:
  | :-
  | Returns
  """
  out_dic = {}
  for g in glycan_list:
      out_dic[g] = [sum(col)/len(col) for col in zip(*df_sub[df_sub.Prediction == g].binned_intensities.values.tolist())]
      if len(glycan_list) == 2 and glycan_list.index(g) == 1:
        plt.plot(mz_range, list(map(neg, out_dic[g]))[:num_bins_plot])
      else:
        plt.plot(mz_range, out_dic[g][:num_bins_plot])
  plt.xlabel("m/z")
  plt.ylabel("Relative intensity")
  plt.legend(glycan_list)

def get_averaged_spectra(df, glycan_list, max_mz = 3000, min_mz = 39.714, bin_num = 2048,
                         num_bins_plot = 500, conf_analysis = False):
  """averages spectra for two glycans and plots the averaged spectra in comparison mode\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing predictions and prediction confidences for every spectrum
  | glycan_list (list): list of two glycans in IUPAC-condensed nomenclature
  | max_mz (float): maximum m/z value considered for model training; default:3000, do not change
  | min_mz (float): minimum m/z value considered for model training; default:39.714, do not change
  | bin_num (int): number of bins to bin m/z range, used for model training; default:2048, change if you binned differently
  | num_bins_plot (int): number of bins to use for plotting the averaged spectra; default:500
  | conf_analysis (bool): whether to plot the spectra comparisons separately for different levels of spectrum quality; default:False\n
  | Returns:
  | :-
  | Returns comparison plots for the averaged spectra and a dictionary of form glycan : averaged intensities
  """
  df_sub = df[df.Prediction.isin(glycan_list)].reset_index(drop = True)
  mz_range= [min_mz+((max_mz-min_mz)/bin_num)*k for k in range(num_bins_plot)]
  out_dic = {}
  if conf_analysis:
    conf_brackets = [(0.9,1.0), (0.6,0.9), (0.3,0.6), (0,0.3)]
    for bracket in conf_brackets:
      plt.clf()
      get_plots(df_sub[df_sub.Confidence.between(bracket[0], bracket[1])], glycan_list, num_bins_plot, mz_range)
      plt.title("Confidence range: "+ str(bracket))
      plt.show()
  else:
    get_plots(df_sub, glycan_list, num_bins_plot, mz_range)
  return out_dic

def run_controls(df_a, df_b):
  """checks for systematic differences in tested glycans (length & branching)\n
  | Arguments:
  | :-
  | df_a (dataframe): dataframe containing spectra with predictions and prediction confidences
  | df_b (dataframe): dataframe containing spectra with predictions and prediction confidences\n
  | Returns:
  | :-
  | Returns printed p-values of comparing systematic differences in tested glycans (length & branching)
  """
  len_comp = ttest_ind([len(k) for k in df_a.Prediction.values.tolist()],
                       [len(k) for k in df_b.Prediction.values.tolist()], equal_var = False)
  print("p-value (Welch's t-test) for differences in glycan length: " + str(len_comp))
  branch_comp = ttest_ind([k.count('[') for k in df_a.Prediction.values.tolist()],
                       [k.count('[') for k in df_b.Prediction.values.tolist()], equal_var = False)
  print("p-value (Welch's t-test) for differences in glycan branching: " + str(branch_comp))

def get_sig_bins(df, glycan_list, conf_range = None, mz_cap = 3000, max_mz = 3000, min_mz = 39.714, bin_num = 2048,
                 motif = None, motif2 = None, controls = False):
  """searching diagnostic ions or ion ratios between two glycans in MS/MS spectra\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing spectra, predictions, and prediction confidences
  | glycan_list (list): list of two glycans in IUPAC-condensed nomenclature
  | conf_range (list): list of two confidence values that denote the confidence bracket used for spectra extraction, default uses all; default:None
  | mz_cap (float): maximum m/z value considered for analysis; default:3000
  | max_mz (float): maximum m/z value considered for model training; default:3000, do not change
  | min_mz (float): minimum m/z value considered for model training; default:39.714, do not change
  | bin_num (int): number of bins to bin m/z range, used for model training; default:2048, change if you binned differently
  | motif (string): if a glycan motif is specified in IUPAC-condensed, all glycans with and without will be compared; default:None
  | motif2 (string): if this and motif is specified, spectra of those motifs will be compared (with no spectra of molecules containing both motifs); default:None
  | controls (bool): whether to check for systematic differences in tested glycans (length & branching)\n
  | Returns:
  | :-
  | Returns a list of tuples of the form (peak m/z, corrected p-value, effect size via Cohen's d)
  """
  max_bin = round((mz_cap-min_mz) / ((max_mz-min_mz)/bin_num))
  if motif is None:
    df_a = df[df.Prediction == glycan_list[0]].reset_index(drop = True)
    df_b = df[df.Prediction == glycan_list[1]].reset_index(drop = True)
  else:
    df_a = df[df.Prediction.str.contains(motif, regex = False)].reset_index(drop = True)
    if motif2 is None:
      df_b = df[~df.Prediction.str.contains(motif, regex = False)].reset_index(drop = True)
    else:
      df_a = df_a[~df_a.Prediction.str.contains(motif2, regex = False)].reset_index(drop = True)
      df_b = df[df.Prediction.str.contains(motif2, regex = False)].reset_index(drop = True)
      df_b = df_b[~df_b.Prediction.str.contains(motif, regex = False)].reset_index(drop = True)
  if conf_range is not None:
    df_a = df_a[df_a.Confidence.between(conf_range[0], conf_range[1])]
    df_b = df_b[df_b.Confidence.between(conf_range[0], conf_range[1])]
  if controls:
    print("Number of spectra in df_a: " + str(len(df_a)))
    print("Number of spectra in df_b: " + str(len(df_b)))
    run_controls(df_a, df_b)
  df_r = pd.concat([df_a, df_b], axis = 0)
  remainder = [np.median(col) for col in zip(*df_r.mz_remainder.values.tolist())]
  df_a = np.array(df_a.binned_intensities.values.tolist())
  df_b = np.array(df_b.binned_intensities.values.tolist())
  pvals = [ttest_ind(df_a[:,k], df_b[:,k], equal_var = False)[1] for k in range(max_bin)]
  pvals = multipletests(pvals)[1]
  cohensd = [cohen_d(df_a[:,k], df_b[:,k]) for k in range(max_bin)]
  sig_bins = [k for k in range(max_bin) if pvals[k] < 0.05]
  sig_bins= [(min_mz+((max_mz-min_mz)/bin_num)*k + remainder[k], pvals[k], cohensd[k]) for k in sig_bins]
  return sorted(sig_bins, key = lambda x: (x[1],1/abs(x[2])))

def follow_sigs(df, glycan_list, mz_cap = 3000, max_mz = 3000, min_mz = 39.714, bin_num = 2048,
                 motif = None, motif2 = None, thresh = 0.5, conf_range = [(0.9, 1.0), (0.8, 0.9),(0.6, 0.8),(0.4, 0.6),
               (0.2, 0.4), (0, 0.2)]):
  """following diagnostic ions or ion ratios between two glycans across MS/MS spectra of different quality\n
  | Arguments:
  | :-
  | df (dataframe): dataframe containing binned spectra, predictions, and prediction confidences
  | glycan_list (list): list of two glycans in IUPAC-condensed nomenclature
  | conf_range (list): list of two confidence values that denote the confidence bracket used for spectra extraction, default uses all; default:None
  | mz_cap (float): maximum m/z value considered for analysis; default:3000
  | max_mz (float): maximum m/z value considered for model training; default:3000, do not change
  | min_mz (float): minimum m/z value considered for model training; default:39.714, do not change
  | bin_num (int): number of bins to bin the m/z range, used for model training; default:2048, change if you binned differently
  | motif (string): if a glycan motif is specified in IUPAC-condensed, all glycans with and without will be compared; default:None
  | motif2 (string): if this and motif is specified, spectra of those motifs will be compared (with no spectra of molecules containing both motifs); default:None
  | thresh (float): effect size threshold to exclude fragments with max value below thresh; default:0.5
  | conf_range(list): list of tuples with prediction confidence boundaries to bin the data according to prediction confidences\n
  | Returns:
  | :-
  | Returns a plot of fragment effect size across prediction confidences and a dictionary of form peak : list of effect sizes across prediction confidences
  """
  max_bin = round((mz_cap-min_mz) / ((max_mz-min_mz)/bin_num))
  df_a = df[df.Prediction == glycan_list[0]].reset_index(drop = True)
  df_b = df[df.Prediction == glycan_list[1]].reset_index(drop = True)
  df_r = pd.concat([df_a, df_b], axis = 0)
  remainder = [np.median(col) for col in zip(*df_r.mz_remainder.values.tolist())]
  bins = {min_mz+((max_mz-min_mz)/bin_num)*k + remainder[k]:[] for k in range(max_bin)}
  for conf in conf_range:
    df_a2 = df_a[df_a.Confidence.between(conf[0], conf[1])]
    df_b2 = df_b[df_b.Confidence.between(conf[0], conf[1])]
    df_a2 = np.array(df_a2.binned_intensities.values.tolist())
    df_b2 = np.array(df_b2.binned_intensities.values.tolist())
    cohensd = [cohen_d(df_a2[:,k], df_b2[:,k]) for k in range(max_bin)]
    pvals = [ttest_ind(df_a2[:,k], df_b2[:,k], equal_var = False)[1] for k in range(max_bin)]
    pvals = multipletests(pvals)[1]
    for c in range(len(cohensd)):
      if pvals[c] < 0.05:
        bins[list(bins.keys())[c]] += [cohensd[c]]
      else:
        bins[list(bins.keys())[c]] += [0]
  bins = {k:v for k,v in bins.items() if max([abs(v2) for v2 in v]) >= thresh and not any([math.isinf(v2) for v2 in v])}
  conf_idx = [c[1] for c in conf_range]
  for key, data_list in bins.items():
    plt.plot(conf_idx, data_list, label = key)
  return bins
