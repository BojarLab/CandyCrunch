import copy
import math
from collections import Counter
import re
import warnings
from itertools import combinations_with_replacement, product
from operator import neg
import bisect
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from glycowork.motif.processing import canonicalize_composition, is_composition, rescue_glycans, get_class
from glycowork.motif.tokenization import map_to_basic, HYDROGEN_MASS, PROTON_MASS, calculate_adduct_mass, \
    composition_to_mass, glycan_to_composition, get_core, get_modification, compositions_to_structures
from glycowork.motif.graph import glycan_to_nxGraph, get_possible_topologies
from glycowork.glycan_data.stats import cohen_d, correct_multiple_testing
from scipy.stats import ttest_ind

mono_attributes = {
    'Hex': {'mass': {'03X': 72.0211, '02X': 42.0106, '15X': 27.9949, '13A': 60.0211, '24A': 60.0211, '15A': 134.057859,
                     '04A': 60.0211, '35A': 74.0368, '25A': 104.0473, '02A': 120.0423, '03A': 90.0317, '24X': 102.0317,
                     '04X': 102.0317, '35X': 88.016, 'Hex': 162.0528},
            'atoms': {'03X': [1, 2, 3], '02X': [1, 2], '15X': [1], '13A': [2, 3], '24A': [3, 4], '15A': [2, 3, 4, 5, 6],
                      '04A': [5, 6], '35A': [4, 5, 6], '25A': [3, 4, 5, 6], '02A': [3, 4, 5, 6], '03A': [4, 5, 6],
                      '24X': [1, 2, 5, 6], '04X': [1, 2, 3, 4],
                      '35X': [1, 2, 3], 'Hex': [1, 2, 3, 4, 5, 6]}},
    'HexNAc': {'mass': {'04A': 60.0211, '24A': 60.0211, '35A': 74.0368, '03A': 90.0317, '25A': 104.0473, '25X': 99.0321,
                        '02A': 120.0423, '24X': 143.0583, '14A': 131.058244, '15A': 175.084458, '15X': 27.994941,
                        '04X': 143.0583, '35X': 129.0426, '02X': 83.037114,
                        '14X': 72.021156, '13X': 102.031741, '13A': 101.047659, 'HexNAc': 203.0794},
               'atoms': {'04A': [5, 6], '24A': [3, 4], '35A': [4, 5, 6], '03A': [4, 5, 6], '25A': [3, 4, 5, 6],
                         '25X': [1, 2], '02X': [1, 2],
                         '02A': [3, 4, 5, 6], '24X': [1, 2, 5, 6], '14A': [2, 3, 4], '15A': [2, 3, 4, 5, 6], '15X': [1],
                         '04X': [1, 2, 3, 4], '35X': [1, 2, 3],
                         '14X': [1, 5, 6], '13X': [1, 4, 5, 6], '13A': [2, 3], 'HexNAc': [1, 2, 3, 4, 5, 6]}},
    'Neu5Ac': {'mass': {'02X': 70.0055, '04X': 170.0453, '24X': 191.0556, '02A': 221.0899,
                        '04A': 121.0501, '24A': 100.0398, 'Neu5Ac': 291.0954},
               'atoms': {'02X': [1, 2, 3], '04X': [1, 2, 3, 4, 5], '24X': [1, 2, 3, 6, 7, 8, 9],
                         '02A': [4, 5, 6, 7, 8, 9],
                         '04A': [6, 7, 8, 9], '24A': [4, 5], 'Neu5Ac': [1, 2, 3, 4, 5, 6, 7, 8, 9]}},
    'Neu5Gc': {'mass': {'02X': 70.0055, '04X': 186.0402, '24X': 191.0556, '02A': 237.0848,
                        '04A': 121.0501, '24A': 116.0347, 'Neu5Gc': 307.0903},
               'atoms': {'02X': [1, 2, 3], '04X': [1, 2, 3, 4, 5], '24X': [1, 2, 3, 6, 7, 8, 9],
                         '02A': [4, 5, 6, 7, 8, 9],
                         '04A': [6, 7, 8, 9], '24A': [4, 5], 'Neu5Gc': [1, 2, 3, 4, 5, 6, 7, 8, 9]}},
    'Kdn': {'mass': {'02X': 70.0055, '04X': 129.0188, '24X': 191.0556, '02A': 180.0634,
                     '04A': 121.0501, '24A': 59.0133, 'Kdn': 250.0689},
            'atoms': {'02X': [1, 2, 3], '04X': [1, 2, 3, 4, 5], '24X': [1, 2, 3, 6, 7, 8, 9], '02A': [4, 5, 6, 7, 8, 9],
                      '04A': [6, 7, 8, 9], '24A': [4, 5], 'Kdn': [1, 2, 3, 4, 5, 6, 7, 8, 9]}},
    'HexA': {'mass': {'02X': 42.0106, '02A': 134.02159, '24X': 116.01099, '24A': 60.0211, 'HexA': 176.03209},
             'atoms': {'02X': [1, 2], '02A': [3, 4, 5, 6], '24X': [1, 2, 5, 6], '24A': [3, 4],
                       'HexA': [1, 2, 3, 4, 5, 6]}},
    'dHex': {'mass': {'02X': 42.0106, '02A': 104.0474, '25X': 58.0055, '25A': 88.0524, 'dHex': 146.0579},
             'atoms': {'02X': [1, 2], '02A': [3, 4, 5, 6], '25X': [1, 2], '25A': [3, 4, 5, 6],
                       'dHex': [1, 2, 3, 4, 5, 6]}},
    'Pen': {'mass': {'01A': 120.0423, '02A': 90.0317, '03A': 60.0211, '15X': 27.994941, '15A': 104.047359,
                     '12X': 102.0317, '03X': 72.0211, '02X': 42.0106, 'Pen': 132.0423},
            'atoms': {'01A': [2, 3, 4, 5], '02A': [3, 4, 5], '03A': [4, 5], '15X': [1], '15A': [2, 3, 4, 5],
                      '12X': [1, 3, 4, 5], '03X': [1, 2, 3], '02X': [1, 2], 'Pen': [1, 2, 3, 4, 5]}},
    'Global': {'mass': {'H2O': -18.0105546, 'NH3': -17.026549, 'CH2O': -30.0106, 'C2H2O': -42.0106, 'CO2': -43.9898,
                        'SO4': -79.9568, 'PO4': -79.9663, 'C3H8O4': -108.0423, '+Acetonitrile': +41.0265, 'C2H4O2': -60.0211,
                        '+Acetate': 59.013851,
                        '+Na': +22.989218, '+K': 38.963158}}
}
WATER_MASS = 18.0105546
CH2_MASS = calculate_adduct_mass('CH2')
bond_type_helper = {1: ['bond', 'no_bond'], 2: ['red_bond', 'red_no_bond'], 3: ['peptide_a', 'peptide_b', 'peptide_c'],
                    4: ['peptide_y', 'peptide_z', 'peptide_w']}
# Neutral radical lost from a z. ion by Cbeta-Cgamma homolysis, giving the w ion; residues without a
# gamma atom (G, A, P) cannot form w ions and are absent by design
W_SIDE_CHAIN_LOSSES = {'C': 32.979896, 'D': 44.997655, 'E': 59.013305, 'F': 77.039125, 'H': 67.029623,
                       'I': 29.039125, 'K': 58.065674, 'L': 43.054775, 'M': 61.011196, 'N': 44.013639,
                       'Q': 58.029289, 'R': 86.071822, 'S': 17.002740, 'T': 15.023475, 'V': 15.023475,
                       'W': 116.049024, 'Y': 93.034040}
W_SIDE_CHAIN_LOSSES['c'] = 90.001360  # Carbamidomethyl Cys, cleaved at Cbeta-Sgamma
W_SIDE_CHAIN_LOSSES['m'] = W_SIDE_CHAIN_LOSSES['M'] + 15.994915  # Met sulfoxide, the lost radical keeps the oxygen
W_SIDE_CHAIN_LOSSES['j'] = W_SIDE_CHAIN_LOSSES['k'] = W_SIDE_CHAIN_LOSSES['K']
cut_type_dict = {'bond': 'Y', 'no_bond': 'Z', 'red_bond': 'C', 'red_no_bond': 'B',
                 '13A': '13A', '14A': '14A', '15A': '15A', '24A': '24A', '04A': '04A', '35A': '35A', '03A': '03A',
                 '25A': '25A', '02A': '02A',
                 '02X': '02X', '03X': '03X', '04X': '04X', '12X': '12X', '13X': '13X', '14X': '14X', '15X': '15X',
                 '24X': '24X', '35X': '35X'}
A_cross_rings = {c for c in cut_type_dict if c[-1] == 'A'}
X_cross_rings = {c for c in cut_type_dict if c[-1] == 'X'}
ranks = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu']
AA_masses = {'A': 71.0371, 'R': 156.1011, 'N': 114.0429, 'D': 115.0269,
             'C': 103.0091, 'E': 129.0425, 'Q': 128.0585, 'G': 57.0214, 'H': 137.0589,
             'I': 113.0840, 'L': 113.0840, 'K': 128.0949, 'k': 357.25783, 'M': 131.0404, 'F': 147.0684,
             'P': 97.0527, 'S': 87.0320, 'T': 101.0476, 'W': 186.0793, 'Y': 163.0633, 'V': 99.0684}
AA_masses['c'] = 103.0091 + 57.02146  # Carbamidomethyl Cys
AA_masses['j'] = 128.0949 + 42.02180  # Guanidinyl Lys
AA_masses['m'] = 131.0404 + 15.99491  # Met sulfoxide, the variable modification of nearly every glycoproteomics search
MODIFICATION_TOKENS = {
    'Carbamidomethyl': {'C': 'c'},
    'Guanidinyl': {'K': 'j'},
    'Oxidation': {'M': 'm'},
}
# Relative abundance of the backbone ion types within a fragmentation method; the method already gates
# which types can occur, so these only rank the allowed types against each other
PEPTIDE_ION_PRIORS = {'y': 1.0, 'b': 0.9, 'c': 0.9, 'z': 0.9, 'a': 0.3, 'w': 0.3, 'x': 0.1}
N_TERM_IONS = {'a', 'b', 'c'}
C_TERM_IONS = {'w', 'x', 'y', 'z'}
PEPTIDE_ION_TYPES = {'CID': {'b', 'y'}, 'HCD': {'a', 'b', 'y'}, 'ETD': {'c', 'z'}, 'ECD': {'c', 'z'},
                     'EThcD': {'b', 'c', 'w', 'y', 'z'}, 'ETciD': {'b', 'c', 'w', 'y', 'z'},
                     None: N_TERM_IONS | C_TERM_IONS}
tester_ma_addition = {k: {'mass': {k: v}, 'atoms': {k: [1, 2, 3, 4, 5, 6]}} for k, v in AA_masses.items()}
mono_attributes = mono_attributes | tester_ma_addition
bond_masses = {'red_bond': WATER_MASS, 'no_bond': -WATER_MASS, 'peptide_b': -WATER_MASS,
               'peptide_c': -(WATER_MASS - 17.026549),
               'peptide_z': -(17.026549 - 1.007825),
               'peptide_w': -(17.026549 - 1.007825),  # plus a residue-specific side-chain loss
               'peptide_a': -(WATER_MASS + 27.994915)}  # z here is the EThcD radical z. convention
# Atom positions whose -OH (or amide N-H, -NH2, -COOH) groups get derivatized, per base monosaccharide; C1 (C2 of the
# nonulosonic acids) is excluded as the glycosidic bond takes it in the residue mass convention. A position carrying two
# groups is listed twice (the free amine of HexN, the N-glycolyl of Neu5Gc). Permethylation also methylates amide N-H and
# carboxylic acids, peracetylation does neither
derivatization_sites = {
    'permethylated': {'Hex': (2, 3, 4, 6), 'HexNAc': (2, 3, 4, 6), 'HexN': (2, 2, 3, 4, 6), 'dHex': (2, 3, 4),
                      'Pen': (2, 3, 4), 'HexA': (2, 3, 4, 6), 'Neu5Ac': (1, 4, 5, 7, 8, 9),
                      'Neu5Gc': (1, 4, 5, 5, 7, 8, 9), 'Kdn': (1, 4, 5, 7, 8, 9)},
    'peracetylated': {'Hex': (2, 3, 4, 6), 'HexNAc': (3, 4, 6), 'HexN': (2, 3, 4, 6), 'dHex': (2, 3, 4),
                      'Pen': (2, 3, 4), 'HexA': (2, 3, 4), 'Neu5Ac': (4, 7, 8, 9), 'Neu5Gc': (4, 5, 7, 8, 9),
                      'Kdn': (4, 5, 7, 8, 9)}}
DERIVATIZATION_MASSES = {'permethylated': CH2_MASS, 'peracetylated': calculate_adduct_mass('C2H2O')}
# Substituents of modified monosaccharides (GlcNAc6S, Neu5Ac9Ac, ...) and of compositions, with how each changes the
# number of derivatized groups as glycowork has it (an O-sulfate takes a hydroxyl out of permethylation: -1, ...)
SUBSTITUENTS = {s: {'mass': composition_to_mass({s: 1}) - composition_to_mass({})} | {
    prep: round((composition_to_mass({s: 1}, sample_prep = prep) - composition_to_mass({}, sample_prep = prep) -
                 composition_to_mass({s: 1}) + composition_to_mass({})) / deriv_mass)
    for prep, deriv_mass in DERIVATIZATION_MASSES.items()} for s in ('S', 'P', 'Me', 'Ac', 'PCho', 'PEtN', '-H2O')}
# HexN is HexNAc without the N-acetyl on C2
mono_attributes['HexN'] = {
    'atoms': {'HexN' if f == 'HexNAc' else f: atoms for f, atoms in mono_attributes['HexNAc']['atoms'].items()},
    'mass': {
        'HexN' if f == 'HexNAc' else f: m - SUBSTITUENTS['Ac']['mass'] * (2 in mono_attributes['HexNAc']['atoms'][f])
        for f, m in mono_attributes['HexNAc']['mass'].items()}}
# how many derivatized groups each fragment of a base monosaccharide keeps
for mono in derivatization_sites['permethylated']:
    mono_attributes[mono] |= {prep: {frag: sum(s in atoms for s in sites[mono]) for frag, atoms in
                                     mono_attributes[mono]['atoms'].items()} for prep, sites in
                              derivatization_sites.items()}
# to be updated with a more empirical estimation once we have clear-cut annotation data
fragmentation_priors = {
    'cleavage_type': {
        -1: {  # negative ion mode
            'Y': 1.0, 'Z': 0.7, 'C': 0.4, 'B': 0.3,
            '02A': 0.6, '24A': 0.5, '03A': 0.45, '04A': 0.4,
            '35A': 0.35, '25A': 0.35, '13A': 0.3, '14A': 0.3, '15A': 0.25,
            '02X': 0.3, '04X': 0.25, '24X': 0.25, '03X': 0.2,
            '35X': 0.2, '25X': 0.2, '13X': 0.15, '14X': 0.15, '15X': 0.15, '12X': 0.15,
        },
        1: {  # positive ion mode
            'B': 1.0, 'Y': 0.9, 'C': 0.5, 'Z': 0.3,
            '02A': 0.5, '24A': 0.45, '04A': 0.4, '03A': 0.35,
            '35A': 0.3, '25A': 0.3, '13A': 0.25, '14A': 0.25, '15A': 0.2,
            '02X': 0.25, '04X': 0.2, '24X': 0.2, '03X': 0.15,
            '35X': 0.15, '25X': 0.15, '13X': 0.1, '14X': 0.1, '15X': 0.1, '12X': 0.1,
        },
    },
    'global_mod': {
        'H2O': 0.9, 'NH3': 0.7, 'CH2O': 0.5, 'C2H2O': 0.4, 'CO2': 0.7,
        'C2H4O2': 0.3, 'SO4': 0.6, 'PO4': 0.6, 'C3H8O4': 0.2,
        '+Acetonitrile': 0.3, '+Acetate': 0.4, '+Na': 0.5, '+K': 0.3,
    },
    'multi_cleavage_penalty': {'glycosidic': 0.6, 'cross_ring': 0.35},
    # Alkali-cation-adducted fragments only (Na+/CID; from doi:10.1021/acs.jpca.6c00953): cross-ring dissociation needs reducing-end
    # ring-opening via the free anomeric O1, so it concentrates on the reducing-terminal residue.
    # X_1 cross-rings retain the reducing end and are the diagnostic class; cross-rings on
    # non-reducing-terminal residues are strongly disfavored. Detected per-fragment via the M_+Na
    # token, so protonated and negative-mode fragments (where internal 02A/24A are diagnostic) are untouched.
    'reducing_end_cross_ring': {'boost': 1.6, 'non_reducing_penalty': 0.2, 'adducts': {'+Na', '+K'}},
}
linkage_lability = {
    ('Neu5Ac', '2-3'): 1.0, ('Neu5Ac', '2-6'): 0.3, ('Neu5Ac', '2-8'): 0.5,
    ('Neu5Gc', '2-3'): 1.0, ('Neu5Gc', '2-6'): 0.3, ('Neu5Gc', '2-8'): 0.5,
    ('Kdn', '2-3'): 1.0, ('Kdn', '2-6'): 0.3,
    ('dHex', '1-2'): 0.7, ('dHex', '1-3'): 0.9, ('dHex', '1-4'): 0.6, ('dHex', '1-6'): 0.4,
}
DEFAULT_LABILITY = 0.5


def combine_global_mods(mods):
    """Canonical token for a combination of global modifications, e.g. ('H2O', 'H2O') -> '2H2O'"""
    counts = Counter(mods)
    return '|'.join(f"{n if n > 1 else ''}{name}" for name, n in sorted(counts.items()))


def parse_global_mod(global_mod):
    """Splits a global modification token back into its component modifications"""
    if not global_mod:
        return []
    components = []
    for chunk in global_mod.split('|'):
        multiple = re.match(r'(\d+)(.+)', chunk)
        components.extend([multiple.group(2)] * int(multiple.group(1)) if multiple else [chunk])
    return components


def global_mod_mass(global_mod, mode_mass = 0.0):
    """Total mass shift of a global modification token, adducts corrected for the charge carrier"""
    return sum(mono_attributes['Global']['mass'][x] - (mode_mass if x in ADDUCT_GLOBAL_MODS else 0)
               for x in parse_global_mod(global_mod))


def glycan_to_graph_monos(glycan):
    """Monosaccharide-only view of glycowork's glycan graph; every floating part ({...}) is placed at its first possible position\n
    | Arguments:
    | :-
    | glycan (string): IUPAC-condensed glycan sequence\n
    | Returns:
    | :-
    | (1) a dictionary of node : monosaccharide
    | (2) an adjacency matrix of size monosaccharide X monosaccharide
    | (3) a dictionary of node : monosaccharide/linkage
    """
    # get_possible_topologies places one floating part per call, and a part left floating breaks the even/odd node order
    while '{' in glycan:
        glycan = get_possible_topologies(glycan)[0]
    ggraph = glycan_to_nxGraph(glycan)
    all_mask_dic = nx.get_node_attributes(ggraph, 'string_labels')
    mono_mask_dic = {k // 2: v for k, v in all_mask_dic.items() if not k % 2}
    # A modified monosaccharide (GlcNAc6S, Neu5Ac9Ac, IdoA2S, GlcNS, ...) gets the tables of its base residue: each
    # substituent adds its mass and derivatization change to the cross-ring fragments keeping its position. One at an
    # unknown position (GalOS, ManOMe) is taken to sit on every fragment keeping a hydroxyl (the peracetylation sites) it
    # could occupy, as a sulfate carries the charge of the fragments that get observed
    for label in set(mono_mask_dic.values()):
        if (key := map_to_basic(label, obfuscate_ptm = False)) in mono_attributes:
            continue
        try:
            comp = glycan_to_composition(label)
        except ValueError:
            continue
        base = [k for k in comp if k not in SUBSTITUENTS]
        modification = get_modification(label)
        positions = re.findall(r'(\d|O)?(PCho|PEtN|Ac|Me|S|P)', modification)
        # glycowork's composition drops what it does not know (Qui3NAc becomes dHex, ManNAcA HexNAc), so a residue is
        # only derived if its substituents, an alditol, a stereo prefix, a lactone or an aglycone fully explain it
        if (len(base) != 1 or comp[base[0]] != 1 or base[0] not in derivatization_sites['permethylated'] or
                Counter(s for _, s in positions) != {k: v for k, v in comp.items() if k in SUBSTITUENTS and k != '-H2O'} or
                re.sub(r'(\d|O)?(PCho|PEtN|Ac|Me|S|P)|\d,\dlactone|^[DL]-|-ol$|^1(Ser|Thr|Asn|Cer)$', '', modification)):
            continue
        base = base[0]
        # a substituent without a number sits on the amine of a hexosamine (GlcNS), one on O at an unknown hydroxyl
        subs = [(s, int(p) if p.isdigit() else 2 if not p and get_core(label).endswith('N') else None) for p, s in
                positions] + [('-H2O', None)] * comp.get('-H2O', 0)
        base_attr = mono_attributes[base]
        frags = list(base_attr['mass'])
        hydroxyls = set(derivatization_sites['peracetylated'][base])
        in_frag = {f: [s for s, p in subs if p in base_attr['atoms'][f] or (p is None and hydroxyls & set(base_attr['atoms'][f]))]
                   for f in frags}
        mono_attributes[key] = {'atoms': {key if f == base else f: base_attr['atoms'][f] for f in frags}} | {
            prop: {key if f == base else f: base_attr[prop][f] + sum(SUBSTITUENTS[s][prop] for s in in_frag[f])
                   for f in frags} for prop in ['mass', *DERIVATIZATION_MASSES]}
    adj_matrix = np.zeros((len(mono_mask_dic), len(mono_mask_dic)), dtype = int)
    # glycowork alternates monosaccharide (even) and linkage (odd) nodes, with edges pointing parent -> linkage -> child
    for parent, link in ggraph.edges():
        if link % 2:
            adj_matrix[link // 2, parent // 2] = 1
    return mono_mask_dic, adj_matrix, all_mask_dic


def create_edge_labels(gr, all_dict):
    """Helper to create a dictionary linking graph edges with bond labels\n
    | Arguments:
    | :-
    | gr (networkx_object): graph to be modified
    | all_dict (dict): dictionary mapping original node format to bonds and monos\n
    | Returns:
    | :-
    | Returns a dict mapping each gr edge to its bond label
    """
    return {(e[0], e[1]): {'bond_label': all_dict[(e[0] * 2) + 1]} for e in gr.edges}


def mono_graph_to_nx(mono_graph, directed = True):
    """Modified version of glycan_to_nxGraph, converts a mono adjacency matrix into a networkx graph, adds bonds as edge labels, and terminal,reducing end labels\n
    | Arguments:
    | :-
    | mono_graph (string): output of glycan_to_graph_monos
    | directed (bool): if True, creates a directed graph with bonds pointing from leaf to reducing end ; default:True\n
    | Returns:
    | :-
    | Returns networkx graph object of a glycan made up of only monosaccharides
    """
    template = nx.DiGraph if directed else nx.Graph
    node_dict_mono, adj_matrix, all_dict = mono_graph
    if len(node_dict_mono) > 1:
        gr = nx.from_numpy_array(adj_matrix, create_using = template)
        for n1, n2, d in gr.edges(data = True):
            del d['weight']
    else:
        gr = template()
        gr.add_node(0)
    nx.set_node_attributes(gr, node_dict_mono, 'string_labels')
    nx.set_node_attributes(gr, {k: 'terminal' if gr.degree[k] == 1 else 'internal' for k in gr.nodes()}, 'termini')
    # Add safety check
    if len(gr.nodes) > 0:
        reducing_node = max(gr.nodes())
        nx.set_node_attributes(gr, {int(reducing_node): 2}, 'reducing_end')
    bond_dict = create_edge_labels(gr, all_dict)
    nx.set_edge_attributes(gr, bond_dict)
    return gr


def enumerate_subgraphs(nx_mono):
    """Returns the node sets of all connected induced subgraphs of a graph\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): monosaccharide only graph\n
    | Returns:
    | :-
    | Returns a list of node sets, one per connected induced subgraph (nx_mono.subgraph(node_set) gives the subgraph)
    """
    all_subgraphs = []
    if nx_mono.number_of_nodes() > 1:
        # one search collects all sizes, a search per size would repeat every smaller level; the stable sort by size
        # restores the order of concatenated per-size searches
        neighbor_dict = {v: set(nx_mono.predecessors(v)) | set(nx_mono.successors(v)) for v in nx_mono.nodes}
        for node in nx_mono.nodes():
            extend_subgraph({node}, {x for x in neighbor_dict[node] if x > node}, node, nx_mono.number_of_nodes() - 1,
                            all_subgraphs, neighbor_dict, nx_mono, all_sizes = True)
    return sorted(all_subgraphs, key = len)


def enumerate_k_graphs(nx_mono, k):
    """Finds all connected induced subgraphs of size k, implementation of Wernicke, S. (2005). A Faster Algorithm for Detecting Network Motifs. In: Casadio, R., Myers, G. (eds) Algorithms in Bioinformatics\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): monosaccharide only graph
    | k (int): size of subgraphs to be enumerated\n
    | Returns:
    | :-
    | Returns a list of all networkx subgraphs of size k
    """
    neighbor_dict = {v: set(nx_mono.predecessors(v)) | set(nx_mono.successors(v)) for v in nx_mono.nodes}
    k_subgraphs = []
    for node in nx_mono.nodes():
        node_neighbors = {x for x in neighbor_dict[node] if x > node}
        subgraph = {node}
        extend_subgraph(subgraph, node_neighbors, node, k, k_subgraphs, neighbor_dict, nx_mono)
    return k_subgraphs


def extend_subgraph(subgraph, extension, node, k, k_subgraphs, neighbor_dict, nx_mono, all_sizes = False):
    """Main recursive feature of enumerate_k_graphs, calls itself to grow subgraph via an arbitrary path until it reaches size k\n
    | Arguments:
    | :-
    | subgraph (set): nodes making up a connected induced subgraph
    | extension (set): nodes neighbouring the subgraph with a higher node label that the start node
    | node (set): single node used to start the subgraph and take neighbours higher than
    | k (int): size of subgraph at which to stop extending and add it to the list of subgraphs
    | k_subgraphs (list): list used to accumulate all subgraphs already found of size k
    | neighbor_dict (dict): mapping of all nodes and their neighbours in the original graph
    | nx_mono (networkx_object): the original monosaccharide only graph being searched
    | all_sizes (bool): whether to also collect every smaller subgraph passed on the way to size k, as node sets instead of subgraph views; default:False\n
    | Returns:
    | :-
    | Returns None
    """
    if len(subgraph) == k:
        k_subgraphs.append(subgraph if all_sizes else nx_mono.subgraph(subgraph))
        return None
    if all_sizes:
        k_subgraphs.append(subgraph)
    while extension:
        w = min(extension)
        extension.discard(w)
        exclusive_neighbors = get_exclusive_neighbors(w, subgraph, neighbor_dict)
        new_extension = extension | {x for x in exclusive_neighbors if x > node}
        extend_subgraph(subgraph | {w}, new_extension, node, k, k_subgraphs, neighbor_dict, nx_mono, all_sizes)


def get_exclusive_neighbors(w, subgraph, neighbor_dict):
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
    all_neighbors = {x for n in subgraph for x in neighbor_dict[n]}
    w_neighbors = {x for x in neighbor_dict[w]}
    exclusive_neighbors = w_neighbors - all_neighbors
    return exclusive_neighbors


def get_broken_bonds(subg, nx_mono, nx_edge_dict):
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
    subg_nodes = set(subg.nodes())
    # subg is an induced subgraph, so its floating bonds are exactly the edges with one end inside it
    return {bond: label['bond_label'] for bond, label in nx_edge_dict.items() if (bond[0] in subg_nodes) != (bond[1] in subg_nodes)}


def get_terminals(neighbor_dict, subg):
    """Determines all of the monosaccharides with fewer bonds than the original graph\n"""
    subg_nodes = set(subg)
    return [x for x in subg if not neighbor_dict[x] <= subg_nodes]


def atom_mods_init(subg, present_breakages, terminals, terminal_labels):
    """Creates the initial nested dict of each terminal node with floating bonds labeled 1 and the reducing end floating bond labeled 2\n
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
    for terminal, terminal_label in zip(terminals, terminal_labels):
        terminal_label = map_to_basic(terminal_label, obfuscate_ptm = False)
        atomic_mod_dict[terminal] = {y: 0 for y in mono_attributes[terminal_label]['atoms'][terminal_label]}
    for bond, bond_label in present_breakages.items():
        if bond_label == 'glycosite':
            if bond[0] in subg.nodes():
                atomic_mod_dict[bond[0]][3] = 1
            else:
                atomic_mod_dict[bond[1]][1] = 2
            continue
        if bond_label == 'peptide':
            if bond[0] in subg.nodes():
                atomic_mod_dict[bond[0]][6] = 3
            else:
                atomic_mod_dict[bond[1]][1] = 4
            continue
        elif bond[0] in subg.nodes():
            # the child's own carbon in the bond, with or without an anomeric letter (a2-3, or 1-5 in teichoic acids)
            red_breakage = int(bond_label.split('-')[0][-1])
            atomic_mod_dict[bond[0]][red_breakage] = 2
        else:
            # '?' linkages take the first free position, most common linkage first, so that two unknown
            # linkages on one residue cannot collapse onto the same atom
            free = atomic_mod_dict[bond[1]]
            breakage = int(bond_label[-1]) if bond_label[-1].isdigit() else next(
                (a for a in (3, 6, 2, 4, 5) if a in free and not free[a]), 3)
            atomic_mod_dict[bond[1]][breakage] = 1
    return atomic_mod_dict


def get_mono_mods_list(root_node, subg, terminals, terminal_labels, nx_edge_dict, allowed_X_cleavages,
                       disable_A_cross_rings = False):
    """Determines all possible cross-ring modifications for each node label in terminals\n
    | Arguments:
    | :-
    | root_node (int): node label which is the root of the directed nx_mono the subgraph comes from
    | subg (networkx_object): a subgraph
    | terminals (list): node labels of nodes with floating bonds
    | terminal_labels (list): string labels of nodes in terminals
    | atomic_mods (dict): nested dict of each terminal node with floating bonds labelled at each atom
    | nx_edge_dict (dict): a mapping of each edge in the original graph to its bond label
    | disable_A_cross_rings (bool): whether to strip out any A-type cross-rings; default: False\n
    | Returns:
    | :-
    | Returns a nested list with one list of modifications per terminal node
    """
    terminal_mods = []
    for node, label in zip(terminals, terminal_labels):
        basic_label = map_to_basic(label, obfuscate_ptm = False)
        if node == root_node:
            valid_A_frags = get_valid_A_frags(subg, node, label, nx_edge_dict)
            if disable_A_cross_rings:
                valid_A_frags = [x for x in valid_A_frags if x not in A_cross_rings]
            terminal_mods.append(valid_A_frags)
        elif subg.degree()[node] > 1:
            terminal_mods.append([label])
        else:
            terminal_mods.append(
                [x for x in mono_attributes[basic_label]['mass'] if x in allowed_X_cleavages or x == basic_label])
    return terminal_mods


def get_valid_A_frags(subg, node, label, nx_edge_dict):
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
    idx_label = map_to_basic(label, obfuscate_ptm = False)
    A_mods_list = [x for x in mono_attributes[idx_label]['mass'] if x in A_cross_rings or x == label]
    bond_numbers = set(int(nx_edge_dict[bond]['bond_label'][-1])
                       for bond in subg.in_edges(node)
                       if nx_edge_dict[bond]['bond_label'][-1].isdigit())
    valid_A_mods_list = [mod for mod in A_mods_list
                         if bond_numbers <= set(mono_attributes[idx_label]['atoms'][mod])]
    return valid_A_mods_list


def create_dict_perms(dicty):
    """Returns all bond permutations of an atom level dict with string labels describing the floating bonds\n
    | Arguments:
    | :-
    | dicty (dict): indicates which atoms on a monosaccharide have floating bonds\n
    | Returns:
    | :-
    | Returns a list of dicts corresponding to possible fragmentations on one node
    """
    dict_perms = []
    modded_atoms = [k for k, v in dicty.items() if v in bond_type_helper]
    perms = product(*(bond_type_helper[dicty[y]] for y in modded_atoms))
    dict_perms = [{**dicty, **dict(zip(modded_atoms, perm))} for perm in perms]
    return dict_perms


def generate_mod_permutations(terminals, terminal_labels, mono_mods_list, atomic_mod_dict_subg):
    """Determines all possible monosaccharide modifications and their respective atom level representations\n
    | Arguments:
    | :-
    | terminals (list): node labels of nodes with floating bonds
    | terminal_labels (list): string labels of nodes in terminals
    | mono_mods_list (list): a nested list with one list of modifications per terminal node
    | atomic_mod_dict_subg (dict): nested dict of each terminal node with floating bonds labelled at each atom\n
    | Returns:
    | :-
    | (1) a nested list of all possible cross ring level fragmentations for each terminal node
    | (2) a nested list of all possible bond fragmentation dictionaries for each terminal node
    """
    all_terminal_perms, all_mono_mods = [], []
    for node, label, mono_mods in zip(terminals, terminal_labels, mono_mods_list):
        label = map_to_basic(label, obfuscate_ptm = False)
        possible_node_atoms = [{k: v for k, v in atomic_mod_dict_subg[node].items() if
                                k in mono_attributes[label]['atoms'][map_to_basic(mod, obfuscate_ptm = False)]} for mod
                               in mono_mods]
        all_atom_dict_perms, all_mono_mod_perms = [], []
        for i, atom_dict in enumerate(possible_node_atoms):
            dict_perms = create_dict_perms(atom_dict)
            if label not in W_SIDE_CHAIN_LOSSES:
                dict_perms = [x for x in dict_perms if 'peptide_w' not in x.values()]
            all_atom_dict_perms.extend(dict_perms)
            all_mono_mod_perms.extend(len(dict_perms) * [mono_mods[i]])
        all_terminal_perms.append(all_atom_dict_perms)
        all_mono_mods.append(all_mono_mod_perms)
    return all_mono_mods, all_terminal_perms


def precalculate_mod_masses(all_mono_mods, all_terminal_perms, terminal_labels, global_mods,
                            sample_prep = 'underivatized', charge = -1):
    """Determines the masses of all possible monosaccharide modifications and their respective atom level representations\n
    | Arguments:
    | :-
    | all_mono_mods (list): all possible cross ring level fragmentations
    | all_terminal_perms (list): all possible bond fragmentation dictionaries
    | terminal_labels (list): string labels of nodes in terminals
    | global_mods (list): possible global modifications
    | sample_prep (string): underivatized/permethylated/peracetylated
    | charge (int): negative/positive charge state of precursor\n
    | Returns:
    | :-
    | (1) a nested list with the mass of each cross ring option per terminal node
    | (2) a nested list with the mass of each bond fragmentation option per terminal node
    | (3) a list of masses corresponding to each of the global mods
    """
    deriv_mass = DERIVATIZATION_MASSES.get(sample_prep, 0)
    all_mono_mod_masses = []
    for mods, label in zip(all_mono_mods, terminal_labels):
        label_basic = map_to_basic(label, obfuscate_ptm = False)
        masses = []
        for mod in mods:
            mod_basic = map_to_basic(mod, obfuscate_ptm = False)
            masses.append(mono_attributes[label_basic]['mass'][mod_basic] +
                          get_derivatization_count(label_basic, mod_basic, sample_prep) * deriv_mass)
        all_mono_mod_masses.append(masses)
    # a glycosidic cleavage frees the hydroxyl the bond had taken, which stays underivatized
    active_bond_masses = bond_masses | {'bond': -deriv_mass, 'no_bond': -(WATER_MASS + deriv_mass)} if deriv_mass else bond_masses
    all_atom_dict_masses = []
    for node, label in zip(all_terminal_perms, terminal_labels):
        node_dict_masses = []
        for mod in node:
            present_atom_mods = [active_bond_masses[x] for x in mod.values() if x in active_bond_masses]
            if 'peptide_w' in mod.values():
                present_atom_mods.append(-W_SIDE_CHAIN_LOSSES[label])
            node_dict_masses.append(sum(present_atom_mods))
        all_atom_dict_masses.append(node_dict_masses)
    mode_mass = -PROTON_MASS if charge < 0 else PROTON_MASS
    global_mods_mass = [global_mod_mass(x, mode_mass) for x in global_mods[1:]]
    return all_mono_mod_masses, all_atom_dict_masses, global_mods_mass


def temporary_root_calc_func(subg, parent_graph = None):
    """Determines whether and where to add label and extra oxygen masses to the reducing end of a glycan\n"""
    reducing_ends = [x for x in nx.get_node_attributes(subg, 'reducing_end')]
    if not reducing_ends:
        return False, None
    graph = subg if parent_graph is None else parent_graph
    bond_labels = nx.get_edge_attributes(graph, 'bond_label')
    for red_end in reducing_ends:
        for x in graph.in_edges(red_end):
            if bond_labels.get(x) == 'glycosite':
                return False, None
    return True, red_end


def preliminary_calculate_mass(mono_mods_mass, atom_mods_mass, global_mods_mass, terminals,
                               inner_mass, bonus_root_mass, bonus_root_node, mass_tag, charge, mono_mod_perms,
                               perm_indices, sample_prep = 'underivatized', root_label = None):
    """Determines the mass of every requested permutation of monosaccharide, atom, and global modification\n
    | Arguments:
    | :-
    | mono_mods_mass (list): nested list with the mass of each cross ring option per terminal node
    | atom_mods_mass (list): nested list with the mass of each bond fragmentation option per terminal node
    | global_mods_mass (list): masses corresponding to each of the global mods
    | terminals (list): string labels of nodes in terminals
    | inner_mass (float): total mass of non-terminal nodes in subgraph
    | true_root_node (int): the node label corresponding to the root of the parent glycan
    | mass_tag (float): mass of the glycan label or reducing end modification
    | charge (int): assumed charge of glycan
    | perm_indices (array): one row per permutation to calculate, holding the option index of each terminal node
    | sample_prep (string): underivatized/permethylated/peracetylated\n
    | Returns:
    | :-
    | Returns a list every single mass of each modification combination for each cross ring combination
    """
    mode_mass = -PROTON_MASS if charge < 0 else PROTON_MASS
    bonus_pep_mass = WATER_MASS if [x for x in terminals if isinstance(x, str) if x.split('-')[0] == '0'] else 0
    mono_arr = np.stack([np.array(x)[perm_indices[:, i]] for i, x in enumerate(mono_mods_mass)], axis = 1)
    atom_arr = np.stack([np.array(x)[perm_indices[:, i]] for i, x in enumerate(atom_mods_mass)], axis = 1)
    base_masses = inner_mass + mode_mass + bonus_pep_mass + mono_arr.sum(axis = 1) + atom_arr.sum(axis = 1)
    if bonus_root_mass:
        root_node_idx = terminals.index(bonus_root_node)
        root_mods = mono_mod_perms[root_node_idx]
        is_not_A = np.array([rm not in A_cross_rings for rm in root_mods])
        # a derivatized reducing end carries one more group, and an alditol (mass_tag of 2 H) one more again
        re_deriv = DERIVATIZATION_MASSES.get(sample_prep, 0) * (1 + (abs(mass_tag - 2 * HYDROGEN_MASS) < 0.01))
        bonus = np.where(is_not_A, WATER_MASS + mass_tag + re_deriv, 0.0)
        if root_label is not None:
            root_basic = map_to_basic(root_label, obfuscate_ptm = False)
            for idx in np.where(~is_not_A)[0]:
                rm = root_mods[idx]
                if 1 in mono_attributes.get(root_basic, {}).get('atoms', {}).get(
                        map_to_basic(rm, obfuscate_ptm = False), []):
                    bonus[idx] += mass_tag + re_deriv
        base_masses += bonus[perm_indices[:, root_node_idx]]
    if not global_mods_mass:
        return base_masses.tolist()
    global_arr = np.array(global_mods_mass)
    expanded = base_masses[:, np.newaxis] + np.concatenate([[0.0], global_arr])
    return expanded.ravel().tolist()


def add_to_subgraph_fragments(subgraph_fragments, nx_mono_list, mass_list):
    """Helper to add lists of subgraphs and their respective masses to a dict\n
    | Arguments:
    | :-
    | subg_frags (dict): lists of networkx subgraphs indexed by their mass
    | nx_mono_list (list): list of networkx objects to be added to subgraph_fragments
    | mass_list (list): respective masses of the networkx objects to be added to subgraph_fragments\n
    | Returns:
    | :-
    | Returns an updated subgraph_fragments dict
    """
    for nx_mono, mass in zip(nx_mono_list, mass_list):
        subgraph_fragments.setdefault(mass, []).append(nx_mono)
    return subgraph_fragments


GLYCAN_ONLY_GLOBAL_MODS = {'CH2O', 'C2H2O', 'C2H4O2', 'C3H8O4', 'CO2', 'SO4', 'PO4'}
PEPTIDE_ONLY_GLOBAL_MODS = {'NH3'}
ADDUCT_GLOBAL_MODS = {'+Na', '+K', '+Acetate', '+Acetonitrile'}
# Only small neutral losses realistically occur more than once on a single fragment; combining the
# bulkier cross-ring losses would inflate the search space without describing real chemistry
REPEATABLE_GLOBAL_MODS = ('H2O', 'NH3')
BASIC_RESIDUES = {'H', 'K', 'R', 'j', 'k'}


def update_global_mods(subg, global_mods, special_residues):
    """Returns the valid list of global modifications for a given subgraph\n
    | Arguments:
    | :-
    | subg (networkx_object): a subgraph
    | node_dict (dict): a dictionary relating the integer label of each node with the monosaccharide it represents\n
    | charge (int): the assumed charge of the glycan\n
    | Returns:
    | :-
    | Returns a list of modification names
    """
    all_labels = list(nx.get_node_attributes(subg, 'string_labels').values())
    node_labels = ''.join(v for v in all_labels if len(v) > 1)
    excluded = set()
    if not node_labels:
        excluded |= GLYCAN_ONLY_GLOBAL_MODS
    if all(len(v) > 1 for v in all_labels):
        excluded |= PEPTIDE_ONLY_GLOBAL_MODS
    subg_global_mods = [x for x in global_mods if not (excluded & set(parse_global_mod(x)))]
    present_specials = [x for x in special_residues if x in node_labels]
    if not present_specials:
        return subg_global_mods
    if any(k in present_specials for k in ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn']):
        subg_global_mods.append('CO2')
    if 'S' in present_specials:
        subg_global_mods.append('SO4')
    if 'P' in node_labels:
        subg_global_mods.append('PO4')
    return subg_global_mods


def mod_count(node_mod, global_mod):
    """Given a description of subgraph modifications, returns a summed count of modifications\n
    | Arguments:
    | :-
    | node_mod (nested list): nested list of (i) monosaccharide level modifications and (ii) atom level modifications
    | global_mod (string): description of global modification, or None if no modification\n
    | Returns:
    | :-
    | Returns a sum of modifications
    """
    c = 1 if global_mod is not None else 0
    c += sum([1 for k in node_mod[0] if k in A_cross_rings or k in X_cross_rings])
    c += sum(1 for n in node_mod[1] for k in n.values() if isinstance(k, str))
    return c


def extend_masses(fragment_masses, charge):
    """Extends a list of masses with the additional masses to include multiply charged fragments in the filter\n
    | Arguments:
    | :-
    | fragment_masses (list): a list of observed masses to be searched for possible fragments
    | charge (int): the charge to use when calculating multiply charged masses\n
    | Returns:
    | :-
    | Returns a list containing both the input masses and the other masses at which to search to assign multiply charged fragments to the inital masses
    """
    if abs(charge) == 1:
        return fragment_masses
    modifier = np.sign(charge)
    all_masses = list(fragment_masses)
    for z in range(2, abs(charge) + 1):
        z_masses = [(k * z) - (z - 1) * PROTON_MASS * modifier for k in fragment_masses]
        all_masses.extend(z_masses)
    return all_masses


def annotate_subgraph(subg, node_mod, global_mod, terminals):
    """Applies the node, atom, and global modification attributes to a subgraph\n
    | Arguments:
    | :-
    | subg (networkx_object): a graph or subgraph of monosaccharides
    | node_mod (list): a nested list containing cleavage type at each terminal node and atom dictionary at each terminal node
    | global_mod (string): the chemical species globally lost or gained by the graph
    | terminals (list): the range around the observed mass in which constrain potential fragments\n
    | Returns:
    | :-
    | Returns a copy of the input subgraph with networkx node attributes describing the modifications
    """
    mod_subg = subg.copy()
    nx.set_node_attributes(mod_subg, dict(zip(terminals, node_mod[0])), 'mod_labels')
    nx.set_node_attributes(mod_subg, dict(zip(terminals, node_mod[1])), 'atomic_mod_dict')
    if global_mod:
        nx.set_node_attributes(mod_subg, [global_mod], 'global_mod')
    return mod_subg


def generate_atomic_frags(nx_mono, global_mods, special_residues, allowed_X_cleavages, max_cleavages = 3,
                          fragment_masses = [], subgraphs = None,
                          threshold = 0.5, mass_tag = None, charge = -1, sample_prep = 'underivatized',
                          disable_A_cross_rings = False):
    """Calculates the graph and mass of all possible fragments of the input\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): the original monosaccharide only graph
    | global_mods (list):
    | special_residues (list):
    | allowed_X_cleavages (list):
    | max_cleavages (int): maximum number of allowed concurrent fragmentations per mass; default:3
    | fragment_masses (list): all masses which are to be annotated with a fragment name
    | threshold (float): the range around the observed mass in which constrain potential fragments
    | mass_tag (float): mass of the glycan label or reducing end modification; default:2.0156
    | charge (int): the maximum possible charge on the fragments to be matched; default:-1
    | sample_prep (string): underivatized/permethylated/peracetylated
    | disable_A_cross_rings (bool): whether to strip out any A-type cross-rings; default: False\n
    | Returns:
    | :-
    | Returns a dict of lists of networkx subgraphs
    """
    if mass_tag is None:
        mass_tag = 2 * HYDROGEN_MASS
    charge_masses = np.array(extend_masses(fragment_masses, charge))
    sorted_charge_masses = sorted(charge_masses)
    unfiltered = not len(charge_masses)
    threshold = abs(threshold)
    true_root_node = [v for v, d in nx_mono.out_degree() if d == 0][0]
    all_other_terminals = {node for node in nx_mono.nodes() if nx_mono.degree()[node] < 2 or node == true_root_node}
    nx_edge_dict = {(node[0], node[1]): node[2] for node in nx_mono.edges(data = True)}
    node_dict = nx.get_node_attributes(nx_mono, 'string_labels')
    node_dict_basic = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in node_dict.items()}
    subgraph_fragments = {}
    subgraphs = (enumerate_subgraphs(nx_mono) + [nx_mono]) if subgraphs is None else subgraphs
    present_global_masses = [global_mod_mass(x, -PROTON_MASS if charge < 0 else PROTON_MASS)
                             for x in global_mods] + [0.0]
    max_global_mass = max(present_global_masses)
    min_global_mass = min(present_global_masses)
    # degree lookups on subgraph views are slow, so terminals are found via neighbor sets of the parent graph
    neighbor_dict = {v: set(nx_mono.predecessors(v)) | set(nx_mono.successors(v)) for v in nx_mono.nodes}
    deriv_mass = DERIVATIZATION_MASSES.get(sample_prep, 0)
    full_masses = {k: mono_attributes[v]['mass'][v] + get_derivatization_count(v, v, sample_prep) * deriv_mass for k, v in
                   node_dict_basic.items()}
    # a derivatized glycosidic cleavage loses one more group, which the lower bound has to allow for
    min_bond_mass = min(bond_masses.values()) - deriv_mass - max(W_SIDE_CHAIN_LOSSES.values())
    peptide_graph = isinstance(true_root_node, str)
    for i, subg in enumerate(subgraphs):
        # most node sets from enumerate_subgraphs have too many cleavages, which their nodes alone tell, so only the rest get a subgraph view
        # (slow to create; built from the same node set as before, so it iterates its nodes in the same order)
        if isinstance(subg, set):
            if sum(not neighbor_dict[x] <= subg and x not in all_other_terminals for x in subg) > max_cleavages:
                continue
            subg = nx_mono.subgraph(subg)
        # a view of few nodes iterates them in set order, which for the string nodes of a glycopeptide follows PYTHONHASHSEED and decided
        # between equally good fragments (02X_5_Alpha or 02X_5_Beta), so those are walked in the order of the parent graph
        nodes = [v for v in nx_mono if v in subg] if peptide_graph else subg
        terminals = get_terminals(neighbor_dict, nodes)
        new_terminals = [x for x in terminals if x not in all_other_terminals]
        if len(new_terminals) > max_cleavages:
            continue
        other_terminals = [x for x in nodes if x in all_other_terminals and x not in terminals]
        terminals = terminals + other_terminals
        # every glycosidic bond takes one derivatizable group of the residue it is attached to (a view's edges are slow to count, so only if derivatized)
        inner_mass = sum(full_masses[m] for m in nodes if m not in terminals) - (subg.number_of_edges() * deriv_mass if deriv_mass else 0)
        max_graph_mass = inner_mass + sum(full_masses[m] for m in terminals) + WATER_MASS * len(terminals)
        max_graph_mass += max_global_mass + max(mass_tag, 0) + PROTON_MASS + 2 * deriv_mass
        min_terminal_mass = sum(
            min(mono_attributes[node_dict_basic[m]]['mass'].values()) + min_bond_mass for m in terminals)
        min_graph_mass = inner_mass + min_terminal_mass + min_global_mass
        avg_graph_mass = (min_graph_mass + max_graph_mass) / 2
        graph_mass_thresh = (max_graph_mass - min_graph_mass) / 2 + threshold
        if not unfiltered:
            lo = bisect.bisect_left(sorted_charge_masses, avg_graph_mass - graph_mass_thresh)
            if lo >= len(sorted_charge_masses) or sorted_charge_masses[lo] > avg_graph_mass + graph_mass_thresh:
                continue
        # a real copy of the surviving subgraph view makes every later degree/edge query and fragment copy far cheaper; glycopeptide
        # subgraphs are copied in parent order, as the first reducing end temporary_root_calc_func finds depends on it
        if peptide_graph:
            subg_copy = nx.DiGraph()
            subg_copy.add_nodes_from((v, nx_mono.nodes[v]) for v in nodes)
            subg_copy.add_edges_from((u, v, d) for u, v, d in nx_mono.edges(data = True) if u in subg and v in subg)
            subg = subg_copy
        else:
            # what copying the view gives (its node order, each node's out-edges in parent order, attribute dicts copied), built straight
            # from the parent graph, as copying through the view's filtered adjacency is slow
            subg_copy = nx.DiGraph()
            subg_copy.graph.update(nx_mono.graph)
            subg_copy.add_nodes_from((v, nx_mono.nodes[v].copy()) for v in subg)
            subg_copy.add_edges_from(
                (u, v, d.copy()) for u in subg_copy for v, d in nx_mono.succ[u].items() if v in subg_copy)
            subg = subg_copy
        bonus_root_mass, bonus_root_node = temporary_root_calc_func(subg, nx_mono)
        terminal_labels = [node_dict_basic[x] for x in terminals]
        subg_global_mods = update_global_mods(subg, global_mods, special_residues)
        present_breakages = get_broken_bonds(subg, nx_mono, nx_edge_dict)
        # glycopeptide subgraphs have two sinks (C-terminal residue, glycan reducing end); pick in parent graph order
        # like true_root_node, as the subgraph view iterates string node labels in hash-randomized set order
        root_node = next(v for v in nx_mono if v in subg and subg.out_degree(v) == 0)
        atomic_mod_dict_subg = atom_mods_init(subg, present_breakages, terminals, terminal_labels)
        mono_mods_list = get_mono_mods_list(root_node, subg, terminals, terminal_labels, nx_edge_dict,
                                            allowed_X_cleavages, disable_A_cross_rings)
        mono_mod_perms, atom_dict_perms = generate_mod_permutations(terminals, terminal_labels, mono_mods_list,
                                                                    atomic_mod_dict_subg)
        mono_masses, atom_masses, global_masses = precalculate_mod_masses(mono_mod_perms, atom_dict_perms,
                                                                          terminal_labels, subg_global_mods,
                                                                          sample_prep = sample_prep, charge = charge)
        # nearly all permutations exceed max_cleavages, so cleavages are counted before any mass is calculated
        inner_counts = np.zeros(1, dtype = int)
        for mods, atom_dicts in zip(mono_mod_perms, atom_dict_perms):
            inner_counts = (inner_counts[:, np.newaxis] + [mod_count([[mod], [atom_dict]], None) for mod, atom_dict in
                                                           zip(mods, atom_dicts)]).ravel()
        global_counts = np.array([mod_count([[], []], x) for x in subg_global_mods], dtype = int)
        inner_idx = np.flatnonzero(inner_counts + global_counts.min() <= max_cleavages)
        if inner_idx.size == 0:
            continue
        root_label = node_dict.get(bonus_root_node) if bonus_root_mass else None
        initial_masses = np.array(
            preliminary_calculate_mass(mono_masses, atom_masses, global_masses, terminals, inner_mass, bonus_root_mass,
                                       bonus_root_node, mass_tag, charge, mono_mod_perms,
                                       vectorized_lazy_product_indices(mono_mod_perms, inner_idx)[:, ::-1],
                                       sample_prep = sample_prep, root_label = root_label))
        m_thresh = 1 if charge < 0 else 2
        counts = (inner_counts[inner_idx, np.newaxis] + global_counts).ravel()
        has_adduct = np.tile([bool(ADDUCT_GLOBAL_MODS & set(parse_global_mod(x))) for x in subg_global_mods], inner_idx.size)
        keep = (counts <= max_cleavages) & ~((counts > m_thresh) & has_adduct)
        if not unfiltered:
            keep &= check_masses(charge_masses, initial_masses, threshold)
        if not keep.any():
            continue
        valid_idx = (inner_idx[:, np.newaxis] * len(subg_global_mods) + np.arange(len(subg_global_mods))).ravel()[keep]
        permutation_list = nested_lazy_product_vect(mono_mod_perms, atom_dict_perms, subg_global_mods, valid_idx)
        for perms, mass in zip(permutation_list, np.round(initial_masses[keep], 5)):
            annotated_subg = annotate_subgraph(subg, perms[:2], perms[2], terminals)
            subgraph_fragments = add_to_subgraph_fragments(subgraph_fragments, [annotated_subg], [mass])
    return subgraph_fragments


def rank_chains(nx_mono):
    """Ranks each glycan chain (terminal to reducing end) by mass in the form alpha, beta, etc.\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): the original monosaccharide only graph\n
    | Returns:
    | :-
    | A iterable of tuples containing the string rank and a list of integer node labels describing the chain
    """
    node_dict = nx.get_node_attributes(nx_mono, 'string_labels')
    og_root = [v for v, d in nx_mono.out_degree() if d == 0][0]
    og_leaves = set(v for v, d in nx_mono.in_degree() if d == 0)
    leaf_chains = []
    for og_leaf in og_leaves:
        leaf_chains.append(nx.shortest_path(nx_mono, source = og_leaf))
    main_chains = sorted([branch_path[og_root] for branch_path in leaf_chains],
                         key = lambda x: sum([mono_attributes[map_to_basic(node_dict[n], obfuscate_ptm = False)][
                                                  'mass'][map_to_basic(node_dict[n], obfuscate_ptm = False)] for n in
                                              x]), reverse = True)
    return zip(ranks, main_chains)


def domon_costello_to_node_labels(fragment, chain_rank):
    """Determines the cleavage points on each different glycan chain\n
    | Arguments:
    | :-
    | fragment (list): containing underscore separated string forms of Domon-Costello e.g(['Y_1_Alpha'])
    | chain_rank (dict): a dictionary keyed by rank with each pointing to a list of integer node labels representing the glycan chain\n
    | Returns:
    | :-
    | (1) a dict keyed by integer node label with each pointing to a cleavage type
    | (2) an integer node label at which the B or C type cleavage occurred, otherwise None
    | (3) a string describing the global mass change to the glycan, otherwise None
    """
    skelly_dict = {}
    global_mod = None
    post_mono = None
    for cut in fragment:
        if cut.startswith('M'):
            global_mod = cut.split('_')[-1]
            continue
        cut_type, cut_num, chain_label = cut.split('_')
        cut_type_last_char = cut_type[-1]
        cut_num = int(cut_num)
        chain = chain_rank[chain_label]
        if cut_type_last_char in 'YZ':
            mono = chain[::-1][cut_num]
        elif cut_type_last_char == 'X':
            mono = chain[::-1][cut_num - 1]
        elif cut_type_last_char in 'BC':
            mono = chain[cut_num]
            post_mono = chain[cut_num - 1]
        elif cut_type_last_char == 'A':
            mono = chain[cut_num - 1]
        skelly_dict[mono] = cut_type
    return skelly_dict, post_mono, global_mod


def node_labels_to_domon_costello(cuts, chain_rank, global_mods = {}):
    """Converts the cleavages, ranks, and global modifications into the Domon & Costello fragment name\n
    | Arguments:
    | :-
    | cuts (list): a list of tuples each containing the cleavage type, and related integer node labels
    | chain_rank (list): a list of tuples each containing rank and a list of integer node labels representing the glycan chain
    | global_mods (dict): the output of get_node_attributes(subg, 'global_mod') a dict containing integer node labels each pointing to a global mass change\n
    | Returns:
    | :-
    | Returns a list containing all of the cleavages making up the fragment in Domon-Costello form
    """
    dc_cuts = []
    for cut in cuts:
        cut_type = cut_type_dict[cut[0]]
        if cut_type[-1] in {'B', 'C', 'A'}:
            cut_rank, cut_chain = [(rank, chain) for rank, chain in chain_rank if cut[1] in chain][0]
            cut_number = cut_chain.index(cut[1]) + 1
        elif cut_type[-1] in {'Y', 'Z'}:
            cut_rank, cut_chain = [(rank, chain) for rank, chain in chain_rank if cut[1] in chain][0]
            cut_number = cut_chain[::-1].index(cut[2]) + 1
        elif cut_type[-1] in {'X'}:
            cut_rank, cut_chain = [(rank, chain) for rank, chain in chain_rank if cut[1] in chain][0]
            cut_number = cut_chain[::-1].index(cut[1]) + 1
        dc_cuts.append(f"{cut_type}_{cut_number}_{cut_rank}")
    if global_mods:
        global_mods = list(global_mods.values())[0][0]
        dc_cuts.append(f"M_{global_mods}")
    return dc_cuts or ['M']


def find_main_chain(subgraph, leaves, root_node):
    """Calculates the main chain of a subgraph\n
    | Arguments:
    | :-
    | subg (networkx_object): a subgraph
    | leaves (list): integer labels of leaf nodes
    | root_node (list): integer label of the root node\n
    | Returns:
    | :-
    | Returns a list of integer node labels representing the inputs main chain
    """
    bond_dict = nx.get_edge_attributes(subgraph, 'bond_label')
    all_paths = [path for leaf in leaves for path in nx.all_simple_paths(subgraph, source = leaf, target = root_node)]
    main_chain = max(all_paths, key = len)
    main_chain_len = len(main_chain)
    main_chains = [k for k in all_paths if len(k) == main_chain_len]
    for i in range(main_chain_len)[::-1]:
        step = [c[i] for c in main_chains]
        if len(set(step)) == 1:
            pass
        else:
            bond_nums = [bond_dict[i][-1] for j in step for i in subgraph.edges if j == i[0]]
            main_chains = [main_chains[i] for i, x in enumerate(bond_nums) if x == min(bond_nums)]
    return main_chains[0]


def subgraph_to_label_skeleton(sub_g):
    """Breaks up a graph object into the skeleton of the IUPAC condensed nomenclature\n
    | Arguments:
    | :-
    | subg (networkx_object): a graph or subgraph of monosaccharides\n
    | Returns:
    | :-
    | Returns a list of integer node labels and branching brackets in the same order as the IUPAC condensed string
    """
    if len(sub_g.nodes) == 1:
        return [str(next(iter(sub_g.nodes)))]
    root_node = next(k for k, v in sub_g.out_degree if v == 0)
    leaves = [v for v, d in sub_g.in_degree() if d == 0]
    main_chain = []
    n_skelly = []
    main_chain = [str(i) for i in find_main_chain(sub_g, leaves, root_node)]
    n_skelly = main_chain
    while set([str(k) for k in sub_g.nodes]) != set([k for k in n_skelly if k.isnumeric()]):
        for i in [m for m in main_chain if m.isnumeric()][::-1]:
            new_root_node = [x for x in nx.all_neighbors(sub_g, int(i)) if str(x) not in main_chain]
            if len(new_root_node) < 1:
                continue
            new_root_node = new_root_node[0]
            new_leaves = set(nx.ancestors(sub_g, new_root_node)) & set(leaves)
            if len(new_leaves) == 0:
                new_chain = [new_root_node]
            else:
                new_chain = find_main_chain(sub_g, new_leaves, new_root_node)
            n_skelly[n_skelly.index(i): n_skelly.index(i)] = ['['] + [str(k) for k in new_chain] + [']']
        main_chain = n_skelly
    return n_skelly


def label_skeleton_to_string(n_skelly, sub_g):
    """Converts a glycan skeleton into a canonical string representation\n
    | Arguments:
    | :-
    | n_skelly (list): a list of integer node labels and branching brackets in the same order as the IUPAC condensed string
    | subg (networkx_object): a graph or subgraph of monosaccharides\n
    | Returns:
    | :-
    | Returns an IUPAC condensed representation of the input graph, in the case of fragment graphs it returns the closest canonical string representation
    """
    bond_dict = nx.get_edge_attributes(sub_g, 'bond_label')
    mono_to_bond = {k[0]: v for k, v in bond_dict.items()}
    string_labels = nx.get_node_attributes(sub_g, 'string_labels')
    for i in n_skelly:
        if i.isnumeric():
            if int(i) in mono_to_bond:
                n_skelly.insert(n_skelly.index(i) + 1, f'({mono_to_bond[int(i)]})')
            n_skelly[n_skelly.index(i)] = string_labels[int(i)]
    return ''.join(n_skelly)


def mono_frag_to_string(sub_g):
    """Converts a monosaccharide graph to a string\n
    | Arguments:
    | :-
    | subg (networkx_object): a graph or subgraph of monosaccharides\n
    | Returns:
    | :-
    | Returns an IUPAC condensed representation of the input graph, in the case of fragment graphs it returns the closest canonical string representation
    """
    return label_skeleton_to_string(subgraph_to_label_skeleton(sub_g), sub_g)


def domon_costello_to_fragIUPAC(glycan_string, fragment):
    """Converts a glycan string and a Domon-Costello fragment name into a fragmented version of the orignal string\n
    | Arguments:
    | :-
    | glycan_string (string): glycan in IUPAC-condensed format
    | fragment (list): underscore separated string form of Domon-Costello e.g(['Y_1_Alpha'])\n
    | Returns:
    | :-
    | Returns the fragmented glycan in a version of IUPAC condensed which is GlycoDraw compatible
    """
    global_mod = None
    mono_graph = glycan_to_graph_monos(glycan_string)
    nx_mono = mono_graph_to_nx(mono_graph, directed = True)
    chain_rank = dict(rank_chains(nx_mono))
    skelly_dict, post_mono, global_mod = domon_costello_to_node_labels(fragment, chain_rank)
    excluded_nodes = set()
    for k, v in skelly_dict.items():
        if v[-1] == 'A':
            excluded_nodes.update(nx_mono.nodes() ^ nx.ancestors(nx_mono, k).union({k}))
        elif v[-1] in {'B', 'C'}:
            excluded_nodes.update(nx_mono.nodes() ^ nx.ancestors(nx_mono, post_mono).union({post_mono, k}))
        elif v[-1] in {'X', 'Y', 'Z'}:
            excluded_nodes.update(nx.ancestors(nx_mono, k))
    frag_subg = nx_mono.subgraph(set(nx_mono.nodes()) ^ set(excluded_nodes))
    label_skelly = subgraph_to_label_skeleton(frag_subg)
    skelly_dict = {str(k): v for k, v in skelly_dict.items()}
    mono_to_bond = {str(k[0]): v for k, v in nx.get_edge_attributes(frag_subg, 'bond_label').items()}
    node_dict = {str(k): v for k, v in nx.get_node_attributes(frag_subg, 'string_labels').items()}
    for i in label_skelly[::-1]:
        if i in mono_to_bond:
            label_skelly.insert(label_skelly.index(i) + 1, f'({mono_to_bond[i]})')
        if i in skelly_dict:
            label_skelly[label_skelly.index(i)] = skelly_dict[i]
        elif i in node_dict:
            label_skelly[label_skelly.index(i)] = node_dict[i]
    full_skelly = ''.join(label_skelly)
    if global_mod:
        global_mod_list = list(global_mod)
        for i, char in enumerate(global_mod_list):
            if char.isnumeric():
                global_mod_list[i] = chr(0x2080 + int(char))
        format_global_mod = ''.join(global_mod_list)
        full_skelly = '{- ' + format_global_mod + '}' + full_skelly
    return full_skelly


def domon_costello_to_html(dc_name):
    """Converts a Domon-Costello fragment name to a prettified HTML string\n
    | Arguments:
    | :-
    | dc_name (list): a list of Domon-Costello cleavage names\n
    | Returns:
    | :-
    | Returns a HTML ready string containing the correctly formatted superscript and subscript elements of the fragment name
    """
    html_name = []
    for nom in dc_name:
        html_nom = nom
        html_nom_parts = html_nom.split('_')
        if len(html_nom_parts) == 3:
            branch = html_nom_parts[2]
            branch_symbol = branch[0].lower() + branch[1:]
            html_nom = html_nom.replace(f'_{branch}', f"<sub>&{branch_symbol};</sub>")
            branch_number = html_nom_parts[1]
            html_nom = html_nom.replace(f'_{branch_number}', f"<sub>{branch_number}</sub>")
            frag_type = html_nom_parts[0]
            if len(frag_type) > 1:
                html_nom = html_nom.replace(f'{frag_type}', f"<sup>{frag_type[0]},{frag_type[1]}</sup>{frag_type[2]}")
        if len(html_nom_parts) == 2:
            mass_loss = list(html_nom_parts[1])
            for i, char in enumerate(mass_loss):
                if char.isnumeric():
                    mass_loss[i] = f"<sub>{char}</sub>"
            html_nom = html_nom.replace(f'_{html_nom_parts[1]}', f" - {''.join(mass_loss)}")
        html_name.append(html_nom)
    return ", ".join(html_name)


def subgraphs_to_domon_costello(nx_mono, subgs, chain_rank = None):
    """Converts the subgraphs of a given graph object into their canonical Domon & Costello fragment names\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): the original monosaccharide only graph
    | subg (list): a list of modified networkx subgraphs\n
    | Returns:
    | :-
    | Returns a nested list with one list of fragment labels for each subgraph
    """
    ion_names = []
    node_dict = nx.get_node_attributes(nx_mono, 'string_labels')
    node_dict = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in node_dict.items()}
    if chain_rank is None:
        chain_rank = list(rank_chains(nx_mono))
    in_children = {}
    for bonding_node, bonded_node, atts in nx_mono.edges(data = True):
        in_children.setdefault(bonded_node, []).append((bonding_node, atts['bond_label'][-1]))
    for subg in subgs:
        cuts = []
        # plain reads of the node data, as nx.get_node_attributes costs more than the naming itself
        node_data = subg.nodes(data = True)
        global_mods = {n: d['global_mod'] for n, d in node_data if 'global_mod' in d}
        mono_mod_dict = {n: d['mod_labels'] for n, d in node_data if 'mod_labels' in d}
        atomic_mod_dict = {n: d['atomic_mod_dict'] for n, d in node_data if 'atomic_mod_dict' in d}
        for node, atom_mods in atomic_mod_dict.items():
            cut_children = set()
            for atom, atom_mod in atom_mods.items():
                if atom_mod in {'bond', 'no_bond'}:
                    children = [(c, pos) for c, pos in in_children.get(node, []) if c not in cut_children]
                    cut_node = [c for c, pos in children if pos == str(atom)] or [c for c, pos in children if
                                                                                  pos == '?']
                    if cut_node:
                        cut_children.add(cut_node[0])
                        cuts.append((atom_mod, cut_node[0], node))
                if atom_mod in {'red_bond', 'red_no_bond'}:
                    cuts.append((atom_mod, node))
        cross_rings = [(v, k) for k, v in mono_mod_dict.items() if (k, v) not in node_dict.items()]
        cuts.extend(cross_rings)
        dc_cuts = node_labels_to_domon_costello(cuts, chain_rank, global_mods = global_mods)
        ion_names.append(sorted(dc_cuts))
    return ion_names


def score_fragment_prior(dc_name, charge):
    """Calculates an empirical prior score for a Domon-Costello fragment based on known fragmentation tendencies\n
    | Arguments:
    | :-
    | dc_name (list): list of Domon-Costello cleavage names making up a fragment
    | charge (int): charge state of the precursor ion\n
    | Returns:
    | :-
    | Returns a float score where higher values indicate more commonly observed fragmentation patterns
    """
    if not dc_name:
        return 0.0
    mode = np.sign(charge)
    cleavage_weights = fragmentation_priors['cleavage_type'].get(mode, fragmentation_priors['cleavage_type'][-1])
    re_weights = fragmentation_priors['reducing_end_cross_ring']
    cation_adduct = any(c.startswith('M_') and c[2:] in re_weights['adducts'] for c in dc_name)
    score = 0.0
    n_cleavages = 0
    for cut in dc_name:
        parts = cut.split('_')
        if parts[0] == 'M':
            component_scores = [fragmentation_priors['global_mod'].get(x, 0.2)
                                for x in parse_global_mod('_'.join(parts[1:]))]
            score += math.prod(component_scores) if component_scores else 0.2
            continue
        cut_score = cleavage_weights.get(parts[0], 0.1)
        if cation_adduct and (parts[0] in A_cross_rings or parts[0] in X_cross_rings):
            cut_score *= re_weights['boost'] if parts[0] in X_cross_rings and parts[1] == '1' else re_weights[
                'non_reducing_penalty']
        score += cut_score
        n_cleavages += 1
    if n_cleavages > 1:
        n_cross = sum(1 for c in dc_name if c.split('_')[0] in A_cross_rings or c.split('_')[0] in X_cross_rings)
        n_glyco = n_cleavages - n_cross
        penalties = fragmentation_priors['multi_cleavage_penalty']
        score *= penalties['glycosidic'] ** (max(n_glyco - 1, 0) if n_glyco else 0)
        score *= penalties['cross_ring'] ** (n_cross if n_glyco else max(n_cross - 1, 0))
    return score


def compute_fragment_lability(edge_lability, subg):
    """Scores how labile the broken glycosidic bonds are (edge_lability: bond lability by edge of the parent glycan); higher means more expected cleavage"""
    broken = [lability for (u, v), lability in edge_lability.items() if (u in subg) != (v in subg)]
    return sum(broken) / max(len(broken), 1)


def merge_gp_global_mods(gp_names):
    """Reports a fragment-wide global modification only once across the peptide and glycan label lists"""
    seen, out = set(), []
    for sub in gp_names:
        new = []
        for y in sub:
            if y.startswith('M_'):
                if y in seen:
                    continue
                seen.add(y)
            new.append(y)
        out.append(new if new else ['M'])
    return out


def count_gp_cleavages(gp_names):
    """Counts the bond cleavages a glycopeptide fragment label actually implies"""
    n, global_mods = 0, set()
    for sub in gp_names:
        for y in sub:
            if y in ('Peptide', 'No Peptide', 'M') or y.startswith('loss of glycan'):
                continue
            if y.startswith('M_'):
                global_mods.add(y)
                continue
            n += 1
    return n + len(global_mods)


def score_gp_prior(gp_names, charge):
    """Prior score for the glycan cleavages and global modifications of a glycopeptide fragment"""
    cuts, pep_scores = [], []
    for sub in gp_names:
        for y in sub:
            if y in ('Peptide', 'No Peptide', 'M') or y.startswith('loss of glycan'):
                continue
            if y[0] in N_TERM_IONS | C_TERM_IONS and '_' in y and y.split('_')[-1].isdigit():
                pep_scores.append(PEPTIDE_ION_PRIORS[y[0]])
            else:
                cuts.append(y)
    if not cuts and not pep_scores:
        return 1.0
    return min((score_fragment_prior(cuts, charge) + sum(pep_scores)) / (len(cuts) + len(pep_scores)), 1.0)


def glycopeptide_frag_to_string(pep_gr, subg):
    """IUPAC-condensed representation of a glycopeptide fragment, glycans delimited by asterisks"""
    peptide_nodes = set(pep_gr.nodes)
    labels = nx.get_node_attributes(subg, 'string_labels')
    pep_part = ''.join(v for k, v in sorted(((k, v) for k, v in labels.items() if k in peptide_nodes),
                                            key = lambda x: int(x[0].split('-')[1])))
    glycan_strings = []
    for prefix in sorted({x.split('-')[0] for x in subg.nodes if x not in peptide_nodes}, key = int):
        glyc = subg.subgraph([x for x in subg.nodes if x.split('-')[0] == prefix])
        glycan_strings.append(mono_frag_to_string(nx.relabel_nodes(glyc, {x: int(x.split('-')[1]) for x in glyc.nodes},
                                                                   copy = True)))
    if not pep_part:
        return '*'.join(glycan_strings)
    if not glycan_strings:
        return pep_part
    return pep_part + '*' + '*'.join(glycan_strings)


def priority_filter(dc_names, diffs, peptide = False, charge = -1, lability = None):
    """Filters Domon-Costello fragment names by number of cleavages, fragmentation prior, and difference from observed mass\n
    | Arguments:
    | :-
    | dc_names (list): a nested list of Domon-Costello fragment grouped by mass
	| diffs (list): a nested list of mass differences between the masses of Domon-Costello fragments and the observed masses
	| peptide (bool): whether the input is a glycopeptide; default:False
	| charge (int): charge state of the precursor ion; default:-1\n
    | Returns:
    | :-
    | Returns a list of Domon-Costello fragment names sorted by number of cleavages, prior score, and the observed mass difference
    """
    lability = [DEFAULT_LABILITY] * len(dc_names) if lability is None else list(lability)
    if peptide:
        sorted_frags = sorted(list(zip(dc_names, diffs, lability)),
                              key = lambda x: (count_gp_cleavages(x[0]), -score_gp_prior(x[0], charge), x[1]))
    else:
        sorted_frags = sorted(list(zip(dc_names, diffs, lability)),
                              key = lambda x: (len(x[0]), -score_fragment_prior(x[0], charge), x[1]))
    return [f[0] for f in sorted_frags], [f[1] for f in sorted_frags], [f[2] for f in sorted_frags]


def max_fragment_charge(graph, peptide = False):
    """Caps a fragment's charge at the number of sites that can plausibly carry one: the N-terminus and basic residues of its peptide
    part, plus, with glycan residues on it, one more on a backbone fragment (EThcD c3 2+ of ETQ with sialyl-T) or one per four glycan
    residues on the intact peptide (EEQYNSTYR, one basic residue, is seen at 3+ with G0F and at 4+/5+ with sialylated glycans)"""
    if not peptide:
        return graph.number_of_nodes()
    labels = nx.get_node_attributes(graph, 'string_labels')
    pep_labels = [v for k, v in labels.items() if k.split('-')[0] == '0']
    if not pep_labels:
        return graph.number_of_nodes()
    n_glycan = len(labels) - len(pep_labels)
    backbone = any(isinstance(x, str) and x.startswith('peptide_') for d in nx.get_node_attributes(graph, 'atomic_mod_dict').values()
                   for x in d.values())
    return 1 + sum(1 for v in pep_labels if v in BASIC_RESIDUES) + (min(n_glycan, 1) if backbone else math.ceil(n_glycan / 4))


def match_fragment_properties(subg_frags, mass, mass_threshold, charge, sorted_frag_keys = None, peptide = False,
                              mass_threshold_ppm = None):
    """Searches subg_frags for any fragments which could correspond to the observed mass and its charge\n
    | Arguments:
    | :-
    | subg_frags (dict): lists of networkx subgraphs indexed by their mass
    | mass (float): the observed mass to match potential fragments against
    | mass_threshold (float): the range around the observed mass in which to match potential fragments
    | charge (int): the maximum possible charge on the fragments to be matched\n
    | Returns:
    | :-
    | (1) a list of only the observed mass with length equal to the number of matched outputs
    | (2) a list of the theoretical masses of the fragments matched with the observed mass
    | (3) a list of each of the differences from the matched fragments and the observed mass
    | (4) a list of the charge of each matched fragment
    | (5) a list of networkx objects of each matched fragment
    """
    fragment_properties = []
    modifier = np.sign(charge)
    if sorted_frag_keys is None:
        sorted_frag_keys = sorted(subg_frags.keys())
    for z in range(1, abs(charge) + 1):
        charged_mass = (mass * z) - (z - 1) * PROTON_MASS * modifier
        lo = bisect.bisect_left(sorted_frag_keys, charged_mass - mass_threshold)
        hi = bisect.bisect_right(sorted_frag_keys, charged_mass + mass_threshold)
        for frag_mass in sorted_frag_keys[lo:hi]:
            if mass_threshold_ppm is not None and abs(charged_mass - frag_mass) > frag_mass * mass_threshold_ppm / 1e6:
                continue
            for graph in subg_frags[frag_mass]:
                if z > 1 and z > max_fragment_charge(graph, peptide):
                    continue
                fragment_properties.append((mass, frag_mass, abs(charged_mass - frag_mass), modifier * z, graph))
    if fragment_properties:
        return list(zip(*fragment_properties))
    else:
        return [[], [], [], [], []]


def simplify_fragments(dc_names, peptide = False, diffs = None, intensities = None, charge = -1, prior_weight = 1.0,
                       lability_scores = None, mass_threshold = 0.5):
    """Sorts a list of possible fragments for each observed mass into a list of one fragment per observed mass\n
	| Arguments:
	| :-
	| dc_names (list): a list of Domon-Costello fragment names grouped by mass
	| peptide (bool): whether the input is a glycopeptide; default:False
	| diffs (list): a nested list of mass differences; default:None
	| intensities (list): not yet used; default:None
	| charge (int): charge state of the precursor ion; default:-1
	| prior_weight (float): scaling factor for fragmentation prior contribution to scoring; default:1.0\n
	| Returns:
	| :-
	| Returns a nested list with each list containing a single fragment or being empty
    """
    observed_frags = []
    if peptide:
        diff_weight = 10.0 / mass_threshold if mass_threshold > 0 else 25.0
        for i, possible_frags in enumerate(dc_names):
            if not possible_frags or len(possible_frags[0]) == 0:
                observed_frags.append([])
                continue
            frag_diffs = diffs[i] if diffs and diffs[i] else [0.0] * len(possible_frags)
            paired = sorted(zip(possible_frags, frag_diffs),
                            key = lambda x: (count_gp_cleavages(x[0]) + diff_weight * x[1] -
                                             0.5 * prior_weight * score_gp_prior(x[0], charge), x[1]))
            observed_frags.append([paired[0][0]])
        return observed_frags
    observed_frags = [[] for _ in dc_names]
    # how many of the fragments chosen so far contain each cleavage, so an option's overlap with them is one lookup per cleavage
    seen_cuts = Counter()
    order = sorted(range(len(dc_names)), key = lambda j: -intensities[j]) if intensities else list(range(len(dc_names)))
    for i in order:
        possible_frags = sorted(dc_names[i], key = len)
        if not possible_frags or len(possible_frags[0]) == 0:
            continue
        elif len(possible_frags[0]) == 1:
            observed_frags[i] = [possible_frags[0]]
        else:
            frag_options = [x for x in possible_frags if len(x) == len(possible_frags[0])]
            # a shared global modification is not evidence of a shared cleavage
            max_overlaps_seen = [sum(seen_cuts[c] for c in set(f)) - any(c[0] == 'M' for c in f) for f in frag_options]
            prior_scores = [score_fragment_prior(f, charge) for f in frag_options]
            min_cleavages = len(possible_frags[0])
            if lability_scores and lability_scores[i]:
                option_lability = [l for f, l in zip(possible_frags, lability_scores[i]) if len(f) == min_cleavages]
            else:
                option_lability = [DEFAULT_LABILITY] * len(frag_options)
            if diffs and diffs[i]:
                option_diffs = [d for f, d in zip(possible_frags, diffs[i]) if len(f) == min_cleavages]
                scores = [overlap - 0.5 * diff + prior_weight * (prior + lability)
                          for overlap, diff, prior, lability in
                          zip(max_overlaps_seen, option_diffs, prior_scores, option_lability)]
            else:
                scores = [overlap + prior_weight * (prior + lability)
                          for overlap, prior, lability in zip(max_overlaps_seen, prior_scores, option_lability)]
            max_overlap_idx = np.argsort(scores, kind = 'stable')[-1]
            observed_frags[i] = [frag_options[max_overlap_idx]]
        seen_cuts.update(set(observed_frags[i][0]))
    return observed_frags


def get_initial_global_mods(nx_mono, charge, disable_global_mods = False, max_global_mods = 1):
    """Creates a list of global modifications dependent on the original structure and ion mode"""
    if disable_global_mods:
        return [None], []
    global_mods = [x for x in mono_attributes['Global']['mass'] if x not in ['CO2', 'SO4', 'PO4']]
    charge_mods = {-1: ['+Na', '+K', '+Acetonitrile'], 1: ['+Acetate', '+Acetonitrile']}
    global_mods = [mod for mod in global_mods if mod not in charge_mods[np.sign(charge)]]
    node_labels = ''.join(v for v in nx.get_node_attributes(nx_mono, 'string_labels').values() if len(v) > 1)
    special_mod_residues = ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn', 'S', 'P']
    present_special_residues = [x for x in special_mod_residues if x in node_labels]
    combos = sorted(global_mods)
    if max_global_mods > 1:
        repeatable = [x for x in combos if x in REPEATABLE_GLOBAL_MODS]
        combos += sorted({combine_global_mods(c) for n in range(2, max_global_mods + 1)
                          for c in combinations_with_replacement(repeatable, n)})
    return [None] + combos, present_special_residues


def infer_glycosites(peptide_seq, glycan_class = None):
    """Infers potential N- and O-glycosylation sites from a peptide sequence\n
    | Arguments:
    | :-
    | peptide_seq (string): amino acid sequence
    | glycan_class (string): 'N', 'O', or None (both); default:None\n
    | Returns:
    | :-
    | Returns a list of 0-indexed positions
    """
    sites = []
    for i, aa in enumerate(peptide_seq):
        if glycan_class != 'O' and aa == 'N' and i + 2 < len(peptide_seq) and peptide_seq[i + 1] != 'P' and peptide_seq[
            i + 2] in 'ST':
            sites.append(i)
        elif glycan_class != 'N' and aa in 'ST':
            sites.append(i)
    return sites


def build_glycopeptide_input(peptide, modification_str, structures = None):
    """Parses a Peptide Modification string and returns a CandyCrumbs-ready input dict\n
    | Arguments:
    | :-
    | peptide (string): amino acid sequence
    | modification_str (string): modification string from glycoproteomics search (e.g., 'T1(Hex(1)HexNAc(1));K14(Guanidinyl)')
    | structures (dict): optional map of composition string to IUPAC-condensed structure, e.g.
    |                    {'Hex(1)HexNAc(1)': 'Gal(b1-3)GalNAc'}, for structure-level annotation; default:None\n
    | Returns:
    | :-
    | Returns a dict with 'peptide' (modified sequence), 'glycans' (list of composition dicts), and 'glycosites' (list of 0-indexed positions)
    """
    peptide = list(peptide)
    glycans = []
    glycosites = []
    for mod in modification_str.split(';'):
        mod = mod.strip()
        aa = mod[0]
        rest = mod[1:]
        pos_match = re.match(r'\d+', rest)
        if not pos_match or not rest.endswith(')'):
            raise ValueError(f"Could not parse modification '{mod}'; expected, e.g., 'T1(Hex(1)HexNAc(1))'")
        pos_str = pos_match.group()
        pos = int(pos_str) - 1
        if not 0 <= pos < len(peptide):
            raise ValueError(f"Modification '{mod}' is at position {pos + 1}, outside peptide of length {len(peptide)}")
        if peptide[pos] != aa:
            raise ValueError(
                f"Modification '{mod}' expects {aa} at position {pos + 1} but the peptide has {peptide[pos]}")
        content = rest[len(pos_str):][1:-1]
        if content in MODIFICATION_TOKENS and aa in MODIFICATION_TOKENS[content]:
            peptide[pos] = MODIFICATION_TOKENS[content][aa]
            continue
        if structures and content in structures:
            glycans.append(structures[content])
        else:
            # without strict, canonicalize_composition takes any word as a residue ('Oxidation' became the glycan {'Oxidation': 1}, which has
            # no mass and moved every site after it), and a phosphate alone is no glycan, so a glycan needs a monosaccharide too
            try:
                comp = canonicalize_composition(content, strict = True)
            except ValueError:
                comp = {}
            if not any(k in derivatization_sites['permethylated'] for k in comp):
                raise ValueError(
                    f"Modification '{mod}' is neither a glycan composition nor one of {sorted(MODIFICATION_TOKENS)} on its residue")
            glycans.append(comp)
        glycosites.append(pos)
    return {'peptide': ''.join(peptide), 'glycans': glycans, 'glycosites': glycosites}


def create_peptide_graph(pep_seq):
    """Creates a network object of a peptide"""
    pep_arr = np.roll(np.eye(len(pep_seq)), (2, 1), axis = (1, 0))
    pep_arr[:, 0] = 0
    pep_gr = nx.from_numpy_array(pep_arr, create_using = nx.DiGraph)
    nx.set_node_attributes(pep_gr, dict(enumerate(pep_seq)), 'string_labels')
    bond_dict = {(e[0], e[1]): {'bond_label': 'peptide'} for e in pep_gr.edges}
    nx.set_edge_attributes(pep_gr, bond_dict)
    return pep_gr


def create_glycopeptide_graph(peptide, glycans, glycosites):
    """Creates and merges network objects of a peptide and glycans"""
    pep_gr = create_peptide_graph(peptide)
    glycan_graphs = [mono_graph_to_nx(glycan_to_graph_monos(glyc), directed = True) for glyc in glycans]
    red_nodes = [[x for x in nx.get_node_attributes(glyc, 'reducing_end')][0] for glyc in glycan_graphs]
    prefixes = [f'{l}-' for l, g in enumerate([pep_gr] + glycan_graphs)]
    glyco_pep = nx.union_all([pep_gr] + glycan_graphs, rename = prefixes)
    for pref, gsite, red_node in zip(prefixes[1:], glycosites, red_nodes):
        glyco_pep.add_edge(f'0-{gsite}', f'{pref}{red_node}')
        glyco_pep[f'0-{gsite}'][f'{pref}{red_node}']['bond_label'] = 'glycosite'
    pep_gr = copy.deepcopy(glyco_pep.subgraph([x for x in glyco_pep.nodes() if x.startswith('0-')]))
    return glyco_pep, pep_gr


def input_to_graph(input_dict):
    if not input_dict['peptide'] and not input_dict['glycosites']:
        if isinstance(input_dict['glycans'], str):
            nx_mono = mono_graph_to_nx(glycan_to_graph_monos(input_dict['glycans']))
            return nx_mono, None
        if len(input_dict['glycans']) == 1:
            nx_mono = mono_graph_to_nx(glycan_to_graph_monos(input_dict['glycans'][0]))
            return nx_mono, None
    if input_dict['peptide']:
        nx_mono, pep_gr = create_glycopeptide_graph(input_dict['peptide'], input_dict['glycans'],
                                                    input_dict['glycosites'])
        return nx_mono, pep_gr


def get_glycan_cleavages(gp, subg, glycosites, chain_ranks):
    """Return Domon-Costello labels for all glycans on a glycopeptide (chain_ranks: rank_chains of each glycan, keyed by its graph prefix)"""
    subg_glycan_prefixes = {x.split('-')[0] for x in subg.nodes()} - {'0'}
    subg_atom_dict = nx.get_node_attributes(subg, 'atomic_mod_dict')
    all_mods = []
    for prefix, glycosite in enumerate(glycosites, 1):
        if str(prefix) in subg_glycan_prefixes:
            glyc = [x for x in gp.nodes() if x.split('-')[0] == str(prefix)]
            glyc_dc = subgraphs_to_domon_costello(gp.subgraph(glyc), [subg.subgraph(glyc)], chain_ranks[str(prefix)])
            all_mods.extend([x if x else ['M'] for x in glyc_dc])
        elif f'0-{glycosite}' in subg_atom_dict and subg_atom_dict[f'0-{glycosite}'][3]:
            all_mods.append([f'{cut_type_dict[subg_atom_dict[f"0-{glycosite}"][3]]}_0_Alpha'])
        else:
            all_mods.append([f'loss of glycan {prefix}'])
    return all_mods


def peptide_to_RF_nomenclature(peptide, pep_subg, iupac = False, allowed_ion_types = None, allow_internal = False):
    """Return Roepstorff and Fohlman peptide fragment nomenclature, or None if the fragment is rejected"""
    RF_cleavages = []
    all_peptide_nodes = sorted(peptide.nodes, key = lambda x: int(x.split('-')[1]))
    peptide_node_set = set(all_peptide_nodes)
    if not any(x in peptide_node_set for x in pep_subg.nodes):
        return ['No Peptide']
    pep_atom_dict = {k: v for k, v in nx.get_node_attributes(pep_subg, 'atomic_mod_dict').items() if
                     k in peptide_node_set}
    global_mods = nx.get_node_attributes(pep_subg, 'global_mod')
    n_term_cuts, c_term_cuts = 0, 0
    for node, cleavages in pep_atom_dict.items():
        for cleavage in [x for x in cleavages.values() if isinstance(x, str) and x.startswith('peptide_')]:
            ion_type = cleavage[-1]
            if allowed_ion_types is not None and ion_type not in allowed_ion_types:
                return None
            if ion_type in C_TERM_IONS:
                c_term_cuts += 1
                cut_site = all_peptide_nodes[::-1].index(node) + 1
            elif ion_type in N_TERM_IONS:
                n_term_cuts += 1
                cut_site = all_peptide_nodes.index(node) + 1
                if cut_site == 1 and ion_type in ('a', 'b'):
                    return None
            else:
                raise ValueError(f"Unrecognized peptide cleavage type: {cleavage}")
            RF_cleavages.append(f'{ion_type}_{cut_site}')
    if n_term_cuts and c_term_cuts and not allow_internal:
        return None
    if not RF_cleavages:
        RF_cleavages = ['Peptide']
    if global_mods:
        RF_cleavages.append(f"M_{list(global_mods.values())[0][0]}")
    if iupac:
        subg_peptide_labels = {k: v for k, v in nx.get_node_attributes(pep_subg, 'string_labels').items() if
                               k in peptide_node_set}
        return sorted(RF_cleavages), ''.join(
            v for _, v in sorted(subg_peptide_labels.items(), key = lambda x: int(x[0].split('-')[1])))
    return sorted(RF_cleavages)


def nested_lazy_product_vect(perm_lists, atom_dict_lists, global_mods, indices):
    """Calculates the permutation of glycan modifications based on indices of matching masses"""
    inner_indices, global_mod_index = divmod(indices, len(global_mods))
    global_mods = np.array(global_mods)[global_mod_index]
    array_indices = vectorized_lazy_product_indices(atom_dict_lists, inner_indices)
    perms = select_indices(perm_lists, indices, array_indices)
    atom_dicts = select_indices(atom_dict_lists, indices, array_indices)
    return zip(perms, atom_dicts, global_mods)


def vectorized_lazy_product_indices(arrays, indices):
    """Calculates indices of elements in arrays given indices of array permutations"""
    reversed_arrays = arrays[::-1]
    array_sizes = np.array([len(arr) for arr in reversed_arrays])
    weights = np.cumprod([1] + list(array_sizes[:-1]))
    indices_expanded = indices[:, np.newaxis]
    weights_expanded = weights[np.newaxis, :]
    array_sizes_expanded = array_sizes[np.newaxis, :]
    array_indices = (indices_expanded // weights_expanded) % array_sizes_expanded
    return array_indices


def select_indices(arrays, indices, array_indices):
    """Converts arrays indices back to the original elements"""
    reversed_arrays = arrays[::-1]
    result = np.empty((len(indices), len(arrays)), dtype = object)
    for i, arr in enumerate(reversed_arrays):
        result[:, -(i + 1)] = np.array(arr, dtype = object)[array_indices[:, i]]
    return result


def check_masses(desired_masses, new_masses, threshold):
    """Calculates indices of masses which are within a threshold of any mass in another array"""
    d = np.sort(np.asarray(desired_masses))
    f = np.asarray(new_masses)
    idx = np.searchsorted(d, f)
    left = np.abs(f - d[np.clip(idx - 1, 0, len(d) - 1)])
    right = np.abs(f - d[np.clip(idx, 0, len(d) - 1)])
    return np.minimum(left, right) < threshold


def glycopeptide_string_to_input(gpep_string):
    input_dict = {k: [] for k in ['glycans', 'peptide', 'glycosites']}
    split_string = gpep_string.split("*")
    if len(split_string) == 1:
        chem_string = split_string[0]
        # testing for '(' or a trailing 'c' mistook bare monosaccharides ('Man') for peptides and
        # Carbamidomethyl-Cys-terminated peptides ('AVAVTLQSHc') for glycans
        if set(chem_string) <= set(AA_masses):
            input_dict['peptide'] = chem_string
        else:
            input_dict['glycans'] = [chem_string]
        return input_dict
    else:
        input_dict['peptide'] = ''.join(split_string[::2])
        input_dict['glycans'] = split_string[1::2]
        input_dict['glycosites'] = np.cumsum([len(x) for x in split_string[::2][:-1]]) - 1
        return input_dict


def get_derivatization_count(mono_type, fragment_type, sample_prep = 'permethylated'):
    """Returns the number of derivatized groups (methyls when permethylated, acetyls when peracetylated) a fragment keeps"""
    return mono_attributes.get(mono_type, {}).get(sample_prep, {}).get(fragment_type, 0)


def _build_composition_fragments(composition, re_bonus, sample_prep = 'underivatized',
                                 disable_X_cross_rings = False):
    """Builds glycan fragment options from a monosaccharide composition\n
    | Arguments:
    | :-
    | composition (dict): monosaccharide composition, e.g., {'Hex': 5, 'HexNAc': 4, 'S': 1}
    | re_bonus (float): mass bonus for fragments retaining the reducing end (Y/Z/M)
    | sample_prep (string): underivatized/permethylated/peracetylated
    | disable_X_cross_rings (bool): whether to disable X-type cross-ring cleavages\n
    | Returns:
    | :-
    | Returns a list of (mass_without_mode, label_string, n_cleavages) tuples
    """
    deriv_mass = DERIVATIZATION_MASSES.get(sample_prep, 0)
    mono_types = sorted(m for m in composition if m in derivatization_sites['permethylated'] and composition[m] > 0)
    if not mono_types:
        return []
    # a substituent (S, P, Ac, ...) of a composition can sit on any residue, so a fragment can carry any number of them
    parts = mono_types + sorted(m for m in composition if m in SUBSTITUENTS and composition[m] > 0)
    part_masses = {m: mono_attributes[m]['mass'][m] + get_derivatization_count(m, m, sample_prep) * deriv_mass if
                   m in mono_types else SUBSTITUENTS[m]['mass'] + SUBSTITUENTS[m].get(sample_prep, 0) * deriv_mass for m in parts}
    frags = []
    ion_adj = {'Y': (-deriv_mass, re_bonus), 'Z': (-(WATER_MASS + deriv_mass), re_bonus), 'B': (0, 0), 'C': (WATER_MASS, 0)}
    for combo in product(*(range(composition[m] + 1) for m in parts)):
        sub_comp = {m: c for m, c in zip(parts, combo) if c > 0}
        n_monos = sum(c for m, c in sub_comp.items() if m in mono_types)
        if not n_monos:
            continue
        # every glycosidic bond within the fragment takes one derivatizable group
        residue_sum = sum(part_masses[m] * c for m, c in sub_comp.items()) - (n_monos - 1) * deriv_mass
        is_full = all(sub_comp.get(m, 0) == composition[m] for m in parts)
        comp_str = '/'.join(f"{m}({c})" for m, c in sorted(sub_comp.items()))
        if is_full:
            frags.append((residue_sum + re_bonus, f'M {comp_str}', 0))
        else:
            for ion_type, (bond_adj, red_bon) in ion_adj.items():
                frags.append((residue_sum + bond_adj + red_bon, f'{ion_type} {comp_str}', 1))
    allowed_X = X_cross_rings if not disable_X_cross_rings else set()
    for mono in mono_types:
        for frag_type, frag_mass in mono_attributes[mono]['mass'].items():
            if frag_type == mono:
                continue
            if frag_type in A_cross_rings or frag_type in allowed_X:
                frags.append((frag_mass + get_derivatization_count(mono, frag_type, sample_prep) * deriv_mass,
                              f'{frag_type} {mono}', 1))
    return frags


def composition_to_fragments(composition, fragment_masses, mass_threshold, max_cleavages = 3,
                             charge = -1, mass_tag = None, simplify = True, disable_global_mods = False,
                             disable_X_cross_rings = None, sample_prep = 'underivatized',
                             peptide_seq = None, glycosites = None, glycan_class = None,
                             fragmentation_method = None, max_global_mods = 1, mass_threshold_ppm = None):
    """Calculates all possible fragment masses from a monosaccharide composition, optionally on a peptide\n
    | Arguments:
    | :-
    | composition (dict or list): monosaccharide composition(s); a single dict or list of dicts for multi-glycan glycopeptides
    | fragment_masses (list): observed masses to annotate
    | mass_threshold (float): maximum tolerated mass difference for fragment matching
    | max_cleavages (int): maximum number of allowed concurrent fragmentations per mass; default:3
    | charge (int): charge state of the precursor ion; default:-1
    | mass_tag (float): mass of the glycan label or reducing end modification; default:2*H (free glycan) or 0 (glycopeptide)
    | simplify (bool): whether to condense fragment options to the most likely; default:True
    | disable_global_mods (bool): whether to disable global modifications; default:False
    | disable_X_cross_rings (bool): whether to disable X-type cross-ring cleavages; default:False
    | sample_prep (string): underivatized/permethylated/peracetylated
    | peptide_seq (string): amino acid sequence; when provided, generates glycopeptide fragments; default:None
    | glycosites (list): 0-indexed peptide positions for each glycan; inferred from sequence if None; default:None\n
    | Returns:
    | :-
    | Returns a dict keyed by observed mass, each pointing to an annotation dict or None
    """
    if disable_X_cross_rings is None:
        disable_X_cross_rings = charge > 0
    is_glycopeptide = peptide_seq is not None
    if mass_tag is None:
        mass_tag = 0 if is_glycopeptide else 2 * HYDROGEN_MASS
    mode_mass = -PROTON_MASS if charge < 0 else PROTON_MASS
    modifier = np.sign(charge)
    frag_dict = {}
    if is_glycopeptide:
        compositions = [composition] if isinstance(composition, dict) else list(composition)
        if glycosites is None or not len(glycosites):
            glycosites = infer_glycosites(peptide_seq, glycan_class)[:len(compositions)]
        glycosites = list(glycosites)
        if len(glycosites) != len(compositions):
            raise ValueError(
                f"Could not assign {len(compositions)} glycan composition(s) to {len(glycosites)} glycosylation site(s) on {peptide_seq}; pass 'glycosites' explicitly")
        if any(not 0 <= gs < len(peptide_seq) for gs in glycosites):
            raise ValueError(f"Glycosite index out of range for peptide of length {len(peptide_seq)}: {glycosites}")
        n_glycans = len(compositions)
        # Glycan fragment options per glycan (re_bonus=0: reducing end bonded to peptide)
        all_glycan_frags = [_build_composition_fragments(comp, 0, sample_prep, disable_X_cross_rings) for comp in
                            compositions]
        glycan_totals = [next((m for m, lb, nc in gf if nc == 0), 0) for gf in all_glycan_frags]
        total_glycan = sum(glycan_totals)
        # Peptide masses
        pep_res = [AA_masses.get(aa, 0) for aa in peptide_seq]
        n_aa = len(pep_res)
        full_pep = sum(pep_res) + WATER_MASS
        cum_n = list(np.cumsum(pep_res))
        cum_c = list(np.cumsum(pep_res[::-1]))
        allowed_ion_types = PEPTIDE_ION_TYPES[fragmentation_method] if isinstance(fragmentation_method,
                                                                                  (str, type(None))) else set(
            fragmentation_method)
        pep_ions = [('b', lambda i: cum_n[i - 1], 'n'),
                    ('a', lambda i: cum_n[i - 1] - 27.994915, 'n'),
                    ('c', lambda i: cum_n[i - 1] + 17.026549, 'n'),
                    ('y', lambda i: cum_c[i - 1] + WATER_MASS, 'c'),
                    ('z', lambda i: cum_c[i - 1] + WATER_MASS - 17.026549 + HYDROGEN_MASS, 'c'),
                    ('w', lambda i: cum_c[i - 1] + WATER_MASS - 17.026549 + HYDROGEN_MASS - W_SIDE_CHAIN_LOSSES[
                        peptide_seq[n_aa - i]] if peptide_seq[n_aa - i] in W_SIDE_CHAIN_LOSSES else None, 'c')]
        pep_ions = [x for x in pep_ions if x[0] in allowed_ion_types]
        # a1/b1 require N-terminal acylation and are not formed by ordinary peptides
        first_i = {'a': 2, 'b': 2}

        def add(mass, label, nc):
            frag_dict.setdefault(round(mass, 5), []).append((label, nc))

        # Full molecule
        add(full_pep + total_glycan + mode_mass, [['Peptide']] + [['M'] for _ in range(n_glycans)], 0)
        # Per-glycan states: intact, fully cleaved off, or sub-fragmented; a glycan still on the peptide keeps its reducing end, so
        # only Y/Z ions and X cross-rings of the reducing-end residue qualify (B/C/A ones put, e.g., '35A Hex' on a peptide at the mass of
        # y8 + HexNAc). On an N-site that residue is HexNAc, so the piece has to contain one ('02X Hex' and 'Y Hex(2)/dHex(1)' were
        # options on EEQYNSTYR), while O-sites also carry Man, Fuc, Glc, or Xyl
        glycan_options = []
        for g_idx in range(n_glycans):
            opts = [(['M'], glycan_totals[g_idx], 0), ([f'loss of glycan {g_idx + 1}'], 0.0, 1)]
            opts += [([gl], gm, gc) for gm, gl, gc in all_glycan_frags[g_idx] if gc > 0 and (
                    gl.split(' ')[0] in ('Y', 'Z') or gl.split(' ')[0] in X_cross_rings) and (
                    peptide_seq[glycosites[g_idx]] != 'N' or 'HexNAc' in [c.split('(')[0] for c in gl.split(' ')[1].split('/')])]
            glycan_options.append(opts)
        # Intact peptide with any combination of glycan states
        for combo in product(*glycan_options):
            nc = sum(c[2] for c in combo)
            if not 0 < nc <= max_cleavages:
                continue
            add(full_pep + sum(c[1] for c in combo) + mode_mass, [['Peptide']] + [list(c[0]) for c in combo], nc)
        # Peptide backbone fragments; glycans outside the fragment are lost without costing a cleavage
        for ion_name, mass_fn, terminus in pep_ions:
            for i in range(first_i.get(ion_name, 1), n_aa):
                pmass = mass_fn(i)
                if pmass is None:
                    continue
                plabel = f'{ion_name}_{i}'
                opts = [glycan_options[j] if (gs < i if terminus == 'n' else gs >= n_aa - i)
                        else [([f'loss of glycan {j + 1}'], 0.0, 0)] for j, gs in enumerate(glycosites)]
                for combo in product(*opts):
                    nc = 1 + sum(c[2] for c in combo)
                    if nc > max_cleavages:
                        continue
                    add(pmass + sum(c[1] for c in combo) + mode_mass, [[plabel]] + [list(c[0]) for c in combo], nc)
        # Glycan-only (no peptide): oxonium / B-type ions
        for g_idx in range(n_glycans):
            # Full glycan B-ion (single cleavage: glycan detaches from peptide)
            comp = compositions[g_idx]
            comp_str = '/'.join(f"{m}({c})" for m, c in sorted(comp.items()))
            glyc = [[f'loss of glycan {j + 1}'] for j in range(n_glycans)]
            glyc[g_idx] = [f'B {comp_str}']
            add(glycan_totals[g_idx] + mode_mass, [['No Peptide']] + glyc, 1)
            # Sub-composition glycan-only fragments, which lack the reducing end and so are B/C ions or A cross-rings
            for gm, gl, gc in all_glycan_frags[g_idx]:
                if gc == 0 or not (gl.split(' ')[0] in ('B', 'C') or gl.split(' ')[0] in A_cross_rings):
                    continue
                glyc = [[f'loss of glycan {j + 1}'] for j in range(n_glycans)]
                glyc[g_idx] = [gl]
                add(gm + mode_mass, [['No Peptide']] + glyc, gc)
    else:
        # Pure composition (existing behavior)
        # a derivatized reducing end carries one more group, and an alditol (mass_tag of 2 H) one more again
        re_bonus = WATER_MASS + mass_tag + DERIVATIZATION_MASSES.get(sample_prep, 0) * (
                1 + (abs(mass_tag - 2 * HYDROGEN_MASS) < 0.01))
        glycan_frags = _build_composition_fragments(composition, re_bonus, sample_prep, disable_X_cross_rings)
        if not glycan_frags:
            return {m: None for m in fragment_masses}
        for gm, gl, gc in glycan_frags:
            frag_dict.setdefault(round(gm + mode_mass, 5), []).append(([gl], gc))
    # Global modifications (shared)
    if not disable_global_mods:
        adduct_mods = {'+Na', '+K', '+Acetate', '+Acetonitrile'}
        charge_exclude = {-1: ['+Na', '+K', '+Acetonitrile'], 1: ['+Acetate', '+Acetonitrile']}
        excluded = set(charge_exclude.get(np.sign(charge), []))
        if not is_glycopeptide:
            excluded.add('NH3')
        global_mods_dict = {k: v for k, v in mono_attributes['Global']['mass'].items()
                            if k not in ('CO2', 'SO4', 'PO4') and k not in excluded}
        if max_global_mods > 1:
            repeatable = [x for x in global_mods_dict if x in REPEATABLE_GLOBAL_MODS]
            for n in range(2, max_global_mods + 1):
                for combo in combinations_with_replacement(repeatable, n):
                    global_mods_dict[combine_global_mods(combo)] = sum(global_mods_dict[x] for x in combo)
        all_comps = compositions if is_glycopeptide else [composition]
        all_mono_labels = ''.join(m * c for comp in all_comps for m, c in comp.items())
        if any(x in all_mono_labels for x in ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn']):
            global_mods_dict['CO2'] = mono_attributes['Global']['mass']['CO2']
        if 'S' in all_mono_labels:
            global_mods_dict['SO4'] = mono_attributes['Global']['mass']['SO4']
        if any(m == 'P' or m.endswith('P') for comp in all_comps for m in comp):
            global_mods_dict['PO4'] = mono_attributes['Global']['mass']['PO4']
        # snapshot the label lists, else a variant landing on an existing mass is modified again by later global mods
        base_entries = [(base_mass, list(entries)) for base_mass, entries in frag_dict.items()]
        base_masses = np.array([base_mass for base_mass, _ in base_entries])
        window_masses = extend_masses(fragment_masses, charge)
        for gmod, gmod_mass in global_mods_dict.items():
            adj_mass = gmod_mass - mode_mass if gmod in adduct_mods else gmod_mass
            # only variants that can match an observed mass are materialized; the margin covers the 5-decimal rounding
            in_window = check_masses(window_masses, base_masses + adj_mass, mass_threshold + 1e-4) if len(window_masses) else []
            for j in np.flatnonzero(in_window):
                base_mass, entries = base_entries[j]
                new_mass = round(base_mass + adj_mass, 5)
                for label, nc in entries:
                    if nc + 1 <= max_cleavages:
                        if is_glycopeptide:
                            new_label = [sub[:] for sub in label]
                            new_label[0] = new_label[0] + [f'M_{gmod}']
                        else:
                            new_label = label + [f'M_{gmod}']
                        frag_dict.setdefault(new_mass, []).append((new_label, nc + 1))
    # Matching (shared)
    sorted_frag_keys = sorted(frag_dict.keys())
    hit_dict = {}
    for observed_mass in fragment_masses:
        matches = []
        for z in range(1, abs(charge) + 1):
            charged_mass = (observed_mass * z) - (z - 1) * PROTON_MASS * modifier
            lo = bisect.bisect_left(sorted_frag_keys, charged_mass - mass_threshold)
            hi = bisect.bisect_right(sorted_frag_keys, charged_mass + mass_threshold)
            for frag_mass in sorted_frag_keys[lo:hi]:
                if mass_threshold_ppm is not None and abs(
                        charged_mass - frag_mass) > frag_mass * mass_threshold_ppm / 1e6:
                    continue
                for label, nc in frag_dict[frag_mass]:
                    if nc <= max_cleavages:
                        matches.append((frag_mass, label, modifier * z, abs(charged_mass - frag_mass), nc))
        if matches:
            # equal-cleavage glycopeptide options are often isobaric (a5 plus a 1,5X HexNAc is b5 that lost its glycan), so, as in the
            # structure path, the fragmentation prior of the ion types ranks them before the mass error does; a composition fragment
            # ('Y HexNAc(1)', '02X HexNAc') is scored as its cleavage type, an X cross-ring on the peptide being on the reducing-end residue
            matches.sort(key = lambda x: (x[4], -score_gp_prior([[re.sub(r'^([BCYZ]|\d\d[AX]) .*', r'\1_1', y) for y in sub] for sub in
                                                                 x[1]], charge), x[3]) if is_glycopeptide else (x[4], x[3]))
            matches = matches[:1] if simplify else matches[:5]
            hit_dict[observed_mass] = {
                'Theoretical fragment masses': [m[0] for m in matches],
                'Domon-Costello nomenclatures': [m[1] for m in matches],
                'Fragment charges': [m[2] for m in matches],
            }
        else:
            hit_dict[observed_mass] = None
    return hit_dict


@rescue_glycans
def CandyCrumbs(input_string, fragment_masses, mass_threshold = None,
                max_cleavages = 3, simplify = True, charge = -1, mass_tag = None,
                iupac = False, intensities = None, disable_global_mods = False, disable_X_cross_rings = None,
                disable_A_cross_rings = None, sample_prep = 'underivatized', prior_weight = 1.0, glycan_class = None,
                fragmentation_method = None, allow_internal_peptide_fragments = False, mass_threshold_ppm = None,
                max_global_mods = None, ms3_precursor = None):
    """Basic wrapper for the annotation of observed masses with correct nomenclature given a glycan\n
    | Arguments:
    | :-
    | input_string (string): glycan in IUPAC-condensed format (or composition as dict/string)
    | fragment_masses (list): all masses which are to be annotated with a fragment name
    | mass_threshold (float): the maximum tolerated mass difference around each observed mass at which to include fragments; default:None (0.5 Da, or 10 ppm for glycopeptides)
    | max_cleavages (int): maximum number of allowed concurrent fragmentations per mass; default:3
    | simplify (bool): whether to try condensing fragment options to the most likely option; default:True
    | charge (int): the charge state of the precursor ion (singly-charged, doubly-charged); default:-1
    | mass_tag (float): mass of the glycan label or reducing end modification; default:2.0156
    | iupac (bool): whether to add the fragment sequence in IUPAC-condensed nomenclature to the annotations; default:False
    | disable_A_cross_rings (bool): whether to strip out any A-type cross-rings; default: False
    | sample_prep (string): underivatized/permethylated/peracetylated
    | prior_weight (float): weighting of prior-informed scoring in simplify=True
    | glycan_class (string): "N" or "O" if relevant (only used to assign candidate sites in glycopeptides, nowhere else)
    | fragmentation_method (string): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD' to restrict peptide backbone ion types; default:None (all)
    | allow_internal_peptide_fragments (bool): whether to allow peptide fragments cleaved at both termini; default:False
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; default:None
    | max_global_mods (int): how many global modifications may co-occur on one fragment; 2 captures the
    |                        sequential water losses of the oxonium series. Counts as one cleavage either way;
    |                        default:None (2 for glycopeptides, 1 for free glycans)
    | ms3_precursor (float): m/z of the MS2 fragment that was isolated for an MS3 spectrum whose peaks are fragment_masses; these are then only
    |                        annotated as fragments of what that MS2 fragment can be in this glycan (within max_cleavages), with up to
    |                        max_cleavages further cleavages, and all are None if it cannot be any fragment of it; glycan structures only; default:None\n
    | Returns:
    | :-
    | Returns a list of tuples containing the observed mass and all of the possible fragment names within the threshold
    """
    glycopeptide_input = (isinstance(input_string, dict) and bool(input_string.get('peptide'))) or (
            isinstance(input_string, str) and '*' in input_string)
    if max_global_mods is None:
        # Glycopeptide oxonium series routinely lose two waters; free glycans rarely need a second modification
        max_global_mods = 2 if glycopeptide_input else 1
    if mass_threshold_ppm is None and mass_threshold is None and glycopeptide_input:
        # Intact glycopeptides are only measurable on FT instruments, where a fixed Da window is far too
        # loose; an explicitly given mass_threshold is always respected instead
        mass_threshold_ppm = 10
    if mass_threshold is None:
        mass_threshold = 0.5
    if mass_threshold_ppm is not None:
        mass_threshold = min(mass_threshold, max(fragment_masses) * abs(charge) * mass_threshold_ppm / 1e6)
    if disable_A_cross_rings is None:
        disable_A_cross_rings = charge > 0
        # a warning shows once per session, while a print repeated for every glycan CandyCrunch scores in positive mode
        if disable_A_cross_rings:
            warnings.warn(
                "A-type cross-ring fragmentation auto-disabled for positive mode; reducing-end X-type cross-rings kept (Na+/CID diagnostic). Override A with disable_A_cross_rings=False")
    if disable_X_cross_rings is None:
        disable_X_cross_rings = False
    composition = None
    if isinstance(input_string, dict):
        if 'peptide' in input_string:
            glycan_comps = []
            all_comp = True
            for g in input_string.get('glycans', []):
                c = g if isinstance(g, dict) else (
                    canonicalize_composition(g) if isinstance(g, str) and is_composition(g) else None)
                if c:
                    glycan_comps.append(c)
                else:
                    all_comp = False
                    break
            if all_comp and glycan_comps:
                return composition_to_fragments(
                    glycan_comps if len(glycan_comps) > 1 else glycan_comps[0],
                    sorted(fragment_masses), mass_threshold, max_cleavages = max_cleavages,
                    charge = charge, mass_tag = mass_tag, simplify = simplify,
                    disable_global_mods = disable_global_mods,
                    disable_X_cross_rings = disable_X_cross_rings,
                    sample_prep = sample_prep, fragmentation_method = fragmentation_method,
                    max_global_mods = max_global_mods, mass_threshold_ppm = mass_threshold_ppm,
                    peptide_seq = input_string['peptide'],
                    glycosites = list(input_string['glycosites']) if input_string.get('glycosites') is not None and len(
                        input_string['glycosites']) else None, glycan_class = glycan_class)
            else:
                glycans = input_string.get('glycans', [])
                if not glycans:
                    return {m: None for m in fragment_masses}
                peptide = input_string['peptide']
                gc = glycan_class if glycan_class else get_class(glycans[0])
                sites = list(input_string['glycosites']) if input_string.get('glycosites') is not None and len(
                    input_string['glycosites']) else infer_glycosites(peptide, gc)[:len(glycans)]
                if len(sites) != len(glycans):
                    raise ValueError(
                        f"Could not assign {len(glycans)} glycan(s) to {len(sites)} glycosylation site(s) on {peptide}; pass 'glycosites' explicitly")
                for site, glycan in sorted(zip(sites, glycans), reverse = True):
                    peptide = peptide[:site + 1] + '*' + glycan + '*' + peptide[site + 1:]
                input_string = peptide
        else:
            composition = input_string
    elif isinstance(input_string, str) and is_composition(input_string):
        composition = canonicalize_composition(input_string)
        if not composition:
            return {m: None for m in fragment_masses}
    if composition is not None:
        return composition_to_fragments(composition, sorted(fragment_masses), mass_threshold,
                                        max_cleavages = max_cleavages, charge = charge, mass_tag = mass_tag,
                                        simplify = simplify,
                                        disable_global_mods = disable_global_mods,
                                        disable_X_cross_rings = disable_X_cross_rings,
                                        sample_prep = sample_prep, glycan_class = glycan_class,
                                        max_global_mods = max_global_mods, mass_threshold_ppm = mass_threshold_ppm)
    hit_dict = {}
    input_dict = glycopeptide_string_to_input(input_string)
    if input_dict['peptide'] and input_dict['glycans']:
        glycan_comps, all_comp = [], True
        for g in input_dict['glycans']:
            if is_composition(g):
                c = canonicalize_composition(g)
                if c:
                    glycan_comps.append(c)
                else:
                    all_comp = False
                    break
            else:
                all_comp = False
                break
        if all_comp and glycan_comps:
            return composition_to_fragments(
                glycan_comps if len(glycan_comps) > 1 else glycan_comps[0],
                sorted(fragment_masses), mass_threshold, max_cleavages = max_cleavages,
                charge = charge, mass_tag = mass_tag, simplify = simplify,
                disable_global_mods = disable_global_mods,
                disable_X_cross_rings = disable_X_cross_rings,
                sample_prep = sample_prep, fragmentation_method = fragmentation_method,
                max_global_mods = max_global_mods, mass_threshold_ppm = mass_threshold_ppm,
                peptide_seq = input_dict['peptide'],
                glycosites = list(input_dict['glycosites']), glycan_class = glycan_class)
    if intensities is not None:
        fragment_masses, intensities = map(list, zip(*sorted(zip(fragment_masses, intensities))))
    else:
        fragment_masses = sorted(fragment_masses)
    nx_mono, pep_gr = input_to_graph(input_dict)
    node_labels = nx.get_node_attributes(nx_mono, 'string_labels')
    if any(map_to_basic(v, obfuscate_ptm = False) not in mono_attributes for v in node_labels.values() if len(v) > 1):
        return {m: None for m in fragment_masses}
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge,
                                                            disable_global_mods = disable_global_mods,
                                                            max_global_mods = max_global_mods)
    allowed_X_cleavages = [] if disable_X_cross_rings else X_cross_rings
    parents, subgraphs = None, None
    if ms3_precursor is not None and not input_dict['peptide']:
        # An MS3 peak is a fragment of the isolated MS2 fragment, so it can only come from inside one of that fragment's least-cleaved
        # annotations, of no higher charge, keeping its cross-ring cleavages
        prec_frags = generate_atomic_frags(nx_mono, global_mods, special_residues, allowed_X_cleavages, max_cleavages = max_cleavages,
                                           fragment_masses = [ms3_precursor], threshold = mass_threshold, mass_tag = mass_tag,
                                           charge = charge, sample_prep = sample_prep, disable_A_cross_rings = disable_A_cross_rings)
        parents = list(zip(*match_fragment_properties(prec_frags, ms3_precursor, mass_threshold, charge,
                                                      mass_threshold_ppm = mass_threshold_ppm)[1:]))
        if not parents:
            return {m: None for m in fragment_masses}
        cuts = [len(x) for x in subgraphs_to_domon_costello(nx_mono, [p[3] for p in parents])]
        parents = [p for p, c in zip(parents, cuts) if c == min(cuts)]
        subgraphs = [s for s in enumerate_subgraphs(nx_mono) + [set(nx_mono)] if any(s <= set(p[3]) for p in parents)]
        node_basic = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in node_labels.items()}
        max_cleavages = 2 * max_cleavages
    subg_frags = generate_atomic_frags(nx_mono, global_mods, special_residues, allowed_X_cleavages,
                                       max_cleavages = max_cleavages, fragment_masses = fragment_masses,
                                       subgraphs = subgraphs, threshold = mass_threshold, mass_tag = mass_tag,
                                       charge = charge, sample_prep = sample_prep,
                                       disable_A_cross_rings = disable_A_cross_rings)
    sorted_frag_keys = sorted(subg_frags.keys())
    if input_dict['peptide']:
        # each glycan's chains are ranked once, on the integer nodes of the free glycan: rank_chains orders equal-mass chains (the two arms
        # of a biantennary N-glycan) in set order, which for '1-3'-style string nodes swapped Alpha and Beta with PYTHONHASHSEED
        chain_rank = {}
        for prefix in sorted({x.split('-')[0] for x in nx_mono} - {'0'}, key = int):
            glyc = [x for x in nx_mono if x.split('-')[0] == prefix]
            chain_rank[prefix] = [(rank, [f'{prefix}-{n}' for n in chain]) for rank, chain in
                                  rank_chains(nx.relabel_nodes(nx_mono.subgraph(glyc), {x: int(x.split('-')[1]) for x in glyc}))]
    else:
        chain_rank = list(rank_chains(nx_mono))
    # the lability of every glycosidic bond is looked up once, not for every candidate fragment
    edge_lability = {(u, v): linkage_lability.get((map_to_basic(node_labels[u], obfuscate_ptm = False), d['bond_label'][1:] if
                                                   d['bond_label'][0].isalpha() else d['bond_label']), DEFAULT_LABILITY)
                     for u, v, d in nx_mono.edges(data = True) if d['bond_label'] not in ('glycosite', 'peptide')}
    downstream_values = []
    if input_dict['peptide']:
        peptide = True
        allowed_ion_types = PEPTIDE_ION_TYPES[fragmentation_method] if isinstance(fragmentation_method,
                                                                                  (str, type(None))) else set(
            fragmentation_method)
        for observed_mass in fragment_masses:
            all_gp_names, keep = [], []
            fragment_properties = match_fragment_properties(subg_frags, observed_mass, mass_threshold, charge,
                                                            sorted_frag_keys, peptide = True,
                                                            mass_threshold_ppm = mass_threshold_ppm)
            for idx, frag_subg in enumerate(fragment_properties[-1]):
                rf_names = peptide_to_RF_nomenclature(pep_gr, frag_subg, allowed_ion_types = allowed_ion_types,
                                                      allow_internal = allow_internal_peptide_fragments)
                if rf_names is None:
                    continue
                dc_names = get_glycan_cleavages(nx_mono, frag_subg, input_dict['glycosites'], chain_rank)
                all_gp_names.append(merge_gp_global_mods([rf_names] + dc_names))
                keep.append(idx)
            fragment_properties = [[v[i] for i in keep] for v in fragment_properties]
            lability = [compute_fragment_lability(edge_lability, sg) for sg in fragment_properties[-1]]
            downstream_values.append((*fragment_properties, all_gp_names, lability))
    else:
        peptide = False
        for observed_mass in fragment_masses:
            fragment_properties = match_fragment_properties(subg_frags, observed_mass, mass_threshold, charge,
                                                            sorted_frag_keys, mass_threshold_ppm = mass_threshold_ppm)
            if parents is not None:
                keep = [i for i, (mass, z, g) in enumerate(zip(fragment_properties[1], fragment_properties[3], fragment_properties[4])) if any(
                    mass < p_mass and abs(z) <= abs(p_z) and set(g) <= set(p_g) and all(
                        p_g.nodes[n].get('mod_labels') in (None, node_basic[n], g.nodes[n].get('mod_labels')) for n in g)
                    for p_mass, _, p_z, p_g in parents)]
                fragment_properties = [[v[i] for i in keep] for v in fragment_properties]
            dc_names = subgraphs_to_domon_costello(nx_mono, fragment_properties[-1], chain_rank)
            lability = [compute_fragment_lability(edge_lability, sg) for sg in fragment_properties[-1]]
            downstream_values.append((*fragment_properties, dc_names, lability))
    filtered_results = [priority_filter(x[5], x[2], peptide = peptide, charge = charge, lability = x[6]) if x[0]
                        else ([], [], []) for x in downstream_values]
    filtered_dc_names = [r[0] for r in filtered_results]
    if simplify:
        filtered_diffs = [r[1] for r in filtered_results]
        filtered_lability = [r[2] for r in filtered_results]
        filtered_dc_names = simplify_fragments(filtered_dc_names, peptide = peptide, diffs = filtered_diffs,
                                               intensities = intensities, charge = charge, prior_weight = prior_weight,
                                               lability_scores = filtered_lability, mass_threshold = mass_threshold)
    for i, frag_dc_names in enumerate(filtered_dc_names):
        if frag_dc_names:
            filtered_properties = list(zip(*downstream_values[i]))
            final_hits = [[y for y in filtered_properties if y[5] == x][0] for x in frag_dc_names[:5]]
            final_hits = [list(x) for x in list(zip(*final_hits))]
            hit_dict[fragment_masses[i]] = {'Theoretical fragment masses': final_hits[1],
                                            'Domon-Costello nomenclatures': final_hits[5],
                                            'Fragment charges': final_hits[3]}
            if iupac:
                hit_dict[fragment_masses[i]]['Fragment IUPAC'] = [
                    glycopeptide_frag_to_string(pep_gr, x) if peptide else mono_frag_to_string(x) for x in
                    final_hits[4]]
        else:
            hit_dict[fragment_masses[i]] = None
    return hit_dict


def rank_glycopeptide_structures(peptide, modification_str, fragment_masses, intensities = None, charge = 2, structures = None,
                                 fragmentation_method = None, mass_threshold = 0.5, mass_threshold_ppm = 10, kingdom = 'Animalia',
                                 top_n_peaks = 150, max_candidates = 100, **kwargs):
    """Ranks the candidate glycan structures of a glycoproteomics identification (peptide, modifications, glycan composition) by their fragment evidence in its MS2 spectrum\n
    | Arguments:
    | :-
    | peptide (string): amino acid sequence
    | modification_str (string): modifications as a search engine reports them, e.g., 'N5(HexNAc(4)Hex(5)Fuc(1));M2(Oxidation)' (see build_glycopeptide_input)
    | fragment_masses (list or dict): observed m/z values, or a peak dictionary of m/z : intensity such as the peak_d of load_spectra_filepath
    | intensities (list): intensities of the m/z values if fragment_masses is a list; default:None (all equal)
    | charge (int): precursor charge state; default:2
    | structures (list): candidate IUPAC-condensed structures, or one such list per glycan of a multiply glycosylated peptide; default:None (every glycowork database structure of each composition)
    | fragmentation_method (string): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD', e.g., the activation column of load_spectra_filepath; default:None (all peptide ion types)
    | mass_threshold (float): fragment mass tolerance in Da; default:0.5
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; None for low-resolution spectra; default:10
    | kingdom (string): taxonomic kingdom of the database structures; default:'Animalia'
    | top_n_peaks (int): how many of the most intense peaks are scored; default:150
    | max_candidates (int): most candidates (structure combinations, for several glycans) to score, at ~1 s each for N-glycans; default:100
    | **kwargs: passed on to CandyCrumbs, e.g., max_cleavages\n
    | Returns:
    | :-
    | Returns a dataframe with one row per candidate ('structures', one per glycan; candidates with residues CandyCrumbs has no tables for are left out),
    | best first, with 'score' (the product of the next two), 'rank' (candidates with identical fragment masses, such as linkage isomers, tie), 'explained'
    | (share of square-root intensity that CandyCrumbs annotates, weighted by the fragmentation prior of each annotation), 'direct_fragments' (share of the
    | candidate's single-cleavage fragments observed), and 'missing_fragments' (singly protonated m/z of those the spectrum lacks at every charge)
    """
    if not isinstance(fragment_masses, dict):
        fragment_masses = dict(zip(fragment_masses, [1.0] * len(fragment_masses) if intensities is None else intensities))
    if not fragment_masses:
        raise ValueError("No peaks to score")
    peaks = sorted(fragment_masses.items(), key = lambda x: -x[1])[:top_n_peaks]
    # a spectrum without intensities (all zero) weighs its peaks equally
    weights = {float(mz): math.sqrt(i / peaks[0][1]) if peaks[0][1] > 0 else 1.0 for mz, i in peaks}
    observed = np.array(sorted(weights))
    input_dict = build_glycopeptide_input(peptide, modification_str)
    if not input_dict['glycans']:
        raise ValueError(f"'{modification_str}' contains no glycan")
    if structures is None:
        structures = []
        for comp, site in zip(input_dict['glycans'], input_dict['glycosites']):
            candidates = compositions_to_structures(comp, glycan_class = 'N' if peptide[site] == 'N' else 'O', kingdom = kingdom)
            candidates = list(candidates['glycan']) if len(candidates) else []
            # a floating part would be placed at one arbitrary position, while its placed versions are candidates of their own
            structures.append([g for g in candidates if '{' not in g] or candidates)
    elif structures and isinstance(structures[0], str):
        structures = [structures]
    if len(structures) != len(input_dict['glycans']) or not all(structures):
        raise ValueError(f"Need candidate structures for each of the {len(input_dict['glycans'])} glycan(s) in '{modification_str}'")
    # every combination of the sites' candidates is scored, so two large sialylated N-glycans would take most of an hour
    if math.prod(len(x) for x in structures) > max_candidates:
        raise ValueError(f"{' x '.join(str(len(x)) for x in structures)} candidate combinations exceed max_candidates = {max_candidates}; pass "
                         f"fewer structures per glycan or raise max_candidates")
    pep_mass = sum(AA_masses[aa] for aa in input_dict['peptide']) + WATER_MASS
    rows = []
    for combo in product(*structures):
        graphs = [mono_graph_to_nx(glycan_to_graph_monos(g)) for g in combo]
        labels = [{n: map_to_basic(v, obfuscate_ptm = False) for n, v in nx.get_node_attributes(gr, 'string_labels').items()} for gr in graphs]
        if any(v not in mono_attributes for lab in labels for v in lab.values()):
            continue
        res_masses = [{n: mono_attributes[v]['mass'][v] for n, v in lab.items()} for lab in labels]
        total = sum(sum(m.values()) for m in res_masses)
        # A structure predicts the products of each single glycosidic cleavage: the oxonium ion of a branch of up to three residues, and the
        # peptide with the rest at any charge. Absent ones count against it, as a LacdiNAc or Lewis x candidate then lacks 407 or 512
        expected = []
        for gr, m in zip(graphs, res_masses):
            for child, _ in gr.edges():
                branch = nx.ancestors(gr, child) | {child}
                if len(branch) <= 3:
                    expected.append(np.array([sum(m[n] for n in branch) + PROTON_MASS]))
                expected.append((pep_mass + total - sum(m[n] for n in branch) + np.arange(1, abs(charge) + 1) * PROTON_MASS) /
                                np.arange(1, abs(charge) + 1))
        found = [check_masses(observed, mzs, np.minimum(mass_threshold, mzs * mass_threshold_ppm / 1e6) if mass_threshold_ppm else
                              mass_threshold).any() for mzs in expected]
        hits = CandyCrumbs({'peptide': input_dict['peptide'], 'glycans': list(combo), 'glycosites': input_dict['glycosites']},
                           list(weights), mass_threshold, charge = abs(charge), fragmentation_method = fragmentation_method,
                           mass_threshold_ppm = mass_threshold_ppm, **kwargs)
        annotations = {mz: hit['Domon-Costello nomenclatures'][0] for mz, hit in hits.items() if hit}
        # The Y ladder of a glycopeptide forms by consecutive glycosidic cleavages, so how many a Y ion needs is no evidence against a structure,
        # while an oxonium ion that needs more than its own cleavage is weaker evidence than a direct one
        explained = sum(weights[mz] * score_gp_prior(dc if 'No Peptide' in dc[0] else [
            [y for y in sub if not re.fullmatch(r'[YZ]_\d+_[A-Za-z]+', y)] or ['M'] for sub in dc], charge) for mz, dc in
                        annotations.items()) / sum(weights.values())
        direct = np.mean(found) if found else 1.0
        rows.append([list(combo), explained * direct, explained, direct, sorted({round(float(mzs[0]), 4) for mzs, f in zip(expected, found) if not f})])
    if not rows:
        raise ValueError("None of the candidate structures consists of residues CandyCrumbs can fragment")
    df_out = pd.DataFrame(rows, columns = ['structures', 'score', 'explained', 'direct_fragments', 'missing_fragments'])
    df_out = df_out.sort_values('score', ascending = False, kind = 'stable').reset_index(drop = True)
    df_out.insert(2, 'rank', df_out['score'].round(10).rank(ascending = False, method = 'min').astype(int))
    return df_out


def get_fragment_mass(glycan, fragment, charge = -1, mass_tag = None, sample_prep = 'underivatized', max_cleavages = 3):
    """Calculates the theoretical m/z of a named fragment; the inverse of CandyCrumbs\n
    | Arguments:
    | :-
    | glycan (string): glycan in IUPAC-condensed format
    | fragment (string/list): fragment in Domon-Costello nomenclature, either compact ("0,2A5a", "B1a/M-H2O") or as the list CandyCrumbs returns (['02A_5_Alpha'])
    | charge (int): charge state of the fragment ion, sign sets the ion mode; default:-1
    | mass_tag (float): mass of the glycan label or reducing end modification; default:2.0156
    | sample_prep (string): underivatized/permethylated/peracetylated
    | max_cleavages (int): maximum number of allowed concurrent fragmentations; default:3\n
    | Returns:
    | :-
    | Returns the m/z of the fragment, or None if it cannot exist on this glycan
    """
    if isinstance(fragment, str):
        cuts = []
        for part in re.split(r'[/;+]', fragment.replace(' ', '')):
            if part[0] in 'Mm':
                if mod := part[1:].lstrip('-_'):
                    cuts.append(f"M_{mod}")
                continue
            m = re.fullmatch(r'(\d),?(\d)?([ABCXYZabcxyz])_?(\d+)_?([A-Za-z]+)', part) if part[0].isdigit() else re.fullmatch(r'()()([ABCXYZabcxyz])_?(\d+)_?([A-Za-z]+)', part)
            if not m:
                raise ValueError(f"Could not parse fragment name '{part}'")
            ring_1, ring_2, cut_type, cut_num, chain = m.groups()
            chain = chain.capitalize()
            chain_idx = ord(chain.lower()) - 97 if len(chain) == 1 else -1
            if chain not in ranks and not 0 <= chain_idx < len(ranks):
                raise ValueError(f"Could not parse fragment name '{part}'")
            cuts.append(f"{ring_1}{ring_2 or ''}{cut_type.upper()}_{cut_num}_{chain if chain in ranks else ranks[chain_idx]}")
        fragment = cuts or ['M']
    fragment = sorted(fragment)
    nx_mono = mono_graph_to_nx(glycan_to_graph_monos(glycan), directed = True)
    chain_rank = list(rank_chains(nx_mono))
    try:
        skelly_dict, post_mono, global_mod = domon_costello_to_node_labels(fragment, dict(chain_rank))
    except (IndexError, KeyError):
        return None
    node_dict = nx.get_node_attributes(nx_mono, 'string_labels')
    bond_labels = nx.get_edge_attributes(nx_mono, 'bond_label')
    keep = set(nx_mono.nodes())
    for node, cut_type in skelly_dict.items():
        if cut_type[-1] == 'A':
            keep &= nx.ancestors(nx_mono, node) | {node}
            # an A-type cross-ring silently takes any branch attached to a ring atom it does not retain
            retained_atoms = mono_attributes[map_to_basic(node_dict[node], obfuscate_ptm = False)]['atoms'][cut_type]
            for child, _ in nx_mono.in_edges(node):
                if (pos := bond_labels[(child, node)][-1]).isdigit() and int(pos) not in retained_atoms:
                    keep -= nx.ancestors(nx_mono, child) | {child}
        elif cut_type[-1] in {'B', 'C'}:
            keep &= nx.ancestors(nx_mono, post_mono) | {post_mono}
        elif cut_type[-1] == 'X':
            keep -= nx.ancestors(nx_mono, node)
        else:
            keep -= nx.ancestors(nx_mono, node) | {node}
    if not keep:
        return None
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge, disable_global_mods = global_mod is None,
                                                            max_global_mods = 2)
    subg_frags = generate_atomic_frags(nx_mono, global_mods, special_residues, X_cross_rings,
                                       max_cleavages = max(max_cleavages, len(fragment)), fragment_masses = [],
                                       subgraphs = [nx_mono.subgraph(keep)], mass_tag = mass_tag, charge = charge,
                                       sample_prep = sample_prep)
    for mass in sorted(subg_frags):
        if any(sorted(x) == fragment for x in subgraphs_to_domon_costello(nx_mono, subg_frags[mass], chain_rank)):
            return (mass + (abs(charge) - 1) * PROTON_MASS * np.sign(charge)) / abs(charge)
    return None


def get_unique_subgraphs(nx_mono1, nx_mono2):
    """Gets the subgraphs unique to each of two input graphs\n
    | Arguments:
    | :-
    | nx_mono1 (networkx object): a monosaccharide only graph
    | nx_mono2 (networkx object): a different monosaccharide only graph\n
    | Returns:
    | :-
    | Returns two lists of networkx subgraphs of the inputs
    """
    nm = iso.categorical_node_match("string_labels",
                                    None)  # This is the criterion used to match nodes (it can also be something more general i.e Hex, HexNAc etc)
    all_unique_graphs1 = []
    all_unique_graphs2 = []
    # Only compare subgraphs of the same size
    for i in range(1, min(len(nx_mono1.nodes()), len(nx_mono2.nodes()))):
        graphs1 = set()
        graphs2 = set()
        first_graphs = enumerate_k_graphs(nx_mono1, i)
        second_graphs = enumerate_k_graphs(nx_mono2, i)
        undir1 = [g.to_undirected() for g in first_graphs]
        undir2 = [g.to_undirected() for g in second_graphs]
        for a, ua in enumerate(undir1):
            for b, ub in enumerate(undir2):
                if nx.is_isomorphic(ua, ub, node_match = nm):
                    graphs1.add(a)
                    graphs2.add(b)
        # Take only the subgraphs from each graph which are not isomorphic
        kunique_graphs1 = [first_graphs[x] for x in range(len(first_graphs)) if x not in graphs1]
        all_unique_graphs1.extend(kunique_graphs1)
        kunique_graphs2 = [second_graphs[x] for x in range(len(second_graphs)) if x not in graphs2]
        all_unique_graphs2.extend(kunique_graphs2)
    return all_unique_graphs1, all_unique_graphs2


def get_plots(df_sub, glycan_list, num_bins_plot, mz_range):
    """averages and plots spectra for two glycans\n
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
        out_dic[g] = [sum(col) / len(col) for col in
                      zip(*df_sub[df_sub.Prediction == g].binned_intensities.values.tolist())]
        if len(glycan_list) == 2 and glycan_list.index(g) == 1:
            plt.plot(mz_range, list(map(neg, out_dic[g]))[:num_bins_plot])
        else:
            plt.plot(mz_range, out_dic[g][:num_bins_plot])
    plt.xlabel("m/z")
    plt.ylabel("Relative intensity")
    plt.legend(glycan_list)
    return out_dic


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
    mz_range = [min_mz + ((max_mz - min_mz) / (bin_num - 1)) * k for k in range(num_bins_plot)]
    out_dic = {}
    if conf_analysis:
        conf_brackets = [(0.9, 1.0), (0.6, 0.9), (0.3, 0.6), (0, 0.3)]
        for bracket in conf_brackets:
            plt.clf()
            out_dic[bracket] = get_plots(df_sub[df_sub.Confidence.between(bracket[0], bracket[1])], glycan_list,
                                         num_bins_plot, mz_range)
            plt.title("Confidence range: " + str(bracket))
            plt.show()
    else:
        out_dic = get_plots(df_sub, glycan_list, num_bins_plot, mz_range)
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
    len_comp = ttest_ind([len(k) for k in df_a.Prediction],
                         [len(k) for k in df_b.Prediction], equal_var = False)
    print("p-value (Welch's t-test) for differences in glycan length: " + str(len_comp))
    branch_comp = ttest_ind([k.count('[') for k in df_a.Prediction],
                            [k.count('[') for k in df_b.Prediction], equal_var = False)
    print("p-value (Welch's t-test) for differences in glycan branching: " + str(branch_comp))


def get_sig_bins(df, glycan_list, conf_range = None, mz_cap = 3000, max_mz = 3000, min_mz = 39.714, bin_num = 2048,
                 motif = None, motif2 = None, controls = False, min_spectra = 10):
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
    | controls (bool): whether to check for systematic differences in tested glycans (length & branching)
    | min_spectra (int): minimum number of spectra that need to be present for glycans in glycan_list; default:10\n
    | Returns:
    | :-
    | Returns a list of tuples of the form (peak m/z, corrected p-value, effect size via Cohen's d)
    """
    max_bin = round((mz_cap - min_mz) / ((max_mz - min_mz) / (bin_num - 1)))
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
    if len(df_a) < min_spectra or len(df_b) < min_spectra:
        print("Not enough spectra for at least one of the two sequences")
        return []
    if controls:
        print("Number of spectra in df_a: " + str(len(df_a)))
        print("Number of spectra in df_b: " + str(len(df_b)))
        run_controls(df_a, df_b)
    df_r = pd.concat([df_a, df_b], axis = 0)
    remainder = [np.median([x for x in col if x] or [0]) for col in zip(*df_r.mz_remainder.values.tolist())]
    df_a = np.array(df_a.binned_intensities.values.tolist())
    df_b = np.array(df_b.binned_intensities.values.tolist())
    pvals = np.array([ttest_ind(df_a[:, k], df_b[:, k], equal_var = False)[1] for k in range(max_bin)])
    tested = ~np.isnan(pvals)
    pvals[tested] = correct_multiple_testing(pvals[tested], 0.05)[0]
    cohensd = [cohen_d(df_a[:, k], df_b[:, k]) for k in range(max_bin)]
    sig_bins = [k for k in range(max_bin) if pvals[k] < 0.05]
    sig_bins = [(min_mz + ((max_mz - min_mz) / (bin_num - 1)) * k + remainder[k], pvals[k], cohensd[k][0]) for k in
                sig_bins]
    return sorted(sig_bins, key = lambda x: (x[1], 1 / abs(x[2])))


def follow_sigs(df, glycan_list, mz_cap = 3000, max_mz = 3000, min_mz = 39.714, bin_num = 2048,
                motif = None, motif2 = None, thresh = 0.5,
                conf_range = [(0.9, 1.0), (0.8, 0.9), (0.6, 0.8), (0.4, 0.6), (0.2, 0.4), (0, 0.2)]):
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
    max_bin = round((mz_cap - min_mz) / ((max_mz - min_mz) / (bin_num - 1)))
    df_a = df[df.Prediction == glycan_list[0]].reset_index(drop = True)
    df_b = df[df.Prediction == glycan_list[1]].reset_index(drop = True)
    df_r = pd.concat([df_a, df_b], axis = 0)
    remainder = [np.median([x for x in col if x] or [0]) for col in zip(*df_r.mz_remainder.values.tolist())]
    bins = {min_mz + ((max_mz - min_mz) / (bin_num - 1)) * k + remainder[k]: [] for k in range(max_bin)}
    bin_keys = list(bins)
    for conf in conf_range:
        df_a2 = df_a[df_a.Confidence.between(conf[0], conf[1])]
        df_b2 = df_b[df_b.Confidence.between(conf[0], conf[1])]
        df_a2 = np.array(df_a2.binned_intensities.values.tolist())
        df_b2 = np.array(df_b2.binned_intensities.values.tolist())
        cohensd = [cohen_d(df_a2[:, k], df_b2[:, k]) for k in range(max_bin)]
        pvals = np.array([ttest_ind(df_a2[:, k], df_b2[:, k], equal_var = False)[1] for k in range(max_bin)])
        tested = ~np.isnan(pvals)
        pvals[tested] = correct_multiple_testing(pvals[tested], 0.05)[0]
        for c, cd in enumerate(cohensd):
            bins[bin_keys[c]].append(cd[0] if pvals[c] < 0.05 else 0)
    bins = {k: v for k, v in bins.items() if
            max([abs(v2) for v2 in v]) >= thresh and not any([math.isinf(v2) for v2 in v])}
    conf_idx = [c[1] for c in conf_range]
    for key, data_list in bins.items():
        plt.plot(conf_idx, data_list, label = key)
    plt.legend()
    return bins


def domon_costello_to_mpl(dc_name):
    """Converts a Domon-Costello fragment name to matplotlib mathtext format\n
    | Arguments:
    | :-
    | dc_name (list): a list of Domon-Costello cleavage names\n
    | Returns:
    | :-
    | Returns a matplotlib-renderable string with correct superscript/subscript formatting
    """
    greek = {'Alpha': r'\alpha', 'Beta': r'\beta', 'Gamma': r'\gamma',
             'Delta': r'\delta', 'Epsilon': r'\epsilon', 'Zeta': r'\zeta',
             'Eta': r'\eta', 'Theta': r'\theta', 'Iota': r'\iota',
             'Kappa': r'\kappa', 'Lambda': r'\lambda', 'Mu': r'\mu'}
    mpl_parts = []
    for nom in dc_name:
        parts = nom.split('_')
        if len(parts) == 3:
            frag_type, number, chain = parts
            chain_greek = greek.get(chain, chain)
            if len(frag_type) > 1:
                mpl_parts.append(f"$^{{{frag_type[0]},{frag_type[1]}}}{frag_type[2]}_{{{number}{chain_greek}}}$")
            else:
                mpl_parts.append(f"${frag_type}_{{{number}{chain_greek}}}$")
        elif len(parts) >= 2 and parts[0] == 'M':
            # Adducts (+Na) are gains, everything else a loss; element counts become subscripts, a leading count (2H2O) stays a multiplier
            terms = ''.join(f" + {x[1:]}" if x.startswith('+') else f" - {x}" for x in '_'.join(parts[1:]).split('|'))
            mpl_parts.append('$M' + re.sub(r'(?<=[A-Za-z])(\d+)', r'_{\1}', terms) + '$')
        elif parts[0] == 'M':
            mpl_parts.append('$M$')
        else:
            mpl_parts.append(nom)
    return "/".join(mpl_parts)


# Diagnostic oxonium ions as named in the glycoproteomics literature; masses are the singly-protonated ions
def _oxonium_mass(residues, losses = ()):
    return sum(mono_attributes[r]['mass'][r] for r in residues) - sum(losses) + PROTON_MASS


OXONIUM_IONS = {
    'HexNAc': _oxonium_mass(['HexNAc']),
    'HexNAc-H2O': _oxonium_mass(['HexNAc'], [WATER_MASS]),
    'HexNAc-2H2O': _oxonium_mass(['HexNAc'], [2 * WATER_MASS]),
    'HexNAc-C2H4O2': _oxonium_mass(['HexNAc'], [60.0211]),
    'HexNAc-CH2O-2H2O': _oxonium_mass(['HexNAc'], [30.0106, 2 * WATER_MASS]),
    'HexNAc-C2H4O2-H2O': _oxonium_mass(['HexNAc'], [60.0211, WATER_MASS]),
    'Hex': _oxonium_mass(['Hex']),
    'Hex-H2O': _oxonium_mass(['Hex'], [WATER_MASS]),
    'dHex': _oxonium_mass(['dHex']),
    'HexHexNAc': _oxonium_mass(['Hex', 'HexNAc']),
    'HexHexNAc-H2O': _oxonium_mass(['Hex', 'HexNAc'], [WATER_MASS]),
    'Neu5Ac': _oxonium_mass(['Neu5Ac']),
    'Neu5Ac-H2O': _oxonium_mass(['Neu5Ac'], [WATER_MASS]),
    'Neu5Gc': _oxonium_mass(['Neu5Gc']),
    'Neu5Gc-H2O': _oxonium_mass(['Neu5Gc'], [WATER_MASS]),
    'HexNAcdHex': _oxonium_mass(['HexNAc', 'dHex']),
    'HexNAc2': _oxonium_mass(['HexNAc', 'HexNAc']),
    'Neu5AcHex': _oxonium_mass(['Neu5Ac', 'Hex']),
    'HexHexNAcdHex': _oxonium_mass(['Hex', 'HexNAc', 'dHex']),
    'Hex2HexNAc': _oxonium_mass(['Hex', 'Hex', 'HexNAc']),
    'Neu5AcHexHexNAc': _oxonium_mass(['Neu5Ac', 'Hex', 'HexNAc']),
    'Neu5GcHexHexNAc': _oxonium_mass(['Neu5Gc', 'Hex', 'HexNAc']),
}
PEAK_COLORS = {'oxonium': '#b8860b', 'backbone_n': '#1f77b4', 'backbone_c': '#2ca02c',
               'glycopeptide': '#d62728', 'glycan': '#d62728'}


def identify_oxonium(mz, tolerance_ppm = 20):
    """Returns the conventional name of a diagnostic oxonium ion at this m/z, or None"""
    for name, theoretical in OXONIUM_IONS.items():
        if abs(mz - theoretical) <= theoretical * tolerance_ppm / 1e6:
            return name
    return None


def resolve_spectrum_input(input_string):
    """Splits any CandyCrumbs input into its peptide, glycan and glycosite components"""
    if isinstance(input_string, dict) and 'peptide' in input_string:
        return (input_string['peptide'], list(input_string.get('glycans', [])),
                list(input_string.get('glycosites', [])))
    if isinstance(input_string, str) and '*' in input_string:
        parsed = glycopeptide_string_to_input(input_string)
        return parsed['peptide'], list(parsed['glycans']), list(parsed['glycosites'])
    return '', [input_string] if isinstance(input_string, str) else [], []


def classify_fragment(dc_name):
    """Labels a fragment as an oxonium, a peptide backbone, a glycan or an intact-peptide glycopeptide ion"""
    if not dc_name:
        return None
    if not isinstance(dc_name[0], list):
        return 'glycan'
    peptide_part = [str(x) for x in dc_name[0]]
    if 'No Peptide' in peptide_part:
        return 'oxonium'
    backbone = [x[0] for x in peptide_part if re.fullmatch(r'[abcwxyz]_\d+', x)]
    if backbone:
        return 'backbone_n' if backbone[0] in N_TERM_IONS else 'backbone_c'
    return 'glycopeptide'


def fragment_to_fragIUPAC(glycan_string, dc_name):
    """Reduces the glycan part of a fragment label to the IUPAC-condensed structure it describes"""
    flat = [y for sub in dc_name for y in sub] if dc_name and isinstance(dc_name[0], list) else list(dc_name)
    chain_lengths = {rank: len(chain) for rank, chain in
                     rank_chains(mono_graph_to_nx(glycan_to_graph_monos(glycan_string), directed = True))}
    cuts = []
    for cut in flat:
        parts = str(cut).split('_')
        if len(parts) != 3 or parts[2] not in ranks:
            continue
        # A cleavage at or beyond the end of a chain is the bond to the peptide, which the glycan alone lacks
        limit = chain_lengths.get(parts[2], 0)
        if parts[0][-1] in 'BCYZ' and int(parts[1]) >= limit:
            continue
        if parts[0][-1] in 'AX' and int(parts[1]) > limit:
            continue
        cuts.append(cut)
    try:
        return domon_costello_to_fragIUPAC(glycan_string, cuts) if cuts else glycan_string
    except Exception:
        return None


def fragIUPAC_to_image(frag_iupac, padding = 4):
    """Renders a glycan fragment as a cropped SNFG cartoon, or None if the drawing stack is unavailable"""
    try:
        from io import BytesIO
        from glycowork.motif.draw import GlycoDraw
        from glycorender.render import convert_svg_to_png
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(BytesIO(convert_svg_to_png(GlycoDraw(frag_iupac, suppress = True, compact = True,
                                                                dim = 30).as_svg(), None, return_bytes = True)))
    except Exception:
        return None
    box = image.convert('RGBA').getbbox()
    if box:
        image = image.crop((max(0, box[0] - padding), max(0, box[1] - padding),
                            min(image.width, box[2] + padding), min(image.height, box[3] + padding)))
    return image


def place_peak_label(ax, renderer, mz, rel_int, label, color, placed, bounds, fontsize, attempts, pad = 1.5):
    """Puts a label immediately above its peak, lifting it only as far as the neighbors require; if the
    axes run out of room above, the label is flipped to hang below the apex instead"""
    for direction, alignment, vertical, x_offset in ((1, 'left', 'center', 0.0), (-1, 'right', 'top', 2.0)):
        # A flipped label runs down alongside its peak, so it is pushed off the line rather than centered on it
        text = ax.annotate(label, (mz, rel_int), textcoords = 'offset points',
                           xytext = (x_offset, direction * 3.0), ha = alignment, va = vertical,
                           fontsize = fontsize, rotation = 90, rotation_mode = 'anchor', color = color)
        for _ in range(attempts):
            box = text.get_window_extent(renderer)
            if box.y1 > bounds[1] or box.y0 < bounds[0]:
                break
            padded = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
            clashes = [other for other in placed if padded.overlaps(other)]
            if not clashes:
                placed.append(padded)
                return text
            shift = (max(other.y1 for other in clashes) - box.y0 + pad if direction > 0 else
                     min(other.y0 for other in clashes) - box.y1 - pad)
            text.xyann = (text.xyann[0], text.xyann[1] + shift)
        text.remove()
    return None


def draw_peptide_ladder(peptide, hit_dict, glycosites = (), ax = None, fontsize = 9):
    """Draws the peptide fragment ladder: N-terminal ions tick up-left, C-terminal ions tick down-right"""
    n_cuts, c_cuts = {}, {}
    for hit in hit_dict.values():
        if not hit:
            continue
        dc_name = hit['Domon-Costello nomenclatures'][0]
        if not dc_name or not isinstance(dc_name[0], list):
            continue
        for token in dc_name[0]:
            match = re.fullmatch(r'([abcwxyz])_(\d+)', str(token))
            if not match:
                continue
            series, index = match.group(1), int(match.group(2))
            target = n_cuts if series in N_TERM_IONS else c_cuts
            target.setdefault(index if series in N_TERM_IONS else len(peptide) - index, set()).add(
                f'{series}{index}')
    for cuts, sign, align in ((n_cuts, 1, 'right'), (c_cuts, -1, 'left')):
        for bond, tokens in cuts.items():
            x = bond - 0.5
            color = PEAK_COLORS['backbone_n' if sign > 0 else 'backbone_c']
            ax.plot([x, x, x - sign * 0.35], [sign * 0.22, sign * 0.55, sign * 0.55], color = color, lw = 1.1,
                    solid_capstyle = 'round')
            ax.text(x - sign * 0.4, sign * 0.62, '/'.join(sorted(tokens)), ha = align,
                    va = 'bottom' if sign > 0 else 'top', fontsize = fontsize - 3, color = color)
    for i, aa in enumerate(peptide):
        ax.text(i, 0, aa, ha = 'center', va = 'center', fontsize = fontsize, family = 'monospace',
                fontweight = 'bold' if i in glycosites else 'normal',
                color = '#b8860b' if i in glycosites else 'black')
    for site in glycosites:
        ax.plot([site, site], [0.25, 0.95], color = '#b8860b', lw = 0.8, ls = '--')
    ax.text(0.0, 1.0, f'{len(set(n_cuts) | set(c_cuts))}/{len(peptide) - 1} backbone bonds',
            transform = ax.transAxes, ha = 'left', va = 'top', fontsize = fontsize - 2, color = '#666666')
    ax.set_xlim(-1, len(peptide))
    ax.set_ylim(-1.25, 1.25)
    ax.axis('off')
    return ax


def plot_annotated_spectrum(input_string, spectrum, intensities = None, mass_threshold = None,
                            max_cleavages = 3, mass_tag = None, sample_prep = 'underivatized',
                            disable_global_mods = False, prior_weight = 1.0, charge = None, ax = None,
                            annotate_top_n = None, annotation_threshold = 0.05, figsize = None,
                            draw_glycans = True, glycan_zoom = 0.25, max_glycan_cartoons = 8,
                            label_fontsize = 6, max_label_levels = 6, filepath = '', **kwargs):
    """Plots an MS2 spectrum annotated with CandyCrumbs fragment assignments\n
    | Arguments:
    | :-
    | input_string (string/dict): anything CandyCrumbs accepts, i.e., a glycan, a peptide*glycan* string, or an input dict
    | spectrum (dataframe/list): either the prediction dataframe from wrap_inference, or the observed m/z values
    | intensities (list): the peak_d list from wrap_inference when spectrum is a dataframe, else the intensities
    | mass_threshold (float): maximum tolerated mass difference for fragment matching; default:None (see CandyCrumbs)
    | max_cleavages (int): maximum concurrent fragmentations per mass; default:3
    | mass_tag (float): mass of the glycan label or reducing end modification; default:None
    | sample_prep (string): underivatized/permethylated/peracetylated; default:'underivatized'
    | disable_global_mods (bool): whether to disable global modifications; default:False
    | prior_weight (float): weighting of prior-informed scoring; default:1.0
    | charge (int): charge state of the precursor ion; taken from the dataframe when available; default:None
    | ax (matplotlib axis): axis to plot on, creates a new figure if None; default:None
    | annotate_top_n (int): only label the N most intense annotated peaks; default:None (label all)
    | annotation_threshold (float): minimum relative intensity (0-1) to annotate a peak; default:0.05
    | figsize (tuple): figure size if creating a new figure; default:None (chosen for the input type)
    | draw_glycans (bool): whether to add SNFG cartoons for glycan fragments; default:True
    | glycan_zoom (float): scale of the SNFG cartoons; default:0.25
    | max_glycan_cartoons (int): how many cartoons to draw, most intense first; default:8
    | label_fontsize (int): font size of the peak labels; default:6
    | max_label_levels (int): how many times to push a label clear of its neighbors before dropping it; default:6
    | filepath (string): where to save the figure; default:'' (not saved)
    | **kwargs: passed straight to CandyCrumbs, e.g., mass_threshold_ppm, fragmentation_method, max_global_mods\n
    | Returns:
    | :-
    | (1) the CandyCrumbs hit_dict for downstream use
    | (2) the matplotlib axis object
    """
    if isinstance(spectrum, pd.DataFrame):
        top1_preds = [p[0][0] if p else '' for p in spectrum['predictions']]
        matches = [i for i, pred in enumerate(top1_preds) if pred == input_string]
        if not matches:
            raise ValueError(f"'{input_string}' not found as a top1 prediction in the prediction dataframe")
        peak_d = intensities[matches[0]]
        charge = spectrum.iloc[matches[0]]['charge'] if charge is None else charge
        mz_values = np.array(sorted(peak_d.keys()), dtype = float)
        peak_intensities = np.array([peak_d[mz] for mz in mz_values], dtype = float)
    else:
        if intensities is None:
            raise ValueError("intensities must be given alongside a list of m/z values")
        mz_values, peak_intensities = np.asarray(spectrum, dtype = float), np.asarray(intensities, dtype = float)
        order = np.argsort(mz_values)
        mz_values, peak_intensities = mz_values[order], peak_intensities[order]
        if charge is None:
            raise ValueError("charge must be given when passing raw m/z values")
    rel_intensities = peak_intensities / peak_intensities.max() * 100 if peak_intensities.max() > 0 else peak_intensities
    hit_dict = CandyCrumbs(input_string, mz_values.tolist(), mass_threshold, max_cleavages = max_cleavages,
                           simplify = True, charge = charge, mass_tag = mass_tag, sample_prep = sample_prep,
                           disable_global_mods = disable_global_mods, prior_weight = prior_weight, **kwargs)
    peptide, glycans, glycosites = resolve_spectrum_input(input_string)
    glycan_string = glycans[0] if glycans and isinstance(glycans[0], str) and not is_composition(glycans[0]) else None
    ladder_ax = None
    cartoon_room = 0.26 if draw_glycans and glycan_string else 0.02
    if ax is None:
        if peptide:
            _, (ladder_ax, ax) = plt.subplots(2, 1, figsize = figsize or (13, 5.6),
                                              gridspec_kw = {'height_ratios': [1, 3.4],
                                                             'hspace': 0.15 + cartoon_room})
        else:
            _, ax = plt.subplots(figsize = figsize or (12, 4.6))
            ax.figure.subplots_adjust(top = 0.96 - cartoon_room)
    elif peptide:
        ladder_ax = ax.inset_axes([0.28, 0.58, 0.70, 0.30])
    ax.vlines(mz_values, 0, rel_intensities, colors = 'grey', linewidth = 0.8, alpha = 0.4)
    peaks = []
    for mz, rel_int in zip(mz_values, rel_intensities):
        hit = hit_dict.get(mz)
        if not hit:
            continue
        dc_name = hit['Domon-Costello nomenclatures'][0]
        kind = classify_fragment(dc_name)
        oxonium = identify_oxonium(hit['Theoretical fragment masses'][0]) if kind in ('oxonium', 'glycan') and charge > 0 else None
        flat = [y for sub in dc_name for y in sub] if dc_name and isinstance(dc_name[0], list) else list(dc_name)
        shown = [x for x in flat if x != 'No Peptide' and not str(x).startswith('loss of')]
        label = oxonium if oxonium else domon_costello_to_mpl(shown or ['M'])
        z = hit['Fragment charges'][0]
        if abs(z) > 1:
            label += f" [{'+' if z > 0 else ''}{z}]"
        peaks.append((mz, rel_int, label, kind, dc_name))
    if annotate_top_n is not None:
        keep = {id(p) for p in sorted(peaks, key = lambda x: -x[1])[:annotate_top_n]}
        peaks = [p for p in peaks if id(p) in keep]
    for mz, rel_int, _, kind, _ in peaks:
        ax.vlines([mz], 0, rel_int, colors = PEAK_COLORS.get(kind, 'tab:red'), linewidth = 1.3)
    # Each label sits directly above its peak, lifted only as far as its neighbors require, with a dotted
    # leader whenever it had to move; anything that cannot be placed at all is dropped rather than overlaid
    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    placed = []
    for mz, rel_int, label, kind, _ in sorted(peaks, key = lambda x: -x[1]):
        if rel_int < annotation_threshold * 100:
            continue
        color = PEAK_COLORS.get(kind, 'tab:red')
        text = place_peak_label(ax, renderer, mz, rel_int, label, color, placed,
                                (axes_box.y0, axes_box.y1), label_fontsize, max_label_levels)
        if text is not None and abs(text.xyann[1]) > 8:
            ax.annotate('', xy = (mz, rel_int),
                        xytext = (text.xyann[0], text.xyann[1] - np.sign(text.xyann[1])),
                        textcoords = 'offset points',
                        arrowprops = dict(arrowstyle = '-', lw = 0.4, ls = ':', color = color,
                                          shrinkA = 0, shrinkB = 1))
    cartoons_drawn = False
    if draw_glycans and glycan_string:
        blended = ax.get_xaxis_transform()
        # Only glycan-only ions get a cartoon; on a peptide-bearing fragment the picture would just repeat
        # the intact glycan. One cartoon per distinct structure, anchored on its most intense member with a
        # leader from every peak it explains: an oxonium series shares a structure and differs only by a
        # neutral loss, so a box per loss repeats the same picture and drags them away from their peaks
        groups = {}
        for mz, rel_int, _, kind, dc_name in peaks:
            # A leader to a noise-level peak is a line across the plot for nothing, so a cartoon only
            # claims the peaks that were worth labeling in the first place
            if kind not in ('oxonium', 'glycan') or rel_int < annotation_threshold * 100:
                continue
            frag_iupac = fragment_to_fragIUPAC(glycan_string, dc_name)
            if frag_iupac is not None:
                groups.setdefault(frag_iupac, []).append((rel_int, mz))
        ranked = sorted(groups.items(), key = lambda group: -max(group[1])[0])[:max_glycan_cartoons]
        mz_per_pixel = (ax.get_xlim()[1] - ax.get_xlim()[0]) / max(axes_box.width, 1)
        rows, row_ends = [1.08, 1.26], [-np.inf, -np.inf]
        for frag_iupac, members in sorted(ranked, key = lambda group: max(group[1])[1]):
            image = fragIUPAC_to_image(frag_iupac)
            if image is None:
                continue
            half_width = (image.width * glycan_zoom * figure.dpi / 72 + 8) * mz_per_pixel / 2
            anchor = max(members)[1]
            row = min(range(len(rows)), key = lambda i: max(anchor, row_ends[i] + half_width))
            x = max(anchor, row_ends[row] + half_width)
            ax.add_artist(AnnotationBbox(OffsetImage(np.array(image), zoom = glycan_zoom), (x, rows[row]),
                                         xycoords = blended, frameon = True, annotation_clip = False,
                                         bboxprops = dict(boxstyle = 'round,pad=0.15', fc = 'white',
                                                          ec = '#cccccc', lw = 0.5)))
            for rel_int, mz in members:
                ax.plot([mz, x], [rel_int / max(rel_intensities) * 0.95, rows[row] - 0.05], transform = blended,
                        color = '#cccccc', lw = 0.5, ls = '--', clip_on = False)
            row_ends[row] = x + half_width
            cartoons_drawn = True
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity (%)')
    stage = f'MS\u00b3 spectrum of m/z {kwargs["ms3_precursor"]:.2f}' if kwargs.get('ms3_precursor') is not None else 'MS\u00b2 spectrum'
    title = f'Annotated {stage}: {peptide + "*" + str(glycan_string) if peptide else input_string}'
    title_size = 9 if len(title) > 90 else 'large'
    if ladder_ax is not None:
        ladder_ax.set_title(title, fontsize = title_size)
    else:
        ax.set_title(title, fontsize = title_size, y = 1.40 if cartoons_drawn else 1.0)
    ax.set_xlim(mz_values.min() - 20, mz_values.max() + 20)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ladder_ax is not None:
        draw_peptide_ladder(peptide, hit_dict, glycosites, ax = ladder_ax)
    if filepath:
        plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')
    return hit_dict, ax
