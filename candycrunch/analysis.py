import copy
import math
import os
from pathlib import Path
import ast
import pickle
from collections import Counter
from functools import lru_cache
import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations, product
from operator import neg
import bisect
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import pandas as pd
from glycowork.motif.processing import canonicalize_composition, is_composition, rescue_glycans, get_class
from glycowork.motif.tokenization import map_to_basic as glycowork_map_to_basic, HYDROGEN_MASS, PROTON_MASS, ELECTRON_MASS, calculate_adduct_mass, \
    composition_to_mass, glycan_to_composition, get_core, get_modification, compositions_to_structures, AMINO_ACID_FORMULAS, PEPTIDE_MODIFICATIONS, \
    SIALIC_DERIVATIZATIONS, SIALIC_PRODUCTS, SIALIC_PRODUCT_CAPS, SIALIC_KEYS, _carboxyl_sites, modification_formula_dict
from glycowork.motif.graph import glycan_to_nxGraph, get_possible_topologies, graph_to_string
from glycowork.motif.smiles import glycan_to_smiles, parse_smiles, _rings_of_atoms, _number_residue, _split_token
from glycowork.motif.draw import plot_glycans_excel
from glycowork.glycan_data.stats import cohen_d, correct_multiple_testing
from scipy.stats import ttest_ind


@lru_cache(maxsize = 4096)
def map_to_basic(glycoletter, obfuscate_ptm = True):
    """glycowork's map_to_basic, except that Galf (also modified, Galf3S) and Ins keep their own label: glycowork counts them as Hex, whose pyranose
    cross-rings a furanose (2,4A/X of other atoms, no 0,4/2,5/3,5 at all) or a cyclitol does not have"""
    basic = glycowork_map_to_basic(glycoletter, obfuscate_ptm = obfuscate_ptm)
    return glycoletter if basic.startswith('Hex') and re.match(r'(Galf|Ins)(?![a-z])', glycoletter) else basic

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
    'Global': {'mass': {'H2O': -18.0105546, 'NH3': -17.026549, 'CH2O': -30.0106, 'CH4O': -32.026215, 'C2H2O': -42.0106, 'CO2': -43.9898,
                        'SO4': -79.9568, 'PO4': -79.9663, 'C3H8O4': -108.0423, 'C2H4O2': -60.0211, 'CD3OH': -35.045046,
                        '+Acetate': 59.013851,
                        '+Na': +22.989218, '+K': 38.963158, '+Li': 7.015454, '+H': HYDROGEN_MASS, '-H': -HYDROGEN_MASS}}
}
WATER_MASS = 18.0105546
CH2_MASS = calculate_adduct_mass('CH2')
bond_type_helper = {1: ['bond', 'no_bond'], 2: ['red_bond', 'red_no_bond'], 3: ['peptide_a', 'peptide_b', 'peptide_c', 'peptide_c-1'],
                    4: ['peptide_y', 'peptide_z', 'peptide_w', 'peptide_z+1']}
# Neutral radical lost from a z. ion by Cbeta-Cgamma homolysis, giving the w ion; residues without a
# gamma atom (G, A, P) cannot form w ions and are absent by design
W_SIDE_CHAIN_LOSSES = {'C': 32.979896, 'D': 44.997655, 'E': 59.013305, 'F': 77.039125, 'H': 67.029623,
                       'I': 29.039125, 'K': 58.065674, 'L': 43.054775, 'M': 61.011196, 'N': 44.013639,
                       'Q': 58.029289, 'R': 86.071822, 'S': 17.002740, 'T': 15.023475, 'V': 15.023475,
                       'W': 116.049024, 'Y': 93.034040}
W_SIDE_CHAIN_LOSSES['c'] = 90.001360  # Carbamidomethyl Cys, cleaved at Cbeta-Sgamma
W_SIDE_CHAIN_LOSSES['m'] = W_SIDE_CHAIN_LOSSES['M'] + 15.994915  # Met sulfoxide, the lost radical keeps the oxygen
# the radical lost by a labeled Lys keeps the label on its Nepsilon (j: guanidinyl CH2N2, k: TMT)
W_SIDE_CHAIN_LOSSES['j'] = W_SIDE_CHAIN_LOSSES['K'] + 42.02180
W_SIDE_CHAIN_LOSSES['k'] = W_SIDE_CHAIN_LOSSES['K'] + 229.16293
# deamidated Asn and Gln are Asp and Glu
W_SIDE_CHAIN_LOSSES['n'] = W_SIDE_CHAIN_LOSSES['D']
W_SIDE_CHAIN_LOSSES['q'] = W_SIDE_CHAIN_LOSSES['E']
cut_type_dict = {'bond': 'Y', 'no_bond': 'Z', 'red_bond': 'C', 'red_no_bond': 'B',
                 '13A': '13A', '14A': '14A', '15A': '15A', '24A': '24A', '04A': '04A', '35A': '35A', '03A': '03A',
                 '25A': '25A', '02A': '02A',
                 '02X': '02X', '03X': '03X', '04X': '04X', '12X': '12X', '13X': '13X', '14X': '14X', '15X': '15X',
                 '24X': '24X', '25X': '25X', '35X': '35X'}
A_cross_rings = {c for c in cut_type_dict if c[-1] == 'A'}
X_cross_rings = {c for c in cut_type_dict if c[-1] == 'X'}
ranks = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu']
# Residue masses from glycowork's formulas (the old 4-decimal table was up to 0.1 mDa off), plus the modified residues CandyCrumbs encodes
AA_masses = {aa: calculate_adduct_mass(f) for aa, f in AMINO_ACID_FORMULAS.items()}
# the tokens of the modified residues, by residue and the modification's Unimod name in glycowork's PEPTIDE_MODIFICATIONS (TMT-labeled Lys, Carbamidomethyl Cys,
# Guanidinyl Lys, Met sulfoxide, the variable modification of nearly every glycoproteomics search, and deamidated Asn and Gln, which PNGase F leaves at
# released N-glycosites), so that peptide_to_glycowork can write them as glycowork does
TOKEN_MODIFICATIONS = {'k': ('K', 'TMT6plex'), 'c': ('C', 'Carbamidomethyl'), 'j': ('K', 'Guanidinyl'), 'm': ('M', 'Oxidation'), 'n': ('N', 'Deamidated'),
                       'q': ('Q', 'Deamidated')}
AA_masses |= {token: AA_masses[aa] + (float(PEPTIDE_MODIFICATIONS[name]) if re.fullmatch(r'[+-]\d*\.?\d+', PEPTIDE_MODIFICATIONS[name]) else calculate_adduct_mass(PEPTIDE_MODIFICATIONS[name]))
              for token, (aa, name) in TOKEN_MODIFICATIONS.items()}
MODIFICATION_TOKENS = {
    'Carbamidomethyl': {'C': 'c'},
    'Guanidinyl': {'K': 'j'},
    'Oxidation': {'M': 'm'},
    'Deamidated': {'N': 'n', 'Q': 'q'},
}
# Relative abundance of the backbone ion types within a fragmentation method; the method already gates
# which types can occur, so these only rank the allowed types against each other
PEPTIDE_ION_PRIORS = {'y': 1.0, 'b': 0.9, 'c': 0.9, 'z': 0.9, 'a': 0.3, 'w': 0.3, 'x': 0.1}
# Electron transfer also gives c and z ions after a hydrogen atom moved between the two halves: c-1 (c - H, as pyteomics names it) and z+1 (z. + H, pyteomics'
# z+2, as CandyCrunch's z is the radical, pyteomics' z-dot). In ETD runs, unannotated peaks lay 1 H above a z ion as often as at its 13C peak,
# and with both ion types CandyCrumbs told the Byonic glycan of ion-trap ETD spectra from its isobar (Neu5Ac + Hex vs Neu5Gc + Fuc) in 0.41 instead of 0.37
N_TERM_IONS = {'a', 'b', 'c', 'c-1'}
C_TERM_IONS = {'w', 'x', 'y', 'z', 'z+1'}
PEPTIDE_ION_TYPES = {'CID': {'b', 'y'}, 'HCD': {'a', 'b', 'y'}, 'ETD': {'c', 'c-1', 'z', 'z+1'}, 'ECD': {'c', 'c-1', 'z', 'z+1'},
                     'EThcD': {'b', 'c', 'c-1', 'w', 'y', 'z', 'z+1'}, 'ETciD': {'b', 'c', 'c-1', 'w', 'y', 'z', 'z+1'},
                     None: (N_TERM_IONS | C_TERM_IONS) - {'c-1', 'z+1'}} | dict.fromkeys(('EED', 'EDD', 'NETD', 'UVPD'), (N_TERM_IONS | C_TERM_IONS) - {'c-1', 'z+1'})
tester_ma_addition = {k: {'mass': {k: v}, 'atoms': {k: [1, 2, 3, 4, 5, 6]}} for k, v in AA_masses.items()}
mono_attributes = mono_attributes | tester_ma_addition
bond_masses = {'red_bond': WATER_MASS, 'no_bond': -WATER_MASS, 'peptide_b': -WATER_MASS,
               'peptide_c': -(WATER_MASS - 17.026549),
               'peptide_z': -(17.026549 - 1.007825),
               'peptide_w': -(17.026549 - 1.007825),  # plus a residue-specific side-chain loss
               'peptide_a': -(WATER_MASS + 27.994915),  # z here is the EThcD radical z. convention
               'peptide_c-1': -(WATER_MASS - 17.026549) - 1.007825, 'peptide_z+1': -(17.026549 - 2 * 1.007825)}
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
# the mass of one derivatized group; deuteropermethylation (CD3 caps) derivatizes the permethylation sites, whose count tables it shares
DERIVATIZATION_MASSES = {'permethylated': CH2_MASS, 'deuteropermethylated': CH2_MASS + calculate_adduct_mass('D3-H3'), 'peracetylated': calculate_adduct_mass('C2H2O')}
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
        '+Acetate': 0.4, '+Na': 0.5, '+K': 0.3, '+Li': 0.5,
    },
    'multi_cleavage_penalty': {'glycosidic': 0.6, 'cross_ring': 0.35},
    # factor for a fragment one (odd-electron) or two hydrogen atoms off its even-electron mass, of radical activation methods (EED, EDD, NETD, UVPD, ETD)
    'hydrogen_shift': {1: 0.5, 2: 0.25},
    # an ion without a residue (HSO4-, a label's reporter ion) is specific to its group, so it outranks any single cleavage
    'small_ion': 1.5,
    # an open-chain ion of a labelled reducing end (CHAIN_IONS) is a cleavage of that residue, as likely as a Y ion
    'chain_ion': 1.0,
    # Alkali-cation-adducted fragments only (Na+/CID; from doi:10.1021/acs.jpca.6c00953): cross-ring dissociation needs reducing-end
    # ring-opening via the free anomeric O1, so it concentrates on the reducing-terminal residue.
    # X_1 cross-rings retain the reducing end and are the diagnostic class; cross-rings on
    # non-reducing-terminal residues are strongly disfavored. Detected per-fragment via the M_+Na
    # token, so protonated and negative-mode fragments (where internal 02A/24A are diagnostic) are untouched.
    'reducing_end_cross_ring': {'boost': 1.6, 'non_reducing_penalty': 0.2, 'adducts': {'+Na', '+K', '+Li'}},
}
# What a linkage-specific sialic acid derivatization makes of a carboxyl, written into the residue: at C1 of a sialic acid (Neu1Et5Ac is the ethyl ester of
# Neu5Ac, as glycowork writes its methyl ester Neu1Me5Ac), at C6 of a uronic acid (GlcA6Et)
CARBOXYL_TOKENS = {'ethyl ester': 'Et', 'methyl ester': 'Me', 'lactone': 'lactone', 'amide': 'N', 'methylamide': 'NMe', 'dimethylamide': 'NMe2',
                   'isopropylamide': 'NiPr', 'anilide': 'NPh', 'd5-anilide': 'NPhD5'}
CARBOXYL_LABELS = set()  # the residues derivatize_carboxyls made, which a composition counts like base residues
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


# The hand tables split a nonulosonic acid's ring with a hydrogen moving across the 0,4 and 2,4 cleavages (0,4A and 2,4X keep one more)
ULOSONIC_HYDROGEN_SHIFT = {'02': 0, '04': 1, '24': -1}
ELEMENT_VALENCES = {'C': (4,), 'N': (3, 5), 'O': (2,), 'P': (3, 5), 'S': (2, 4, 6), 'F': (1,), 'Cl': (1,), 'Br': (1,), 'I': (1,)}


def smiles_mono_attributes(label):
    """Fragment tables of any monosaccharide glycowork can write as SMILES (Kdo, heptoses, furanoses, QuiNAc, ...), read off its atoms\n
    | Arguments:
    | :-
    | label (string): monosaccharide, e.g. 'Kdo4P', 'LDManHepOPEtN' or 'Araf'\n
    | Returns:
    | :-
    | Returns its mono_attributes entry: mass, retained carbons and derivatized groups of every cross-ring fragment cut_type_dict names
    | (only the 0,2/0,4/2,4 pairs of the hand tables for ulosonic acids) and of the whole residue, keyed by its map_to_basic label
    """
    key, tables = map_to_basic(label, obfuscate_ptm = False), []
    # a substituent at an unknown position (KdoOP, LDManHepOPEtN) counts on every fragment keeping a hydroxyl, as for the residues
    # derived from compositions, so the ring is read without it and the substituent added afterwards
    stripped = label
    for _, mod in [m for m in _split_token(label)[1] if m[0] is None and m[1][0] != 'N' and m[1] != 'A']:
        cut = stripped.rfind(mod)
        stripped = stripped[:cut] + stripped[cut + len(mod):]
    for token in dict.fromkeys((stripped, label)):
        smiles = glycan_to_smiles(token)
        atoms, bonds, rings, _ = parse_smiles(smiles)
        elements = [a[0] for a in atoms]
        adjacency, valence, double = {i: set() for i in range(len(atoms))}, [0] * len(atoms), set()
        for a, b, o in bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
            valence[a] += o
            valence[b] += o
            if o == 2:
                double |= {(a, b), (b, a)}
        if any(e not in ELEMENT_VALENCES or valence[i] > ELEMENT_VALENCES[e][-1] for i, e in enumerate(elements)):
            raise ValueError(f"'{token}' has no SMILES with standard valences")
        hydrogens = [tok.count('H') if tok[0] == '[' else next(v for v in ELEMENT_VALENCES[elements[i]] if v >= valence[i]) - valence[i]
                     for i, tok in enumerate(re.findall(r'\[[^\]]*\]|Br|Cl|[BCNOPSFI]', smiles))]
        numbering = None
        for cycle in _rings_of_atoms(bonds, rings):
            try:
                numbering = (cycle, *_number_residue(cycle, elements, adjacency))
                break
            except ValueError:
                continue
        # atom 0 is the anomeric oxygen, or the group taking its place (Glc1Me, GlcNAc1P), and has to hang off the anomeric carbon alone
        if numbering is None or len(adjacency[0]) != 1:
            raise ValueError(f"'{token}' has no sugar ring with a free anomeric position")
        cycle, number, ring_oxygen, anomeric = numbering
        ring = [ring_oxygen, anomeric]
        while step := [a for a in adjacency[ring[-1]] if a in cycle and a not in ring]:
            ring.append(step[0])
        # every O-H and N-H is methylated but a sulfate's (a phosphate's charged oxygen counts as its OH); acetylation caps alcohols and free
        # amines once, never an O or N next to a carbonyl, sulfonyl or phosphoryl group
        capped = {i for i in range(len(atoms)) if any(elements[o] in 'CSP' and any(elements[q] == 'O' and q != i and (o, q) in double
                                                                                     for q in adjacency[o]) for o in adjacency[i])}
        methyls = [hydrogens[i] + (atoms[i][1] < 0) if elements[i] in 'ON' and not (elements[i] == 'O' and any(elements[o] == 'S' for o in adjacency[i]))
                   else 0 for i in range(len(atoms))]
        acetyls = [min(hydrogens[i], 1) if elements[i] in 'ON' and i not in capped and not atoms[i][1] else 0 for i in range(len(atoms))]
        acid = number[anomeric] == 2 and any(number.get(o) == 1 and any((o, q) in double for q in adjacency[o]) for o in adjacency[anomeric])
        cuts = [(i, j) for i in range(len(ring)) for j in range(i + 2, len(ring)) if (i, j) != (0, len(ring) - 1) and (not acid or f'{i}{j}' in ULOSONIC_HYDROGEN_SHIFT)]
        table = {'mass': {}, 'atoms': {}, 'permethylated': {}, 'peracetylated': {}}
        for i, j in [(None, None)] + cuts:
            # the two parts of the residue once ring bonds i and j are broken; the part with the anomeric carbon is a residue (minus water, and
            # minus the reducing-end group that the reducing-end bonus adds back), the other part keeps its atoms as they are
            broken = set() if i is None else {(ring[i], ring[i + 1]), (ring[i + 1], ring[i]), (ring[j], ring[(j + 1) % len(ring)]), (ring[(j + 1) % len(ring)], ring[j])}
            side, stack = {anomeric}, [anomeric]
            while stack:
                atom = stack.pop()
                for o in adjacency[atom]:
                    if o not in side and (atom, o) not in broken:
                        side.add(o)
                        stack.append(o)
            if i is not None and ring[i + 1] in side and ring[(j + 1) % len(ring)] in side:
                continue  # an anhydro bridge or a cyclic acetal keeps the residue in one piece
            for part, members in [('', side)] if i is None else [('X', side), ('A', set(range(len(atoms))) - side)]:
                name = key if i is None else f'{i}{j}{part}'
                if part and name not in cut_type_dict:
                    continue
                shift = ULOSONIC_HYDROGEN_SHIFT[f'{i}{j}'] * (1 if part == 'A' else -1) if acid and part else 0
                reducing = part != 'A'
                table['mass'][name] = sum(calculate_adduct_mass(elements[a]) + hydrogens[a] * HYDROGEN_MASS for a in members) + (shift - 2 * reducing) * HYDROGEN_MASS - reducing * calculate_adduct_mass('O')
                table['atoms'][name] = sorted(number[a] for a in members if a in number)
                table['permethylated'][name] = sum(methyls[a] for a in members) - reducing
                table['peracetylated'][name] = sum(acetyls[a] for a in members) - reducing
        tables.append(table)
    if len(tables) == 1:
        return tables[0]
    table, full = tables
    delta = {prop: full[prop][key] - table[prop][key] for prop in ['mass', *derivatization_sites]}
    return {prop: {f: v + delta[prop] if prop != 'atoms' and (table['peracetylated'][f] > 0 or f == key) else v for f, v in values.items()}
            for prop, values in table.items()}


# the hand tables of the base residues lack cross-rings (Hex 1,4A, HexNAc 0,3X, most of dHex, HexA, and Pen); they are filled from one stereoisomer each, which
# agrees with every hand entry, as masses, retained carbons and derivatized groups do not depend on the stereochemistry
for base, isomer in {'Hex': 'Glc', 'HexNAc': 'GlcNAc', 'HexN': 'GlcN', 'dHex': 'Fuc', 'HexA': 'GlcA', 'Pen': 'Xyl'}.items():
    generated = smiles_mono_attributes(isomer)
    for prop in ('mass', 'atoms', *derivatization_sites):
        mono_attributes[base][prop] = generated[prop] | mono_attributes[base][prop]
# inositol (GPI anchors) is a cyclitol without a ring oxygen and so without sugar cross-rings: only its residue, with the mass and derivatized groups of a Hex
mono_attributes['Ins'] = {prop: {'Ins': values['Hex']} for prop, values in mono_attributes['Hex'].items()}


def derivatize_carboxyls(glycan, sialic_derivatization):
    """Writes a linkage-specific sialic acid derivatization into a glycan or composition, as residues with fragment tables of their own\n
    | Arguments:
    | :-
    | glycan (string or dict): glycan in IUPAC-condensed format, or a composition with linkage-resolved sialic acids (e.g., 'H5N4E1L1')
    | sialic_derivatization (string or tuple): 'EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5' (glycowork's SIALIC_DERIVATIZATIONS), or a pair of its SIALIC_PRODUCTS
    |                                          for a2-6 and a2-3/8/9 sialic acids\n
    | Returns:
    | :-
    | Returns the glycan with every sialic acid written as its product at C1 (Neu5Ac(a2-6) of ethyl esterification as Neu1Et5Ac(a2-6), a lactone as
    | Neu1lactone5Ac) and every uronic acid as the a2-6 product at C6 (GlcA6Et), or the composition with such residues as keys; the product's mass counts
    | on every fragment that keeps that carbon, the lactone's water loss included (it bridges to the neighbouring residue, which is not modelled)
    """
    products = SIALIC_DERIVATIZATIONS.get(sialic_derivatization) if isinstance(sialic_derivatization, str) else tuple(sialic_derivatization)
    if products is None or len(products) != 2 or not set(products) <= SIALIC_PRODUCTS.keys():
        raise ValueError(f"Unknown sialic_derivatization {sialic_derivatization!r}; use one of {list(SIALIC_DERIVATIZATIONS)} or a pair of products for a2-6 and a2-3/8/9 sialic acids from {list(SIALIC_PRODUCTS)}.")
    deltas = [sum(calculate_adduct_mass(f) for f in SIALIC_PRODUCTS[p]) for p in products]
    # a product permethylated after the derivatization has its N-H capped instead of the carboxyl
    caps = [SIALIC_PRODUCT_CAPS[p] - 1 if SIALIC_PRODUCT_CAPS[p] is not None else 0 for p in products]
    # one product for every sialic acid needs no linkages
    same = products[0] == products[1]
    linked = {k: (base, i) for base, keys in SIALIC_KEYS.items() for i, k in enumerate(keys)} | {'HexA': ('HexA', 0)} | ({base: (base, 0) for base in SIALIC_KEYS} if same else {})
    composition = glycan if isinstance(glycan, dict) else canonicalize_composition(glycan) if is_composition(glycan) else None
    if composition is not None:
        if not same and (unknown := [k for k in SIALIC_KEYS if composition.get(k)]):
            raise ValueError(f"A linkage-specific derivatization makes something else of each sialic acid by its linkage, which {unknown} of {glycan} do not give; write them as a2-6 or a2-3 (e.g., 'H5N4E1L1').")
        sites = [(*linked[k], linked[k][0]) for k in composition if k in linked]
    else:
        sites = _carboxyl_sites(glycan)
        if not same and (unknown := [glycan[span[0]:span[1]] for _, i, span, _ in sites if i is None]):
            raise ValueError(f"A linkage-specific derivatization makes something else of each sialic acid by its linkage, which is unknown for {unknown} in '{glycan}' "
                             f"(a2-3/6, a2-?, or the reducing end); glycowork's resolve_sialic_linkages fills it in from a linkage-resolved composition.")
        glycan_to_graph_monos(glycan)  # the tables of the residues as written, which the derivatized ones build on
        sites = [(base, 0 if i is None else i, glycan[start:end], start, end) for base, i, (start, end), _ in sites]
    labels = {}
    for base, i, label, *_ in sites:
        new = label[:3] + '1' + CARBOXYL_TOKENS[products[i]] + label[3:] if base in SIALIC_KEYS else label + '6' + CARBOXYL_TOKENS[products[i]]
        old_key, new_key = map_to_basic(label, obfuscate_ptm = False), map_to_basic(new, obfuscate_ptm = False)
        if new_key not in mono_attributes and old_key in mono_attributes:
            carbon, table = 1 if base in SIALIC_KEYS else 6, mono_attributes[old_key]
            mono_attributes[new_key] = {'atoms': {new_key if f == old_key else f: a for f, a in table['atoms'].items()}} | {
                prop: {new_key if f == old_key else f: v + (deltas[i] if prop == 'mass' else caps[i] if prop == 'permethylated' else 0) * (carbon in table['atoms'][f]) for f, v in table[prop].items()}
                for prop in ['mass', *derivatization_sites] if prop in table}
        CARBOXYL_LABELS.add(new_key)
        labels[(label, i)] = new
    if composition is not None:
        out = {}
        for k, v in composition.items():
            new = labels[(linked[k][0], linked[k][1])] if k in linked else k
            out[new] = out.get(new, 0) + v
        return out
    for _, i, label, start, end in sorted(sites, key = lambda x: -x[3]):
        glycan = glycan[:start] + labels[(label, i)] + glycan[end:]
    return glycan


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
            comp = {}
        base = [k for k in comp if k not in SUBSTITUENTS]
        modification = get_modification(label)
        positions = re.findall(r'(\d|O)?(PCho|PEtN|Ac|Me|S|P)', modification)
        # glycowork's composition drops what it does not know (Qui3NAc becomes dHex, ManNAcA HexNAc), so a residue is
        # only derived if its substituents, an alditol, a stereo prefix, a lactone or an aglycone fully explain it; a substituent
        # written straight after another one belongs to it (GlcNAc1PP, Man3PMe, GlcN4PPEtN) and does not sit at an unknown position; a furanose (Galf2Ac,
        # GalfNAc) is no modified pyranose, though glycowork's composition counts it as one, and a modified inositol (Ins3PEtN) keeps only its residue
        if (len(base) != 1 or comp[base[0]] != 1 or base[0] not in derivatization_sites['permethylated'] or
                (core := map_to_basic(get_core(label), obfuscate_ptm = False)) not in (base[0], 'Ins') or
                Counter(s for _, s in positions) != {k: v for k, v in comp.items() if k in SUBSTITUENTS and k != '-H2O'} or
                re.sub(r'(\d|O)?(PCho|PEtN|Ac|Me|S|P)|\d,\dlactone|^[DL]-|-ol$|^1(Ser|Thr|Asn|Cer)$', '', modification) or
                re.search(r'(?<=[A-NP-Za-z])(PCho|PEtN|Ac|Me|S|P)', modification)):
            # everything else (Kdo, heptoses, furanoses, QuiNAc, diphosphates, ...) is read off its atoms, if glycowork can write it as SMILES
            try:
                mono_attributes[key] = smiles_mono_attributes(label)
            except ValueError:
                pass
            continue
        base = base[0]
        # a substituent without a number sits on the amine of a hexosamine (GlcNS), one on O at an unknown hydroxyl
        subs = [(s, int(p) if p.isdigit() else 2 if not p and get_core(label).endswith('N') else None) for p, s in
                positions] + [('-H2O', None)] * comp.get('-H2O', 0)
        base_attr = mono_attributes[base]
        frags = [f for f in base_attr['mass'] if f == base or core == base]
        hydroxyls = set(derivatization_sites['peracetylated'][base])
        in_frag = {f: [s for s, p in subs if p in base_attr['atoms'][f] or (p is None and hydroxyls & set(base_attr['atoms'][f]))]
                   for f in frags}
        mono_attributes[key] = {'atoms': {key if f == base else f: base_attr['atoms'][f] for f in frags}} | {
            prop: {key if f == base else f: base_attr[prop][f] + sum(SUBSTITUENTS[s][prop] for s in in_frag[f])
                   for f in frags} for prop in ['mass', *derivatization_sites]}
    adj_matrix = np.zeros((len(mono_mask_dic), len(mono_mask_dic)), dtype = int)
    # glycowork alternates monosaccharide (even) and linkage (odd) nodes, with edges pointing parent -> linkage -> child
    for parent, link in ggraph.edges():
        if link % 2:
            adj_matrix[link // 2, parent // 2] = 1
    return mono_mask_dic, adj_matrix, all_mask_dic


def mono_graph_to_nx(mono_graph, directed = True):
    """Modified version of glycan_to_nxGraph, converts a mono adjacency matrix into a networkx graph, adds bonds as edge labels, and terminal,reducing end labels\n
    | Arguments:
    | :-
    | mono_graph (tuple): output of glycan_to_graph_monos
    | directed (bool): if True, creates a directed graph with bonds pointing from leaf to reducing end ; default:True\n
    | Returns:
    | :-
    | Returns networkx graph object of a glycan made up of only monosaccharides
    """
    template = nx.DiGraph if directed else nx.Graph
    node_dict_mono, adj_matrix, all_dict = mono_graph
    if len(node_dict_mono) > 1:
        gr = nx.from_numpy_array(adj_matrix, create_using = template)
        for _, _, d in gr.edges(data = True):
            del d['weight']
    else:
        gr = template()
        gr.add_node(0)
    nx.set_node_attributes(gr, node_dict_mono, 'string_labels')
    nx.set_node_attributes(gr, {k: 'terminal' if gr.degree[k] == 1 else 'internal' for k in gr.nodes()}, 'termini')
    nx.set_node_attributes(gr, {max(gr.nodes()): 2}, 'reducing_end')
    nx.set_edge_attributes(gr, {(e[0], e[1]): {'bond_label': all_dict[(e[0] * 2) + 1]} for e in gr.edges})
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


def extend_subgraph(subgraph, extension, node, k, k_subgraphs, neighbor_dict, nx_mono, all_sizes = False, neighbors = None):
    """Main recursive feature of enumerate_k_graphs, calls itself to grow subgraph via an arbitrary path until it reaches size k\n
    | Arguments:
    | :-
    | subgraph (set): nodes making up a connected induced subgraph
    | extension (set): nodes neighbouring the subgraph with a higher node label than the start node
    | node (int): single node used to start the subgraph and take neighbors higher than
    | k (int): size of subgraph at which to stop extending and add it to the list of subgraphs
    | k_subgraphs (list): list used to accumulate all subgraphs already found of size k
    | neighbor_dict (dict): mapping of all nodes and their neighbors in the original graph
    | nx_mono (networkx_object): the original monosaccharide only graph being searched
    | all_sizes (bool): whether to also collect every smaller subgraph passed on the way to size k, as node sets instead of subgraph views; default:False
    | neighbors (set): all neighbors of the subgraph's nodes, if already at hand; default:None\n
    | Returns:
    | :-
    | Returns None
    """
    if len(subgraph) == k:
        k_subgraphs.append(subgraph if all_sizes else nx_mono.subgraph(subgraph))
        return None
    if all_sizes:
        k_subgraphs.append(subgraph)
    if neighbors is None:
        neighbors = {x for n in subgraph for x in neighbor_dict[n]}
    while extension:
        w = min(extension)
        extension.discard(w)
        # the exclusive neighbors of w: those that neighbor no node of the subgraph
        new_extension = extension | {x for x in neighbor_dict[w] - neighbors if x > node}
        extend_subgraph(subgraph | {w}, new_extension, node, k, k_subgraphs, neighbor_dict, nx_mono, all_sizes, neighbors | neighbor_dict[w])


def get_broken_bonds(subg, nx_edge_dict):
    """Determines bonds which are floating on the subgraph nodes\n
    | Arguments:
    | :-
    | subg (networkx_object): a subgraph of the original monosaccharide only graph
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
        atomic_mod_dict[terminal] = dict.fromkeys(mono_attributes[terminal_label]['atoms'][terminal_label], 0)
    # bonds at a known position come first, so that an ambiguous (3/4) or unknown (?) linkage cannot take an atom another bond is at
    for bond, bond_label in sorted(present_breakages.items(), key = lambda x: not x[1].split('-')[-1].isdigit()):
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
        if bond[0] in subg.nodes():
            # the child's own carbon in the bond, with or without an anomeric letter (a2-3, or 1-5 in teichoic acids)
            red_breakage = int(bond_label.split('-')[0][-1])
            atomic_mod_dict[bond[0]][red_breakage] = 2
        else:
            # an ambiguous linkage takes the last free position it lists (3/4: 4, else 3), a '?' linkage the first free position, most common
            # linkage first, so that two such linkages on one residue (Lewis a/x: Fuc(a1-3/4)[Gal(b1-3/4)]GlcNAc) cannot collapse onto one atom
            free = atomic_mod_dict[bond[1]]
            positions = [int(p) for p in bond_label.split('-')[-1].split('/') if p.isdigit()][::-1] or [3, 6, 2, 4, 5]
            breakage = next((a for a in positions if a in free and not free[a]), positions[0])
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
    | nx_edge_dict (dict): a mapping of each edge in the original graph to its bond label
    | allowed_X_cleavages (list): X-type cross-rings allowed on the terminal nodes other than root_node
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
        else:
            # an X cross-ring keeps C1 and has to keep the positions of the children still in the fragment (0,4X of a branching Man keeps its 3-arm and
            # loses its 6-arm); a child at an unknown position cannot be placed on the ring part, so it is not kept
            kept = {int(p) if (p := nx_edge_dict[bond]['bond_label'][-1]).isdigit() else None for bond in subg.in_edges(node)
                    if nx_edge_dict[bond]['bond_label'] not in ('glycosite', 'peptide')}
            terminal_mods.append([x for x in mono_attributes[basic_label]['mass'] if x == basic_label or (
                x in allowed_X_cleavages and kept <= set(mono_attributes[basic_label]['atoms'][x]))])
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
    idx_label = map_to_basic(label, obfuscate_ptm = False)
    A_mods_list = [x for x in mono_attributes[idx_label]['mass'] if x in A_cross_rings or x == label]
    bond_numbers = set(int(nx_edge_dict[bond]['bond_label'][-1])
                       for bond in subg.in_edges(node)
                       if nx_edge_dict[bond]['bond_label'][-1].isdigit())
    valid_A_mods_list = [mod for mod in A_mods_list
                         if bond_numbers <= set(mono_attributes[idx_label]['atoms'][mod])]
    return valid_A_mods_list


def generate_mod_permutations(terminals, terminal_labels, mono_mods_list, atomic_mod_dict_subg, peptide_ion_types = None):
    """Determines all possible monosaccharide modifications and their respective atom level representations\n
    | Arguments:
    | :-
    | terminals (list): node labels of nodes with floating bonds
    | terminal_labels (list): string labels of nodes in terminals
    | mono_mods_list (list): a nested list with one list of modifications per terminal node
    | atomic_mod_dict_subg (dict): nested dict of each terminal node with floating bonds labeled at each atom
    | peptide_ion_types (set): the peptide backbone ion types a cut peptide bond may give; default:None (all)\n
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
            # every bond permutation of the atoms with floating bonds
            modded_atoms = [k for k, v in atom_dict.items() if v in bond_type_helper]
            dict_perms = [{**atom_dict, **dict(zip(modded_atoms, perm))} for perm in product(*(bond_type_helper[atom_dict[y]] for y in modded_atoms))]
            if label not in W_SIDE_CHAIN_LOSSES:
                dict_perms = [x for x in dict_perms if 'peptide_w' not in x.values()]
            if peptide_ion_types is not None:
                dict_perms = [x for x in dict_perms if all(v[8:] in peptide_ion_types for v in x.values() if isinstance(v, str) and v.startswith('peptide_'))]
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
    reducing_ends = list(nx.get_node_attributes(subg, 'reducing_end'))
    if not reducing_ends:
        return False, None
    graph = subg if parent_graph is None else parent_graph
    bond_labels = nx.get_edge_attributes(graph, 'bond_label')
    for red_end in reducing_ends:
        for x in graph.in_edges(red_end):
            if bond_labels.get(x) == 'glycosite':
                return False, None
    return True, red_end


def reducing_end_bonus(root_mods, mass_tag, sample_prep = 'underivatized', root_label = None):
    """The mass a fragment holding the reducing end gains from it, for each cross-ring option of the reducing-end residue: its water, label (mass_tag)
    and extra derivatized groups, or with an A cross-ring only the label and groups if it keeps C1 of root_label (none without root_label)"""
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
    return bonus


def preliminary_calculate_mass(mono_mods_mass, atom_mods_mass, global_mods_mass, terminals,
                               inner_mass, bonus_root_mass, bonus_root_node, mass_tag, charge, mono_mod_perms,
                               perm_indices, sample_prep = 'underivatized', root_label = None):
    """Determines the mass of every requested permutation of monosaccharide, atom, and global modification\n
    | Arguments:
    | :-
    | mono_mods_mass (list): nested list with the mass of each cross ring option per terminal node
    | atom_mods_mass (list): nested list with the mass of each bond fragmentation option per terminal node
    | global_mods_mass (list): masses corresponding to each of the global mods
    | terminals (list): node labels of the terminal nodes
    | inner_mass (float): total mass of non-terminal nodes in subgraph
    | bonus_root_mass (bool): whether the subgraph holds the reducing end of the glycan, which adds its reducing-end mass
    | bonus_root_node (int): the node label of that reducing end
    | mass_tag (float): mass of the glycan label or reducing end modification
    | charge (int): assumed charge of glycan
    | mono_mod_perms (list): nested list with the name of each cross ring option per terminal node
    | perm_indices (array): one row per permutation to calculate, holding the option index of each terminal node
    | sample_prep (string): underivatized/permethylated/peracetylated
    | root_label (string): string label of the reducing-end node, for the A cross-rings keeping its C1; default:None\n
    | Returns:
    | :-
    | Returns a list every single mass of each modification combination for each cross ring combination
    """
    mode_mass = -PROTON_MASS if charge < 0 else PROTON_MASS
    bonus_pep_mass = WATER_MASS if any(isinstance(x, str) and x.split('-')[0] == '0' for x in terminals) else 0
    mono_arr = np.stack([np.array(x)[perm_indices[:, i]] for i, x in enumerate(mono_mods_mass)], axis = 1)
    atom_arr = np.stack([np.array(x)[perm_indices[:, i]] for i, x in enumerate(atom_mods_mass)], axis = 1)
    base_masses = inner_mass + mode_mass + bonus_pep_mass + mono_arr.sum(axis = 1) + atom_arr.sum(axis = 1)
    if bonus_root_mass:
        root_node_idx = terminals.index(bonus_root_node)
        base_masses += reducing_end_bonus(mono_mod_perms[root_node_idx], mass_tag, sample_prep, root_label)[perm_indices[:, root_node_idx]]
    if not global_mods_mass:
        return base_masses.tolist()
    global_arr = np.array(global_mods_mass)
    expanded = base_masses[:, np.newaxis] + np.concatenate([[0.0], global_arr])
    return expanded.ravel().tolist()


GLYCAN_ONLY_GLOBAL_MODS = {'CH2O', 'C2H2O', 'C2H4O2', 'C3H8O4', 'CO2', 'SO4', 'PO4'}
PEPTIDE_ONLY_GLOBAL_MODS = {'NH3'}
ADDUCT_GLOBAL_MODS = {'+Na', '+K', '+Li', '+Acetate'}
# the methanol only permethylated glycans lose, with its deuterated caps when deuteropermethylated
METHANOL_LOSSES = {'permethylated': 'CH4O', 'deuteropermethylated': 'CD3OH'}
# Only small neutral losses realistically occur more than once on a single fragment; combining the
# bulkier cross-ring losses would inflate the search space without describing real chemistry
REPEATABLE_GLOBAL_MODS = ('H2O', 'NH3')
# metal cations on a free glycan's fragment: one carries the charge, more neutralize acidic groups or carry more charges ([F+2Na-H]+, [F+2Na]2+)
METAL_GLOBAL_MODS = {'+Na', '+K', '+Li'}
# the ones a metal adduct precursor's fragments may carry unless its ion names another (precursor_metals): a +Li fragment lies one oxygen (16.0 Da) below the
# +Na fragment of the same cleavage, so sodiated precursors would gain chance annotations
DEFAULT_METAL_CATIONS = {'+Na', '+K'}


def precursor_metals(adducts):
    """metal_cations for the fragments of a precursor read as these adduct ions ('Li+', or ion names such as '[M+2Li]2+'): the metals of METAL_GLOBAL_MODS
    beyond DEFAULT_METAL_CATIONS that they name, else True (the default ones)"""
    return {f'+{m}' for a in adducts for m in re.findall(r'[A-Z][a-z]?', a)} & (METAL_GLOBAL_MODS - DEFAULT_METAL_CATIONS) or True
# Global modifications a fragment of a glycopeptide composition can carry, by the kind of ion (glycan-only, intact peptide, backbone ion): in spectra of
# identified glycopeptides (HCD, EThcD, ETD) only these were annotated more often than in the same spectra shifted out of tolerance; metal adducts, CH2O,
# CO2, C3H8O4 and H2O + NH3 never were, nor NH3 or two waters on backbone ions
GLYCOPEPTIDE_GLOBAL_MODS = {'No Peptide': {'H2O', '2H2O', 'C2H4O2'}, 'Peptide': {'H2O', '2H2O', 'NH3', '2NH3', 'C2H2O', 'C2H4O2'}, 'backbone': {'H2O'}}
BASIC_RESIDUES = {'H', 'K', 'R', 'j', 'k'}
# radical activation (electrons or UV photons) gives free glycans fragments one hydrogen atom off their even-electron mass (odd-electron ions) or two
# (hydrogen-shifted ions), written as global modifications (M_+H, M_2-H|H2O) that are no cleavage
RADICAL_METHODS = {'ETD', 'ECD', 'EThcD', 'ETciD', 'EED', 'EDD', 'NETD', 'UVPD'}
# Electron transfer or capture to a multiply protonated glycopeptide leaves the intact molecule with all its protons at a lower charge (no dissociation),
# one hydrogen atom more or less, and losing small groups from the radical: NH3, H2O, CO2 + H, and ketene or acetamide (ketene + NH3) of the HexNAc
# N-acetyl groups. In spectra of identified glycopeptides these were the most intense peaks left unannotated, each loss with these hydrogen shifts
# (relative to all protons kept) matched more often than in the same spectra shifted out of tolerance
CHARGE_REDUCING_METHODS = {'ETD', 'ECD', 'EThcD', 'ETciD'}
CHARGE_REDUCED_LOSSES = (((), (0, 1)), (('NH3',), (0, 1)), (('H2O',), (0, 1)), (('H2O', 'NH3'), (0,)), (('C2H2O',), (0, 1)), (('C2H2O', 'NH3'), (-1, 0)),
                         (('CO2',), (-1,)))
HYDROGEN_ATOMS = {'+H', '-H'}
HYDROGEN_SHIFTS = ('+H', '-H', '2+H', '2-H')
# Ions without a residue, as m/z of the singly charged ion with what a glycan needs to give it and labels for domon_costello_to_mpl/_html: the inorganic
# anions of sulfates and phosphates (negative mode; SO3- is a radical anion), and in positive mode the protonated label of a reductively aminated glycan
# (its net reducing-end formula, glycowork's, lacks the label's oxygen) and the diethylmethyleneammonium ion of procainamide's side chain
SMALL_IONS = {'HSO4': (calculate_adduct_mass('HSO4') + ELECTRON_MASS, 'S', r'$HSO_4^-$', 'HSO<sub>4</sub><sup>-</sup>'),
              'SO3': (calculate_adduct_mass('SO3') + ELECTRON_MASS, 'S', r'$SO_3^{-\bullet}$', 'SO<sub>3</sub><sup>-&bull;</sup>'),
              'H2PO4': (calculate_adduct_mass('H2PO4') + ELECTRON_MASS, 'P', r'$H_2PO_4^-$', 'H<sub>2</sub>PO<sub>4</sub><sup>-</sup>'),
              'PO3': (calculate_adduct_mass('PO3') + ELECTRON_MASS, 'P', r'$PO_3^-$', 'PO<sub>3</sub><sup>-</sup>')} | {
    label: (calculate_adduct_mass(modification_formula_dict[label] + 'O') + PROTON_MASS, label, rf'$[{label}+H]^+$', f'[{label}+H]<sup>+</sup>')
    for label in ('2AA', '2AB', 'procainamide')} | {
    'C5H12N': (calculate_adduct_mass('C5H12N') - ELECTRON_MASS, 'procainamide', r'$C_5H_{12}N^+$', 'C<sub>5</sub>H<sub>12</sub>N<sup>+</sup>')}
# Ions of the open chain a reductive amination label makes of the reducing-end residue, in negative mode, where the label holds the charge: the label with
# carbons C1 to Ck of that residue (^{0,k}X_0, a bond of the chain cleaved), as such or after losing water, hydrogen and, from a HexNAc, acetamide or acetic
# acid, as {name: (k, HexNAc only)}, the losses written in the name. These are the ones 2AB N-glycans give in >= 25% of their HCD spectra against <= 2% for
# m/z-shifted decoys (GPST000651: 6.5 of them per spectrum against 0.08, 173.07 ^{0,3}X_0 - C2H5NO - H2O in 89%); an alditol gives them at chance (0 in the 91
# timsTOF N-glycan spectra whose top 100 peaks reach their m/z, ion traps cut them off).
# Like other X cross-rings, disable_X_cross_rings strips them, so the pipeline's annotation scores do not count them: they are the same for every candidate
# structure, and counting them only loosened the score gates (GPST000651 F1 0.761 -> 0.749, 4 more FP rows of the database fallback). They are in SMALL_IONS
# for their names, without an m/z, which chain_ions computes
CHAIN_IONS = {'02X0': (2, False), '02X0-H2O': (2, False), '02X0-H2': (2, False), '02X0-C2H5NO': (2, True), '02X0-C2H4O2': (2, True), '03X0-H2O': (3, False),
              '03X0-C2H5NO-H2O': (3, True), '04X0': (4, False), '04X0-H2O': (4, False), '04X0-2H2O': (4, False), '04X0-C2H5NO-H2O': (4, True),
              '04X0-C2H5NO-2H2O': (4, True), '04X0-C2H5NO-H2O-H2': (4, True)}
AMINATED_LABELS = ('2AA', '2AB', 'procainamide', 'PA')
SMALL_IONS |= {name: (None, 'chain', rf'$^{{0,{k}}}X_0' + ''.join(' - ' + re.sub(r'(?<=[A-Za-z])(\d+)', r'_{\1}', x) for x in name.split('-')[1:]) + '$',
                      f'<sup>0,{k}</sup>X<sub>0</sub>' + ''.join(' - ' + re.sub(r'(?<=[A-Za-z])(\d+)', r'<sub>\1</sub>', x) for x in name.split('-')[1:]))
               for name, (k, _) in CHAIN_IONS.items()}


def acidic_groups(label):
    """How many acidic groups (carboxyl, sulfate, phosphate) a monosaccharide or a composition key has, each of which can hold a metal cation as a salt"""
    carboxyl = label not in CARBOXYL_LABELS and (label.startswith(('Neu', 'Kdn', 'Kdo')) or map_to_basic(label, obfuscate_ptm = False).startswith('HexA'))
    return carboxyl + len(re.findall(r'(?:\d|O|N|^)(?:S|P)(?![a-z])', label))


def small_ions(residues, charge, mass_tag):
    """The ions without a residue (SMALL_IONS) a glycan gives, as {token: m/z}: in negative mode those of its sulfates and phosphates, in positive
    mode the reporter ions of its reducing-end label, recognized by mass_tag\n
    | Arguments:
    | :-
    | residues (iterable): the glycan's monosaccharide labels, or the keys of a composition
    | charge (int): the precursor's charge, whose sign sets the ion mode
    | mass_tag (float): mass of the glycan label or reducing end modification, as in CandyCrumbs\n
    | Returns:
    | :-
    | Returns a dict of fragment name ('I_HSO4') : m/z
    """
    if charge < 0:
        # a sulfate follows a position, O or the amine of a hexosamine (GlcNS), a phosphate a position or O (Man6P; not the P of Pen, Pse, Pyr); a composition has S/P
        present = {s for s, pattern in (('S', r'(\d|O|N)S(?![a-z])'), ('P', r'(\d|O)P(?![a-z])')) if any(r == s or re.search(pattern, r) for r in residues)}
    else:
        present = {label for label in ('2AA', '2AB', 'procainamide') if mass_tag is not None and
                   abs(mass_tag - calculate_adduct_mass(modification_formula_dict[label])) < 0.01}
    return {f'I_{k}': v[0] for k, v in SMALL_IONS.items() if v[1] in present}


def chain_ions(root_label, charge, mass_tag, sample_prep = 'underivatized'):
    """The open-chain ions of a labelled reducing end (CHAIN_IONS) a glycan gives, as {token: m/z}: in negative mode, for an underivatized Hex or HexNAc
    carrying a reductive amination label, recognized by mass_tag\n
    | Arguments:
    | :-
    | root_label (string): the reducing-end monosaccharide
    | charge (int): the precursor's charge, whose sign sets the ion mode
    | mass_tag (float): mass of the glycan label or reducing end modification, as in CandyCrumbs
    | sample_prep (string): underivatized/permethylated/peracetylated\n
    | Returns:
    | :-
    | Returns a dict of fragment name ('I_02X0') : m/z
    """
    base = map_to_basic(root_label, obfuscate_ptm = False)
    if charge > 0 or sample_prep != 'underivatized' or base not in ('Hex', 'HexNAc') or mass_tag is None or not any(
            abs(mass_tag - calculate_adduct_mass(modification_formula_dict[x])) < 0.01 for x in AMINATED_LABELS):
        return {}
    # each carbon of the chain as CH2O, the N-acetylated C2 of a HexNAc as C3H5NO; the label with C1 alone is the label plus CH2O
    carbons = [calculate_adduct_mass('C3H5NO') if base == 'HexNAc' and i == 1 else calculate_adduct_mass('CH2O') for i in range(6)]
    return {f'I_{name}': mass_tag + sum(carbons[:k]) - sum(int(n or 1) * calculate_adduct_mass(f) for n, f in (re.fullmatch(r'(\d*)(.+)', x).groups()
                                                                                                         for x in name.split('-')[1:])) - PROTON_MASS
            for name, (k, hexnac) in CHAIN_IONS.items() if base == 'HexNAc' or not hexnac}


def update_global_mods(subg, global_mods, special_residues):
    """Returns the valid list of global modifications for a given subgraph\n
    | Arguments:
    | :-
    | subg (networkx_object): a subgraph
    | global_mods (list): global modifications of the whole glycan, from get_initial_global_mods
    | special_residues (list): residues of the glycan with losses of their own (CO2 of acids, SO4 of S, PO4 of P), from get_initial_global_mods\n
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
    # residue by residue (the joined labels of Glc(a1-4)Ara contain GlcA), with every uronic acid (IdoA, GalA) as HexA and only phosphate
    # substituents (Man6P, GlcNAcOP) as P, not the P of Pen, Pse, Pyr or Prop; a derivatized carboxyl (derivatize_carboxyls) loses no CO2
    present_specials = [x for x in special_residues if any(re.search(r'(\d|O)P(?![a-z])', v) if x == 'P' else x in v or (
        x == 'HexA' and map_to_basic(v, obfuscate_ptm = False).startswith(x)) for v in all_labels if len(v) > 1 and v not in CARBOXYL_LABELS)]
    if not present_specials:
        return subg_global_mods
    own = [mod for mod, present in (('CO2', any(k in present_specials for k in ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn'])),
                                    ('SO4', 'S' in present_specials), ('PO4', 'P' in present_specials)) if present]
    # with hydrogen shifts (radical activation), a residue's own loss can come with one too
    return subg_global_mods + own + [combine_global_mods([mod] + parse_global_mod(s)) for mod in own for s in HYDROGEN_SHIFTS if s in global_mods]


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
    # a hydrogen shift alone (M_+H) is no cleavage
    c = 1 if global_mod is not None and not set(parse_global_mod(global_mod)) <= HYDROGEN_ATOMS else 0
    c += sum(1 for k in node_mod[0] if k in A_cross_rings or k in X_cross_rings)
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
    | Returns an array of both the input masses and the other masses at which to search to assign multiply charged fragments to the initial masses
    """
    fragment_masses = np.asarray(fragment_masses, dtype = float)
    return np.concatenate([fragment_masses] + [fragment_masses * z - (z - 1) * PROTON_MASS * np.sign(charge) for z in range(2, abs(charge) + 1)])


def annotate_subgraph(node_data, succ, graph_attributes, mono_mods, atom_mods, global_mod, terminals):
    """Builds a fragment: a graph of the nodes and edges of a subgraph with the node, atom, and global modification attributes describing the cleavages\n
    | Arguments:
    | :-
    | node_data (dict): node : attribute dict of the subgraph, in its node order
    | succ (dict): node : {successor: edge attribute dict} of the subgraph
    | graph_attributes (dict): graph attributes of the subgraph
    | mono_mods (list): cleavage type (cross-ring or residue) at each terminal node
    | atom_mods (list): atom dictionary at each terminal node
    | global_mod (string): the chemical species globally lost or gained by the graph, or None
    | terminals (list): node labels of the terminal nodes, in the order of mono_mods and atom_mods\n
    | Returns:
    | :-
    | Returns a networkx graph with node attributes 'mod_labels', 'atomic_mod_dict' (terminals) and 'global_mod' (all nodes, one shared list)
    """
    # what copying the subgraph and set_node_attributes give (node, attribute and edge order, attribute dicts copied), built straight as dicts, as
    # networkx adds nodes and edges one by one; every candidate fragment is such a graph
    mods = dict(zip(terminals, zip(mono_mods, atom_mods)))
    if global_mod:
        global_mod = [global_mod]
        node_data = {n: {**d, 'mod_labels': mods[n][0], 'atomic_mod_dict': mods[n][1], 'global_mod': global_mod} if n in mods else {**d, 'global_mod': global_mod}
                     for n, d in node_data.items()}
    else:
        node_data = {n: {**d, 'mod_labels': mods[n][0], 'atomic_mod_dict': mods[n][1]} if n in mods else d.copy() for n, d in node_data.items()}
    adj, pred = {n: {} for n in node_data}, {n: {} for n in node_data}
    for u, nbrs in succ.items():
        for v, d in nbrs.items():
            adj[u][v] = pred[v][u] = d.copy()
    graph = nx.DiGraph()
    graph.graph.update(graph_attributes)
    graph._node, graph._adj, graph._pred = node_data, adj, pred
    return graph


class Fragment:
    """A candidate fragment of generate_atomic_frags: a node set of the glycan with the cleavage (cross-ring or residue) and atom dictionary at each
    terminal and a global modification. It iterates and tests membership like its nodes, and its graph (annotate_subgraph) is only built when asked
    for, as most candidates are only named and matched to a peak (building every one was half of an annotation's time)"""
    __slots__ = ('nodes', 'terminals', 'mono_mods', 'atom_mods', 'global_mod', '_parts', '_graph')

    def __init__(self, node_data, succ, graph_attributes, mono_mods, atom_mods, global_mod, terminals):
        # node_data and succ are those of the node set, shared by its fragments (annotate_subgraph copies them into each graph); nodes is node_data, as
        # node : attribute dict of the parent graph
        self.nodes, self.terminals, self.mono_mods, self.atom_mods, self.global_mod = node_data, terminals, mono_mods, atom_mods, global_mod
        self._parts, self._graph = (node_data, succ, graph_attributes), None

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, node):
        return node in self.nodes

    def number_of_nodes(self):
        return len(self.nodes)

    @property
    def graph(self):
        if self._graph is None:
            self._graph = annotate_subgraph(*self._parts, self.mono_mods, self.atom_mods, self.global_mod, self.terminals)
        return self._graph


# Fragment options of single terminals and global modifications of label sets, the node sets of a topology with few enough cut bonds, and what the
# precheck of a glycan's node sets takes before the wanted masses (see generate_atomic_frags), kept across calls
_TERMINAL_OPTIONS = {}
_NODE_SETS = {}
_PREPARED = {}
_GLYCAN_GRAPHS = {}


def generate_atomic_frags(nx_mono, global_mods, special_residues, allowed_X_cleavages, max_cleavages = 3,
                          fragment_masses = [], subgraphs = None,
                          threshold = 0.5, mass_tag = None, charge = -1, sample_prep = 'underivatized',
                          disable_A_cross_rings = False, threshold_ppm = None, peptide_ion_types = None):
    """Calculates the graph and mass of all possible fragments of the input\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): the original monosaccharide only graph
    | global_mods (list): global modifications of the whole glycan, from get_initial_global_mods
    | special_residues (list): residues of the glycan with losses of their own, from get_initial_global_mods
    | allowed_X_cleavages (list): X-type cross-rings allowed on terminal nodes
    | max_cleavages (int): maximum number of allowed concurrent fragmentations per mass; default:3
    | fragment_masses (list): all masses which are to be annotated with a fragment name; default:[] (every fragment is kept)
    | subgraphs (list): node sets or subgraphs of nx_mono to fragment; default:None (all connected subgraphs and nx_mono)
    | threshold (float): the range around the observed mass in which to constrain potential fragments; default:0.5
    | mass_tag (float): mass of the glycan label or reducing end modification; default:None (alditol, 2 H)
    | charge (int): the maximum possible charge on the fragments to be matched; default:-1
    | sample_prep (string): underivatized/permethylated/peracetylated
    | disable_A_cross_rings (bool or string): whether to strip out any A-type cross-rings, or 'non_reducing' for all but those of the reducing-terminal
    |                                         residue on fragments with a metal cation; default: False
    | threshold_ppm (float): relative tolerance in ppm on top of threshold, as match_fragment_properties applies it (a fragment it could never match
    |                        is not made); default:None
    | peptide_ion_types (set): the peptide backbone ion types to make, as CandyCrumbs names only those of its fragmentation method; default:None (all)\n
    | Returns:
    | :-
    | Returns a dict of lists of Fragments by their mass
    """
    if mass_tag is None:
        mass_tag = 2 * HYDROGEN_MASS
    charge_masses = extend_masses(fragment_masses, charge)
    sorted_charge_masses = sorted(charge_masses)
    unfiltered = not len(charge_masses)
    threshold = abs(threshold)
    true_root_node = next(v for v, d in nx_mono.out_degree() if d == 0)
    all_other_terminals = {node for node in nx_mono.nodes() if nx_mono.degree()[node] < 2 or node == true_root_node}
    nx_edge_dict = {(node[0], node[1]): node[2] for node in nx_mono.edges(data = True)}
    node_dict = nx.get_node_attributes(nx_mono, 'string_labels')
    node_dict_basic = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in node_dict.items()}
    subgraph_fragments = {}
    present_global_masses = [global_mod_mass(x, -PROTON_MASS if charge < 0 else PROTON_MASS)
                             for x in global_mods] + [0.0]
    max_global_mass = max(present_global_masses)
    min_global_mass = min(present_global_masses)
    # degree lookups on subgraph views are slow, so terminals are found via neighbor sets of the parent graph
    neighbor_dict = {v: set(nx_mono.predecessors(v)) | set(nx_mono.successors(v)) for v in nx_mono.nodes}
    succ_sets = {v: set(nx_mono.successors(v)) for v in nx_mono.nodes}
    deriv_mass = DERIVATIZATION_MASSES.get(sample_prep, 0)
    full_masses = {k: mono_attributes[v]['mass'][v] + get_derivatization_count(v, v, sample_prep) * deriv_mass for k, v in
                   node_dict_basic.items()}
    # a derivatized glycosidic cleavage loses one more group, which the lower bound has to allow for
    min_bond_mass = min(bond_masses.values()) - deriv_mass - max(W_SIDE_CHAIN_LOSSES.values())
    peptide_graph = isinstance(true_root_node, str)
    check_cuts = subgraphs is not None
    node_list = list(nx_mono)
    membership = None
    if subgraphs is None:
        # Which node sets have few enough cut bonds depends only on the topology, which recurs across structures (isomers, the placements supporting_ions
        # compares, a glycopeptide's spectra), so a glycan's or glycopeptide's list (and which nodes each holds, as a matrix for the precheck) is kept
        # across calls (same sets, same order)
        key = (tuple(nx_mono), tuple(nx_mono.edges()), max_cleavages)
        cached = _NODE_SETS.get(key)
        if cached is None:
            sets = enumerate_subgraphs(nx_mono)
            # which nodes each set holds, as a matrix: a node is a cut terminal if one of its neighbors is outside the set
            pos = {v: j for j, v in enumerate(node_list)}
            member = np.zeros((len(sets), len(node_list)), dtype = bool)
            member[np.repeat(np.arange(len(sets)), [len(x) for x in sets]), [pos[v] for x in sets for v in x]] = True
            cuts = np.zeros(len(sets), dtype = int)
            for j, v in enumerate(node_list):
                if v not in all_other_terminals:
                    cuts += member[:, j] & ~member[:, [pos[u] for u in neighbor_dict[v]]].all(axis = 1)
            ok = np.flatnonzero(cuts <= max_cleavages)
            # the whole glycan (or glycopeptide) joins them as the set of all its nodes, which goes through every step like the graph itself did
            cached = [[sets[k] for k in ok.tolist()] + [set(nx_mono)], np.vstack([member[ok], np.ones((1, len(node_list)), dtype = bool)])]
            if len(_NODE_SETS) > 5000:
                _NODE_SETS.clear()
            _NODE_SETS[key] = cached
        subgraphs, membership = cached[0], cached
    # The options of a terminal (cross-ring and bond permutations, their masses and cleavage counts) depend only on its label, its bonds and which of them are
    # cut, whether it is the root of the node set or the reducing end, and the settings, and the global modifications of a node set only on its labels, so
    # they are kept across calls. With them, the fragment masses of all node sets are checked at once, in arrays, before any of them becomes a subgraph:
    # most node sets have no fragment near any wanted mass (~85% in supporting_ions and the pipeline's scoring). A node set's root is its first sink in
    # node order, as the full calculation takes it (a glycopeptide's set has two, the C-terminal residue of its peptide part and a glycan's reducing end).
    # The check sums in another order than the full calculation, so it allows 1e-6 more
    precheck = not unfiltered
    set_idx = np.array([i for i, s in enumerate(subgraphs) if isinstance(s, set)], dtype = int)
    todo, found, m_thresh = range(len(subgraphs)), {}, 1 if charge < 0 else 2
    if precheck and len(set_idx):
        mode_mass = -PROTON_MASS if charge < 0 else PROTON_MASS
        targets = np.sort(charge_masses)
        red_ends = {v: not any(nx_edge_dict[x]['bond_label'] == 'glycosite' for x in nx_mono.in_edges(v)) for v, d in nx_mono.nodes(data = True) if 'reducing_end' in d}
        settings = (tuple(allowed_X_cleavages), disable_A_cross_rings, sample_prep, mass_tag, None if peptide_ion_types is None else tuple(sorted(peptide_ion_types)))
        global_settings = (tuple(global_mods), tuple(special_residues), charge < 0)
        pos = {v: j for j, v in enumerate(node_list)}
        # Everything before the wanted masses depends only on the glycan, its topology's node sets and the settings, and the same structures come up for
        # many spectra (supporting_ions compares each with its placements for every spectrum), so it is kept across calls for a glycan's enumerated node sets
        prep_key = (key, tuple(node_dict.values()), tuple(d['bond_label'] for d in nx_edge_dict.values()), tuple(red_ends.items()), settings,
                    global_settings) if membership is not None else None
        prepared = _PREPARED.get(prep_key) if prep_key is not None else None
        if prepared is None:
            incident = {v: [] for v in nx_mono.nodes}
            for (u, v), d in nx_edge_dict.items():
                incident[u].append((v, d['bond_label'], True))
                incident[v].append((u, d['bond_label'], False))
            if len(_TERMINAL_OPTIONS) > 50000:
                _TERMINAL_OPTIONS.clear()
            # per node set and node: whether the node has a cut bond, whether it is the root of the set, and which of its bonds stay in the set (bit k for its
            # k-th bond), which only depend on the topology, so they are kept with its node sets
            if membership is not None and len(membership) > 2:
                member, cut, is_root, bonds = membership[1:]
            else:
                if membership is not None:
                    member = membership[1]
                else:
                    sets = [subgraphs[i] for i in set_idx]
                    member = np.zeros((len(sets), len(node_list)), dtype = bool)
                    member[np.repeat(np.arange(len(sets)), [len(x) for x in sets]), [pos[v] for x in sets for v in x]] = True
                cut = np.column_stack([member[:, j] & ~member[:, [pos[u] for u in neighbor_dict[v]]].all(axis = 1) for j, v in enumerate(node_list)])
                is_root = np.column_stack([member[:, j] & ~member[:, [pos[u] for u in succ_sets[v]]].any(axis = 1) for j, v in enumerate(node_list)])
                is_root &= np.cumsum(is_root, axis = 1) == 1
                bonds = np.zeros(member.shape, dtype = np.uint8 if max(map(len, incident.values())) <= 8 else np.int64)
                for j, v in enumerate(node_list):
                    for k, (u, _, _) in enumerate(incident[v]):
                        bonds[:, j] |= member[:, pos[u]].astype(bonds.dtype) << k
                if membership is not None:
                    membership[2:] = [cut, is_root, bonds]
            # a terminal has a cut bond, or is a leaf or the reducing end of the glycan
            other = np.array([v in all_other_terminals for v in node_list])
            terminal = cut | (member & other)
            # node sets with too many cleavages, which their nodes alone tell (enumerated ones are filtered above), are skipped, and ones with two reducing
            # ends are left to the full calculation
            allowed = (cut & ~other).sum(axis = 1) <= max_cleavages if check_cuts else np.ones(len(member), dtype = bool)
            two_reds = member[:, [pos[v] for v in red_ends]].sum(axis = 1) >= 2
            active = allowed & ~two_reds
            # each terminal's options, looked up once per node and kind of terminal (root or not, which of its bonds stay in the set)
            rows, cols = np.nonzero(terminal & active[:, np.newaxis])
            degree = np.array([len(incident[v]) for v in node_list])
            uniq, first, inverse = np.unique((cols.astype(np.int64) << 32) | (is_root[rows, cols].astype(np.int64) << degree[cols]) | bonds[rows, cols].astype(np.int64),
                                             return_index = True, return_inverse = True)
            opt_table, opt_ids = [], np.full(member.shape, -1)
            # a terminal's options are kept by its label, bonds (label, direction), whether it is the root (bit len(bonds)) and which bonds stay (bit k for the
            # k-th), whether it is the reducing end, and the settings
            signatures = {v: (node_dict[v], tuple((label, child) for _, label, child in incident[v]), red_ends.get(v, False), settings) for v in node_list}
            for code, f in zip(uniq.tolist(), first.tolist()):
                v = node_list[code >> 32]
                # read once with get, as another thread (the app's annotation) may clear the store in between
                key = (signatures[v], code & 0xFFFFFFFF)
                opt = _TERMINAL_OPTIONS.get(key)
                if opt is None:
                    rep = subgraphs[set_idx[rows[f]]]
                    view, root = nx_mono.subgraph(rep), next(u for u in node_list if u in rep and not succ_sets[u] & rep)
                    breaks = {b: l for b, l in get_broken_bonds(view, nx_edge_dict).items() if (b[0] if b[0] in rep else b[1]) == v}
                    mods = get_mono_mods_list(root, view, [v], [node_dict_basic[v]], nx_edge_dict, allowed_X_cleavages, disable_A_cross_rings = not red_ends.get(v, False)
                                              if disable_A_cross_rings == 'non_reducing' else disable_A_cross_rings)
                    mono_perms, atom_perms = generate_mod_permutations([v], [node_dict_basic[v]], mods, atom_mods_init(view, breaks, [v], [node_dict_basic[v]]),
                                                                       peptide_ion_types = peptide_ion_types)
                    mono_m, atom_m, _ = precalculate_mod_masses(mono_perms, atom_perms, [node_dict_basic[v]], [None], sample_prep = sample_prep, charge = charge)
                    bonus = red_ends.get(v, False)
                    vals = np.array(
                        preliminary_calculate_mass(mono_m, atom_m, [], [v], 0.0, bonus, v if bonus else None,
                                                   mass_tag, charge, mono_perms,
                                                   np.arange(len(mono_perms[0]))[:, np.newaxis],
                                                   sample_prep = sample_prep,
                                                   root_label = node_dict[v] if bonus else None)) - mode_mass
                    bonus = reducing_end_bonus(mono_perms[0], mass_tag, sample_prep = sample_prep, root_label = node_dict[v]).tolist() if bonus else None
                    # last, its residue, bond and reducing-end masses per option as one array, as the exact calculation sums them
                    opt = _TERMINAL_OPTIONS[key] = (vals, np.array(
                        [mod_count([[m], [a]], None) for m, a in zip(mono_perms[0], atom_perms[0])], dtype = int),
                                                    mono_perms[0], atom_perms[0], mono_m[0], atom_m[0],
                                                    vals.min(initial = np.inf), vals.max(initial = -np.inf), bonus,
                                                    np.array([mono_m[0], atom_m[0], bonus or [0.0] * len(vals)], dtype = float).reshape(3, -1))
                opt_table.append(opt)
            opt_ids[rows, cols] = inverse.ravel()
            # the global modifications of each node set, by the labels it holds
            labels = sorted(set(node_dict.values()))
            holds = np.zeros(len(member), dtype = object if len(labels) > 62 else np.int64)
            for k, label in enumerate(labels):
                holds += member[:, [pos[v] for v in node_list if node_dict[v] == label]].any(axis = 1).astype(holds.dtype) * (1 << k)
            uniq, first, gid = np.unique(holds, return_index = True, return_inverse = True)
            gid, global_table = gid.ravel(), []
            for present, f in zip(uniq.tolist(), first.tolist()):
                key = (frozenset(label for k, label in enumerate(labels) if present >> k & 1), global_settings)
                opt = _TERMINAL_OPTIONS.get(key)
                if opt is None:
                    gm = update_global_mods(nx_mono.subgraph(subgraphs[set_idx[f]]), global_mods, special_residues)
                    g_masses = [global_mod_mass(x, mode_mass = mode_mass) for x in gm[1:]]
                    opt = _TERMINAL_OPTIONS[key] = (gm, g_masses, np.array([0.0] + g_masses), np.array([mod_count([[], []], x) for x in gm], dtype = int),
                                                    np.array([bool(ADDUCT_GLOBAL_MODS & set(parse_global_mod(x))) for x in gm]), min([0.0] + g_masses), max([0.0] + g_masses))
                global_table.append(opt)
            # without a wanted mass between the lightest and heaviest combination of options, a set has no fragment to check one by one
            lens = np.array([len(o[0]) for o in opt_table] + [1])
            offsets = np.cumsum(lens) - lens
            flat_masses = np.concatenate([o[0] for o in opt_table] + [np.zeros(1)])
            flat_counts = np.concatenate([o[1] for o in opt_table] + [np.zeros(1, dtype = int)])
            lightest = np.array([o[6] for o in opt_table] + [0.0])[opt_ids].sum(axis = 1) + np.array([o[5] for o in global_table])[gid]
            heaviest = np.array([o[7] for o in opt_table] + [0.0])[opt_ids].sum(axis = 1) + np.array([o[6] for o in global_table])[gid]
            edges = np.array([[pos[u], pos[v]] for u, v in nx_edge_dict], dtype = int).reshape(-1, 2)
            base = (member & ~terminal) @ np.array([full_masses[v] for v in node_list]) + mode_mass - (
                (member[:, edges[:, 0]] & member[:, edges[:, 1]]).sum(axis = 1) * deriv_mass if deriv_mass else 0)
            if peptide_graph:
                # the options of each peptide terminal carry the water of the peptide's termini, which a fragment gains once
                base -= WATER_MASS * np.maximum((terminal & np.array([v.split('-')[0] == '0' for v in node_list])).sum(axis = 1) - 1, 0)
            caps = max_cleavages - np.array([o[3].min() for o in global_table])[gid]
            # sets of other labels mostly have the same global modifications, so they are checked together
            same = {}
            for o in global_table:
                same.setdefault(tuple(o[0]), o)
            group = np.array([list(same).index(tuple(o[0])) for o in global_table])[gid]
            flat_options = np.concatenate([o[9] for o in opt_table] + [np.zeros((3, 1))], axis = 1)
            # the global modifications of every set's labels, padded to the longest (the padding never kept)
            g_width = max(len(o[0]) for o in global_table)
            g_pad = np.zeros((len(global_table), g_width)), np.full((len(global_table), g_width), max_cleavages + 1), np.zeros((len(global_table), g_width), dtype = bool)
            for k, o in enumerate(global_table):
                g_pad[0][k, :len(o[0])], g_pad[1][k, :len(o[0])], g_pad[2][k, :len(o[0])] = o[2], o[3], o[4]
            prepared = (member, allowed, two_reds, active, opt_table, opt_ids, gid, global_table, lens, offsets, flat_masses, flat_counts, base, base + lightest,
                        base + heaviest, caps, same, group, flat_options, g_pad)
            if prep_key is not None:
                if len(_PREPARED) > 1000:
                    _PREPARED.clear()
                _PREPARED[prep_key] = prepared
        (member, allowed, two_reds, active, opt_table, opt_ids, gid, global_table, lens, offsets, flat_masses, flat_counts, base, lightest, heaviest, caps, same,
         group, flat_options, g_pad) = prepared
        first = np.searchsorted(targets, lightest - threshold - 2e-6, 'right')
        rows = np.flatnonzero(active & (first < len(targets)) & (np.append(targets, np.inf)[first] < heaviest + threshold + 2e-6))
        # every combination of options of these sets, with its index in the product of the set's terminal options in node order, built terminal by terminal
        # (the k-th terminal of a set in node order; a set with fewer terminals takes the empty option, the last); cleavage counts only add up, so
        # partial combinations over the cap are dropped as they appear
        slots = np.take_along_axis(opt_ids[rows], np.argsort(opt_ids[rows] < 0, axis = 1, kind = 'stable'), axis = 1)
        c_set, c_mass, c_count, c_index = np.arange(len(rows)), np.zeros(len(rows)), np.zeros(len(rows), dtype = int), np.zeros(len(rows), dtype = np.int64)
        for k in range(int((slots >= 0).sum(axis = 1).max(initial = 0))):
            o = slots[c_set, k]
            n = lens[o]
            src = np.repeat(np.arange(len(c_set)), n)
            local = np.arange(len(src)) - np.repeat(np.cumsum(n) - n, n)
            option = np.repeat(offsets[o], n) + local
            c_set, c_mass, c_count, c_index = c_set[src], c_mass[src] + flat_masses[option], c_count[src] + flat_counts[option], c_index[src] * n[src] + local
            ok = c_count <= caps[rows[c_set]]
            c_set, c_mass, c_count, c_index = c_set[ok], c_mass[ok], c_count[ok], c_index[ok]
        c_slot, c_set = c_set, rows[c_set]
        # the combinations with any global modification near a wanted mass are the only ones the full calculation has to do, as it would keep no other
        near = np.zeros(len(c_set), dtype = bool)
        bounded = np.append(targets, np.inf)
        for g, (_, _, g_offsets, g_counts, g_adduct, g_lightest, g_heaviest) in enumerate(same.values()):
            sel = np.flatnonzero(group[c_set] == g)
            if len(g_offsets) > 1:
                # first the combinations with a wanted mass between their lightest and heaviest global modification
                masses = c_mass[sel] + base[c_set[sel]]
                sel = sel[bounded[np.searchsorted(targets, masses + g_lightest - threshold - 2e-6, 'right')] < masses + g_heaviest + threshold + 2e-6]
            full = c_count[sel, np.newaxis] + g_counts
            masses = c_mass[sel, np.newaxis] + (base[c_set[sel], np.newaxis] + g_offsets)
            reach = threshold + 1e-6 if threshold_ppm is None else np.minimum(threshold, masses * threshold_ppm / 1e6 + 1e-5) + 1e-6
            near[sel] = ((full <= max_cleavages) & ~((full > m_thresh) & g_adduct) &
                         (bounded[np.searchsorted(targets, masses - reach, 'right')] < masses + reach)).any(axis = 1)
        hit_set, hit_slot, hit_index, hit_count = c_set[near], c_slot[near], c_index[near], c_count[near]
        found = {int(set_idx[row]): row for row in np.unique(hit_set).tolist()}
        # what the precheck left: sets with a combination near a wanted mass, and sets it does not cover (two reducing ends, a graph)
        todo = sorted(set(found) | {i for i, s in enumerate(subgraphs) if not isinstance(s, set)} | set(set_idx[allowed & two_reds].tolist()))
    if not unfiltered:
        # the exact check of check_masses (a mass is kept if the nearest wanted mass on either side is within threshold), with the sorted wanted masses
        # padded at both ends so that the nearest ones are two lookups
        padded = np.sort(charge_masses)
        padded = np.concatenate([padded[:1], padded, padded[-1:]])
    # first each node set's terminals, its node and successor dicts and, for the sets the precheck does not cover, its fragments; then the fragments of the
    # sets it found, all at once; then the fragments, set by set
    plan, exact = [], []
    for i in todo:
        subg = subgraphs[i]
        opts = None
        if isinstance(subg, set):
            if check_cuts and not precheck and sum(not neighbor_dict[x] <= subg and x not in all_other_terminals for x in subg) > max_cleavages:
                continue
            # the order a subgraph view of the set iterates its nodes in, which decides the order of terminals and fragments (networkx walks a set rebuilt
            # from it if it holds less than half of the graph's nodes, else the graph); the view itself is slow to make and never needed. A view of few nodes
            # iterates them in set order, which for the string nodes of a glycopeptide follows PYTHONHASHSEED and decided between equally good fragments
            # (02X_5_Alpha or 02X_5_Beta), so those are walked in the order of the parent graph
            nodes = [v for v in nx_mono if v in subg] if peptide_graph or 2 * len(subg) >= len(nx_mono) else list(set(v for v in subg))
            if i in found:
                row = found[i]
                terms = [x for x in nodes if not neighbor_dict[x] <= subg]
                terms += [x for x in nodes if x in all_other_terminals and x not in terms]
                opts = {t: opt_table[opt_ids[row, pos[t]]] for t in terms}
                reds, root = [v for v in red_ends if v in subg], next(v for v in nodes if not succ_sets[v] & subg)
        else:
            nodes = [v for v in nx_mono if v in subg] if peptide_graph else subg
        terminals = terms if opts else get_terminals(neighbor_dict, nodes)
        new_terminals = [x for x in terminals if x not in all_other_terminals]
        if len(new_terminals) > max_cleavages:
            continue
        if not opts:
            terminals = terminals + [x for x in nodes if x in all_other_terminals and x not in terminals]
        # every glycosidic bond takes one derivatizable group of the residue it is attached to
        inner_mass = sum(full_masses[m] for m in nodes if m not in terminals) - (
            (sum(len(succ_sets[v] & subg) for v in subg) if isinstance(subg, set) else subg.number_of_edges()) * deriv_mass if deriv_mass else 0)
        max_graph_mass = inner_mass + sum(full_masses[m] for m in terminals) + WATER_MASS * len(terminals)
        max_graph_mass += max_global_mass + max(mass_tag, 0) + PROTON_MASS + 2 * deriv_mass
        min_terminal_mass = sum(min(mono_attributes[node_dict_basic[m]]['mass'].values()) + min_bond_mass for m in terminals)
        min_graph_mass = inner_mass + min_terminal_mass + min_global_mass
        avg_graph_mass = (min_graph_mass + max_graph_mass) / 2
        graph_mass_thresh = (max_graph_mass - min_graph_mass) / 2 + threshold
        if not unfiltered:
            lo = bisect.bisect_left(sorted_charge_masses, avg_graph_mass - graph_mass_thresh)
            if lo >= len(sorted_charge_masses) or sorted_charge_masses[lo] > avg_graph_mass + graph_mass_thresh:
                continue
        # the surviving node set as the node and successor dicts a copy of its subgraph view has (its node order, each node's out-edges in parent order,
        # attribute dicts copied), built straight from the parent graph's dicts, as networkx adds nodes and edges one by one; glycopeptide subgraphs
        # are copied in parent order, as the first reducing end temporary_root_calc_func finds depends on it
        node_data = {v: nx_mono._node[v].copy() for v in nodes}
        succ = {u: {v: d.copy() for v, d in nx_mono._succ[u].items() if v in node_data} for u in node_data}
        if opts:
            # the options the precheck took, whose combinations near a wanted mass are checked below
            bonus_root_mass, root_node = bool(reds) and red_ends[reds[0]], root
            bonus_root_node = reds[0] if bonus_root_mass else None
            mono_mod_perms, atom_dict_perms = [opts[t][2] for t in terminals], [opts[t][3] for t in terminals]
            fragments = []
            exact.append((fragments, row, terminals, inner_mass, bonus_root_mass, bonus_root_node, opts))
        else:
            # a real graph makes every later degree/edge query far cheaper than the view
            subg = nx.DiGraph()
            subg.graph.update(nx_mono.graph)
            subg._node, subg._adj, subg._pred = node_data, succ, {v: {u: succ[u][v] for u in node_data if v in succ[u]} for v in node_data}
            terminal_labels = [node_dict_basic[x] for x in terminals]
            bonus_root_mass, bonus_root_node = temporary_root_calc_func(subg, nx_mono)
            subg_global_mods = update_global_mods(subg, global_mods, special_residues)
            present_breakages = get_broken_bonds(subg, nx_edge_dict)
            # glycopeptide subgraphs have two sinks (C-terminal residue, glycan reducing end); pick in parent graph order
            # like true_root_node, as the subgraph view iterates string node labels in hash-randomized set order
            root_node = next(v for v in nx_mono if v in subg and subg.out_degree(v) == 0)
            atomic_mod_dict_subg = atom_mods_init(subg, present_breakages, terminals, terminal_labels)
            mono_mods_list = get_mono_mods_list(root_node, subg, terminals, terminal_labels, nx_edge_dict, allowed_X_cleavages,
                                                not (bonus_root_mass and root_node == bonus_root_node) if disable_A_cross_rings == 'non_reducing' else disable_A_cross_rings)
            mono_mod_perms, atom_dict_perms = generate_mod_permutations(terminals, terminal_labels, mono_mods_list,
                                                                        atomic_mod_dict_subg, peptide_ion_types = peptide_ion_types)
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
            row_counts, has_adduct = inner_counts[inner_idx], np.tile([bool(ADDUCT_GLOBAL_MODS & set(parse_global_mod(x))) for x in subg_global_mods], inner_idx.size)
            perm_indices = vectorized_lazy_product_indices(mono_mod_perms, inner_idx)[:, ::-1]
            initial_masses = np.array(
                preliminary_calculate_mass(mono_masses, atom_masses, global_masses, terminals, inner_mass, bonus_root_mass,
                                           bonus_root_node, mass_tag, charge, mono_mod_perms, perm_indices,
                                           sample_prep = sample_prep, root_label = node_dict.get(bonus_root_node) if bonus_root_mass else None))
            counts = (row_counts[:, np.newaxis] + global_counts).ravel()
            keep = (counts <= max_cleavages) & ~((counts > m_thresh) & has_adduct)
            if not unfiltered:
                idx = np.searchsorted(padded[1:-1], initial_masses)
                # with threshold_ppm, within it too (with the 5-decimal rounding of the fragment masses match_fragment_properties compares)
                keep &= np.minimum(np.abs(initial_masses - padded[idx]), np.abs(initial_masses - padded[idx + 1])) < (
                    threshold if threshold_ppm is None else np.minimum(threshold, initial_masses * threshold_ppm / 1e6 + 1e-5))
            if not keep.any():
                continue
            # each kept fragment's option per terminal, its global modification, mass and cleavage count
            rows, mods = np.divmod(np.flatnonzero(keep), len(subg_global_mods))
            fragments = zip(perm_indices[rows].tolist(), [subg_global_mods[k] for k in mods.tolist()], initial_masses[keep].tolist(), counts[keep].tolist())
        plan.append((node_data, succ, terminals, root_node, bonus_root_mass, bonus_root_node, mono_mod_perms, atom_dict_perms, fragments))
    if exact:
        # The combinations the precheck found near a wanted mass, through the full calculation's exact masses, cleavage counts and check, all at once: each
        # set's terminals in the order the full calculation takes them (cut ones, then the others, in the order of the set's view), the combinations by
        # their option per terminal in that order (decoded from their index over the terminals in node order), and their masses summed as
        # preliminary_calculate_mass sums them (terminal by terminal, so up to 7 one after the other; with 8 or more, numpy sums pairwise, so those go to it)
        entry = np.full(len(member), -1)
        entry[[e[1] for e in exact]] = np.arange(len(exact))
        sel = np.flatnonzero(entry[hit_set] >= 0)
        set_of, slot_ids, count = entry[hit_set[sel]], slots[hit_slot[sel]], hit_count[sel]
        width = max(len(e[2]) for e in exact)
        slot_ids = slot_ids[:, :width]
        digits, index = np.zeros(slot_ids.shape, dtype = np.int64), hit_index[sel]
        for k in range(width - 1, -1, -1):
            index, digits[:, k] = np.divmod(index, lens[slot_ids[:, k]])
        # per set, the terminal in node order of each of its terminals (the empty option, past its last, for the rest), and the reducing end's position
        order = np.full((len(exact), width + 1), width)
        root_at = np.full(len(exact), width)
        for k, (_, _, terminals, _, bonus_root_mass, bonus_root_node, _) in enumerate(exact):
            by_node = sorted(pos[t] for t in terminals)
            order[k, :len(terminals)] = [by_node.index(pos[t]) for t in terminals]
            if bonus_root_mass:
                root_at[k] = terminals.index(bonus_root_node)
        order = order[set_of]
        digits = np.take_along_axis(np.hstack([digits, np.zeros((len(sel), 1), dtype = np.int64)]), order, axis = 1)
        option = offsets[np.take_along_axis(np.hstack([slot_ids, np.full((len(sel), 1), -1)]), order, axis = 1)] + digits
        # sets in their order, a set's combinations as the full calculation sorts them
        rank = np.lexsort([count] + [digits[:, k] for k in range(width - 1, -1, -1)] + [set_of])
        set_of, count, digits, option = set_of[rank], count[rank], digits[rank], option[rank]
        flat_mono, flat_atom, flat_bonus = flat_options
        mono, atom = np.zeros(len(sel)), np.zeros(len(sel))
        for k in range(width + 1):
            mono = mono + flat_mono[option[:, k]]
        for k in range(width + 1):
            atom = atom + flat_atom[option[:, k]]
        # a fragment with a peptide terminal gains the water of the peptide's termini once, as preliminary_calculate_mass adds it
        constant = np.array([e[3] + mode_mass + (WATER_MASS if peptide_graph and any(t.split('-')[0] == '0' for t in e[2]) else 0) for e in exact])
        masses = constant[set_of] + mono + atom + flat_bonus[option[np.arange(len(sel)), root_at[set_of]]]
        for k, (_, _, terminals, inner_mass, bonus_root_mass, bonus_root_node, opts) in enumerate(exact):
            if len(terminals) > 7:
                rows = np.flatnonzero(set_of == k)
                masses[rows] = preliminary_calculate_mass([opts[t][4] for t in terminals], [opts[t][5] for t in terminals], [], terminals, inner_mass, bonus_root_mass,
                                                          bonus_root_node, mass_tag, charge, [opts[t][2] for t in terminals], digits[rows, :len(terminals)],
                                                          sample_prep = sample_prep, root_label = node_dict.get(bonus_root_node) if bonus_root_mass else None)
        # each combination with each global modification of its set
        g_offsets, g_counts, g_adduct = g_pad
        set_gid = np.array([gid[e[1]] for e in exact])[set_of]
        masses, count = masses[:, np.newaxis] + g_offsets[set_gid], count[:, np.newaxis] + g_counts[set_gid]
        idx = np.searchsorted(padded[1:-1], masses)
        rows, g = np.nonzero((count <= max_cleavages) & ~((count > m_thresh) & g_adduct[set_gid]) &
                             (np.minimum(np.abs(masses - padded[idx]), np.abs(masses - padded[idx + 1])) < (
                                 threshold if threshold_ppm is None else np.minimum(threshold, masses * threshold_ppm / 1e6 + 1e-5))))
        digits = digits.tolist()
        for row, col, mass, n_cuts, g_id in zip(rows.tolist(), g.tolist(), masses[rows, g].tolist(), count[rows, g].tolist(), set_gid[rows].tolist()):
            fragments, _, terminals = exact[set_of[row]][:3]
            fragments.append((tuple(digits[row][:len(terminals)]), global_table[g_id][0][col], mass, n_cuts))
    kept_masses, kept_fragments = [], []
    for node_data, succ, terminals, root_node, bonus_root_mass, bonus_root_node, mono_mod_perms, atom_dict_perms, fragments in plan:
        # an alditol (or reductively aminated) reducing end has no ring, so its cross-ring only stands in for a cleavage of the open chain on its own;
        # stacked onto another cleavage it read the Neu5Ac-HexNAc B ion (493) as Neu5Ac on the reduced GalNAc (1,5A + 1,5X, 1,3A - C2H4O2)
        root_pos = terminals.index(root_node) if mass_tag and root_node == true_root_node else None
        # with disable_A_cross_rings 'non_reducing', the A cross-rings of the reducing-terminal residue need a metal cation, which opens that ring (A_R ions)
        metal_pos = terminals.index(root_node) if disable_A_cross_rings == 'non_reducing' and bonus_root_mass and root_node == bonus_root_node else None
        for option, global_mod, mass, count in fragments:
            mono_mods = [perms[k] for perms, k in zip(mono_mod_perms, option)]
            if root_pos is not None and count > 1 and mono_mods[root_pos] in A_cross_rings:
                continue
            if metal_pos is not None and mono_mods[metal_pos] in A_cross_rings and not (global_mod and METAL_GLOBAL_MODS & set(parse_global_mod(global_mod))):
                continue
            kept_masses.append(mass)
            kept_fragments.append(Fragment(node_data, succ, nx_mono.graph, mono_mods, [perms[k] for perms, k in zip(atom_dict_perms, option)], global_mod, terminals))
    # the fragments by their mass rounded to 5 decimals, all rounded at once
    for mass, fragment in zip(np.round(np.array(kept_masses), 5), kept_fragments):
        subgraph_fragments.setdefault(mass, []).append(fragment)
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
    og_root = next(v for v, d in nx_mono.out_degree() if d == 0)
    og_leaves = set(v for v, d in nx_mono.in_degree() if d == 0)
    leaf_chains = [nx.shortest_path(nx_mono, source = og_leaf) for og_leaf in og_leaves]
    basic = {n: map_to_basic(v, obfuscate_ptm = False) for n, v in node_dict.items()}
    main_chains = sorted([branch_path[og_root] for branch_path in leaf_chains],
                         key = lambda x: sum(mono_attributes[basic[n]]['mass'][basic[n]] for n in x), reverse = True)
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
            global_mod = cut.partition('_')[2] or None
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
        cut_rank, cut_chain = next((rank, chain) for rank, chain in chain_rank if cut[1] in chain)
        if cut_type[-1] in {'B', 'C', 'A'}:
            cut_number = cut_chain.index(cut[1]) + 1
        elif cut_type[-1] in {'Y', 'Z'}:
            cut_number = cut_chain[::-1].index(cut[2]) + 1
        elif cut_type[-1] in {'X'}:
            cut_number = cut_chain[::-1].index(cut[1]) + 1
        dc_cuts.append(f"{cut_type}_{cut_number}_{cut_rank}")
    if global_mods:
        dc_cuts.append(f"M_{next(iter(global_mods.values()))[0]}")
    return dc_cuts or ['M']


def find_main_chain(subgraph, leaves, root_node):
    """Calculates the main chain of a subgraph\n
    | Arguments:
    | :-
    | subgraph (networkx_object): a subgraph
    | leaves (list): integer labels of leaf nodes
    | root_node (int): integer label of the root node\n
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
        if len(set(step)) > 1:
            bond_nums = [bond_dict[e][-1] for j in step for e in subgraph.edges if j == e[0]]
            main_chains = [main_chains[k] for k, x in enumerate(bond_nums) if x == min(bond_nums)]
    return main_chains[0]


def subgraph_to_label_skeleton(sub_g):
    """Breaks up a graph object into the skeleton of the IUPAC condensed nomenclature\n
    | Arguments:
    | :-
    | sub_g (networkx_object): a graph or subgraph of monosaccharides\n
    | Returns:
    | :-
    | Returns a list of integer node labels and branching brackets in the same order as the IUPAC condensed string
    """
    if len(sub_g.nodes) == 1:
        return [str(next(iter(sub_g.nodes)))]
    root_node = next(k for k, v in sub_g.out_degree if v == 0)
    leaves = [v for v, d in sub_g.in_degree() if d == 0]
    main_chain = [str(i) for i in find_main_chain(sub_g, leaves, root_node)]
    n_skelly = main_chain
    while {str(k) for k in sub_g.nodes} != {k for k in n_skelly if k.isnumeric()}:
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
    | sub_g (networkx_object): a graph or subgraph of monosaccharides\n
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
    | sub_g (networkx_object): a graph or subgraph of monosaccharides\n
    | Returns:
    | :-
    | Returns an IUPAC condensed representation of the input graph, in the case of fragment graphs it returns the closest canonical string representation
    """
    return label_skeleton_to_string(subgraph_to_label_skeleton(sub_g), sub_g)


def domon_costello_to_fragIUPAC(glycan_string, fragment):
    """Converts a glycan string and a Domon-Costello fragment name into a fragmented version of the original string\n
    | Arguments:
    | :-
    | glycan_string (string): glycan in IUPAC-condensed format
    | fragment (list): underscore separated string form of Domon-Costello e.g(['Y_1_Alpha'])\n
    | Returns:
    | :-
    | Returns the fragmented glycan in a version of IUPAC condensed which is GlycoDraw compatible
    """
    mono_graph = glycan_to_graph_monos(glycan_string)
    nx_mono = mono_graph_to_nx(mono_graph, directed = True)
    chain_rank = dict(rank_chains(nx_mono))
    skelly_dict, post_mono, global_mod = domon_costello_to_node_labels(fragment, chain_rank)
    excluded_nodes = set()
    for k, v in skelly_dict.items():
        if v[-1] == 'A':
            excluded_nodes.update(nx_mono.nodes() ^ nx.ancestors(nx_mono, k).union({k}))
            # as in get_fragment_mass, an A-type cross-ring takes any branch attached to a ring atom it does not retain
            retained_atoms = mono_attributes[map_to_basic(nx_mono.nodes[k]['string_labels'], obfuscate_ptm = False)]['atoms'][v]
            for child, _ in nx_mono.in_edges(k):
                if (pos := nx_mono.edges[child, k]['bond_label'][-1]).isdigit() and int(pos) not in retained_atoms:
                    excluded_nodes.update(nx.ancestors(nx_mono, child) | {child})
        elif v[-1] in {'B', 'C'}:
            excluded_nodes.update(nx_mono.nodes() ^ nx.ancestors(nx_mono, post_mono).union({post_mono, k}))
        elif v[-1] == 'X':
            # an X cross-ring keeps the children on the ring atoms it retains, unless a cleavage of their own takes them
            retained_atoms = mono_attributes[map_to_basic(nx_mono.nodes[k]['string_labels'], obfuscate_ptm = False)]['atoms'][v]
            for child, _ in nx_mono.in_edges(k):
                if not ((pos := nx_mono.edges[child, k]['bond_label'][-1]).isdigit() and int(pos) in retained_atoms):
                    excluded_nodes.update(nx.ancestors(nx_mono, child) | {child})
        elif v[-1] in {'Y', 'Z'}:
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
        # as in domon_costello_to_mpl
        terms = ' '.join(f"{sign} {term}" for sign, term in global_mod_terms(global_mod))
        full_skelly = '{' + re.sub(r'(?<=[A-Za-z])\d+', lambda m: ''.join(chr(0x2080 + int(c)) for c in m.group()), terms) + '}' + full_skelly
    return full_skelly


def global_mod_terms(global_mod):
    """Signed terms of a global modification token for fragment labels: adducts and hydrogen atoms with a + are gains, everything else a loss, a
    leading count (2H2O, 2+Na) stays a multiplier, combinations are split at |, and an odd hydrogen shift marks a radical ('+H|H2O' -> [('+', 'H•'),
    ('-', 'H2O')])"""
    terms = []
    for chunk in global_mod.split('|'):
        count, sign, name = re.fullmatch(r'(\d*)([+-]?)(.+)', chunk).groups()
        terms.append(('+' if sign == '+' else '-', count + name + ('\u2022' if name == 'H' and int(count or 1) % 2 else '')))
    return terms


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
        if len(html_nom_parts) == 2 and html_nom_parts[0] in N_TERM_IONS | C_TERM_IONS:
            html_nom = f"{html_nom_parts[0]}<sub>{html_nom_parts[1]}</sub>"
        elif len(html_nom_parts) == 2 and html_nom_parts[0] == 'I':
            html_nom = SMALL_IONS[html_nom_parts[1]][3]
        elif len(html_nom_parts) >= 2 and html_nom_parts[0] == 'M':
            # as in domon_costello_to_mpl
            terms = ''.join(f" {sign} {term}" for sign, term in global_mod_terms('_'.join(html_nom_parts[1:])))
            html_nom = 'M' + re.sub(r'(?<=[A-Za-z])(\d+)', r'<sub>\1</sub>', terms).replace('\u2022', '&bull;')
        html_name.append(html_nom)
    return ", ".join(html_name)


def subgraphs_to_domon_costello(nx_mono, subgs, chain_rank = None):
    """Converts the subgraphs of a given graph object into their canonical Domon & Costello fragment names\n
    | Arguments:
    | :-
    | nx_mono (networkx_object): the original monosaccharide only graph
    | subgs (list): a list of Fragments (generate_atomic_frags) or modified networkx subgraphs
    | chain_rank (list): rank_chains of nx_mono, if already at hand; default:None\n
    | Returns:
    | :-
    | Returns a nested list with one list of fragment labels for each subgraph
    """
    ion_names = []
    node_dict = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in nx_mono.nodes(data = 'string_labels') if v is not None}
    if chain_rank is None:
        chain_rank = list(rank_chains(nx_mono))
    in_children = {}
    for bonding_node, bonded_node, bond_label in nx_mono.edges(data = 'bond_label'):
        in_children.setdefault(bonded_node, []).append((bonding_node, bond_label.split('-')[-1]))
    # a Fragment's cleavages are named per terminal, once for each of its options (the atom dictionary of an option tells which bonds of the terminal are
    # cut, and so which children the fragment lacks)
    pieces = {}
    for subg in subgs:
        if isinstance(subg, Fragment):
            dc_cuts = [f"M_{subg.global_mod}"] if subg.global_mod else []
            for node, mod_label, atom_mods in zip(subg.terminals, subg.mono_mods, subg.atom_mods):
                key = (node, mod_label, id(atom_mods))
                piece = pieces.get(key)
                if piece is None:
                    cuts, cut_children = [], set()
                    for atom, atom_mod in atom_mods.items():
                        if atom_mod in {'bond', 'no_bond'}:
                            children = [(c, pos) for c, pos in in_children.get(node, []) if c not in cut_children and c not in subg.nodes]
                            cut_node = [c for c, pos in children if pos == str(atom)] or [c for c, pos in children if not pos.isdigit()]
                            if cut_node:
                                cut_children.add(cut_node[0])
                                cuts.append((atom_mod, cut_node[0], node))
                        if atom_mod in {'red_bond', 'red_no_bond'}:
                            cuts.append((atom_mod, node))
                    if node_dict.get(node) != mod_label:
                        cuts.append((mod_label, node))
                    piece = pieces[key] = node_labels_to_domon_costello(cuts, chain_rank) if cuts else []
                dc_cuts.extend(piece)
            ion_names.append(sorted(dc_cuts or ['M']))
            continue
        cuts, global_mods, mono_mod_dict, atomic_mod_dict = [], {}, {}, {}
        # plain reads of the node data in one pass, as nx.get_node_attributes (or even making a NodeDataView of a new graph) costs more than the naming itself
        for n, d in subg._node.items():
            if 'global_mod' in d:
                global_mods[n] = d['global_mod']
            if 'mod_labels' in d:
                mono_mod_dict[n] = d['mod_labels']
            if 'atomic_mod_dict' in d:
                atomic_mod_dict[n] = d['atomic_mod_dict']
        for node, atom_mods in atomic_mod_dict.items():
            cut_children = set()
            for atom, atom_mod in atom_mods.items():
                if atom_mod in {'bond', 'no_bond'}:
                    # only children outside the fragment were cut off; one at an ambiguous (3/4) or unknown (?) position sits wherever atom_mods_init put it
                    children = [(c, pos) for c, pos in in_children.get(node, []) if c not in cut_children and c not in subg]
                    cut_node = [c for c, pos in children if pos == str(atom)] or [c for c, pos in children if not pos.isdigit()]
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


def score_fragment_prior(dc_name, charge, chain_lengths = None):
    """Calculates an empirical prior score for a Domon-Costello fragment based on known fragmentation tendencies\n
    | Arguments:
    | :-
    | dc_name (list): list of Domon-Costello cleavage names making up a fragment
    | charge (int): charge state of the precursor ion
    | chain_lengths (dict): residues per chain rank of the glycan, which tell the A cross-rings of the reducing-terminal residue (A_R); default:None\n
    | Returns:
    | :-
    | Returns a float score where higher values indicate more commonly observed fragmentation patterns
    """
    if not dc_name:
        return 0.0
    mode = np.sign(charge)
    cleavage_weights = fragmentation_priors['cleavage_type'].get(mode, fragmentation_priors['cleavage_type'][-1])
    re_weights = fragmentation_priors['reducing_end_cross_ring']
    cation_adduct = any(c.startswith('M_') and any(a in c for a in re_weights['adducts']) for c in dc_name)
    score = 0.0
    n_cleavages = hydrogens = 0
    for cut in dc_name:
        parts = cut.split('_')
        if parts[0] == 'M':
            # a hydrogen shift scales the whole fragment's prior, everything else adds its own
            components = parse_global_mod('_'.join(parts[1:]))
            other = [x for x in components if x not in HYDROGEN_ATOMS]
            hydrogens += len(components) - len(other)
            if other or not components:
                score += math.prod(fragmentation_priors['global_mod'].get(x, 0.2) for x in other) if other else 0.2
            continue
        if parts[0] == 'I':
            score += fragmentation_priors['chain_ion' if parts[1] in CHAIN_IONS else 'small_ion']
            n_cleavages += 1
            continue
        cut_score = cleavage_weights.get(parts[0], 0.1)
        if cation_adduct and (parts[0] in A_cross_rings or parts[0] in X_cross_rings):
            # the reducing-terminal residue: X_1, or an A cross-ring numbered by the length of its chain (A_R)
            reducing = parts[1] == '1' if parts[0] in X_cross_rings else chain_lengths is not None and chain_lengths.get(parts[2]) == int(parts[1])
            cut_score *= re_weights['boost'] if reducing else re_weights['non_reducing_penalty']
        score += cut_score
        n_cleavages += 1
    if n_cleavages > 1:
        n_cross = sum(1 for c in dc_name if c.split('_')[0] in A_cross_rings or c.split('_')[0] in X_cross_rings)
        n_glyco = n_cleavages - n_cross
        penalties = fragmentation_priors['multi_cleavage_penalty']
        score *= penalties['glycosidic'] ** (max(n_glyco - 1, 0) if n_glyco else 0)
        score *= penalties['cross_ring'] ** (n_cross if n_glyco else max(n_cross - 1, 0))
    return score * fragmentation_priors['hydrogen_shift'].get(hydrogens, 1.0)


def count_cleavages(dc_name):
    """Number of cleavages a fragment name stands for, as fragments are ranked: a global modification counts as one, the intact precursor and a hydrogen
    shift alone (an odd-electron or hydrogen-shifted ion, M_+H) as none"""
    return sum(not c.startswith('M') or (c != 'M' and not (('+H' in c or '-H' in c) and set(parse_global_mod(c[2:])) <= HYDROGEN_ATOMS)) for c in dc_name)


@lru_cache(maxsize = 65536)
def fragment_rank(dc_name, charge, chain_lengths = None):
    """count_cleavages and score_fragment_prior of a fragment name as a tuple (chain_lengths as a tuple of its items), which priority_filter and
    simplify_fragments look up for every peak a fragment can explain"""
    return count_cleavages(dc_name), score_fragment_prior(dc_name, charge, chain_lengths = None if chain_lengths is None else dict(chain_lengths))


def compute_fragment_lability(edge_lability, subg):
    """Scores how labile the broken glycosidic bonds are (edge_lability: bond lability by edge of the parent glycan); higher means more expected cleavage"""
    broken = [lability for (u, v), lability in edge_lability.items() if (u in subg) != (v in subg)]
    return sum(broken) / len(broken) if broken else DEFAULT_LABILITY


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
        glycan_strings.append(mono_frag_to_string(nx.relabel_nodes(glyc, {x: int(x.split('-')[1]) for x in glyc.nodes}, copy = True)))
    if not pep_part:
        return '*'.join(glycan_strings)
    if not glycan_strings:
        return pep_part
    return pep_part + '*' + '*'.join(glycan_strings)


def priority_filter(dc_names, diffs, peptide = False, charge = -1, lability = None, chain_lengths = None):
    """Filters Domon-Costello fragment names by number of cleavages, fragmentation prior, and difference from observed mass\n
    | Arguments:
    | :-
    | dc_names (list): Domon-Costello names (each a list of cleavages) of the fragments matching one observed mass
    | diffs (list): their differences from the observed mass
    | peptide (bool): whether the input is a glycopeptide; default:False
    | charge (int): charge state of the precursor ion; default:-1
    | lability (list): lability score of each fragment; default:None (DEFAULT_LABILITY for all)
    | chain_lengths (dict): residues per chain rank of the glycan, as score_fragment_prior takes them; default:None\n
    | Returns:
    | :-
    | Returns the fragment names sorted by number of cleavages (none for the intact precursor), prior score, observed mass difference and lability,
    | and their mass differences and lability scores in the same order
    """
    lability = [DEFAULT_LABILITY] * len(dc_names) if lability is None else list(lability)
    if peptide:
        sorted_frags = sorted(zip(dc_names, diffs, lability),
                              key = lambda x: (count_gp_cleavages(x[0]), -score_gp_prior(x[0], charge), x[1]))
    else:
        lengths = None if chain_lengths is None else tuple(chain_lengths.items())
        sorted_frags = sorted(zip(dc_names, diffs, lability),
                              key = lambda x: ((rank := fragment_rank(tuple(x[0]), charge, chain_lengths = lengths))[0], -rank[1], x[1], -x[2]))
    return [f[0] for f in sorted_frags], [f[1] for f in sorted_frags], [f[2] for f in sorted_frags]


def max_fragment_charge(graph, peptide = False):
    """Caps a fragment's charge at the number of sites that can plausibly carry one: the N-terminus and basic residues of its peptide
    part, plus, with glycan residues on it, one more on a backbone fragment (EThcD c3 2+ of ETQ with sialyl-T) or one per four glycan
    residues on the intact peptide (EEQYNSTYR, one basic residue, is seen at 3+ with G0F and at 4+/5+ with sialylated glycans)"""
    if not peptide:
        return graph.number_of_nodes()
    # a Fragment's labels and terminal atom dictionaries are read directly, without building its graph
    fragment = isinstance(graph, Fragment)
    labels = {k: d['string_labels'] for k, d in graph.nodes.items()} if fragment else nx.get_node_attributes(graph, 'string_labels')
    pep_labels = [v for k, v in labels.items() if k.split('-')[0] == '0']
    if not pep_labels:
        return graph.number_of_nodes()
    n_glycan = len(labels) - len(pep_labels)
    backbone = any(isinstance(x, str) and x.startswith('peptide_') for d in (graph.atom_mods if fragment else nx.get_node_attributes(graph, 'atomic_mod_dict').values())
                   for x in d.values())
    return 1 + sum(1 for v in pep_labels if v in BASIC_RESIDUES) + (min(n_glycan, 1) if backbone else math.ceil(n_glycan / 4))


def match_fragment_properties(subg_frags, mass, mass_threshold, charge, sorted_frag_keys = None, peptide = False,
                              mass_threshold_ppm = None):
    """Searches subg_frags for any fragments which could correspond to the observed mass and its charge\n
    | Arguments:
    | :-
    | subg_frags (dict): lists of Fragments indexed by their mass, from generate_atomic_frags
    | mass (float): the observed mass to match potential fragments against
    | mass_threshold (float): the range around the observed mass in which to match potential fragments
    | charge (int): the maximum possible charge on the fragments to be matched
    | sorted_frag_keys (list): the sorted keys of subg_frags, if already at hand; default:None
    | peptide (bool): whether the fragments are glycopeptide fragments, which caps their charge; default:False
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; default:None\n
    | Returns:
    | :-
    | (1) a list of only the observed mass with length equal to the number of matched outputs
    | (2) a list of the theoretical masses of the fragments matched with the observed mass
    | (3) a list of each of the differences from the matched fragments and the observed mass
    | (4) a list of the charge of each matched fragment
    | (5) a list of the matched Fragments
    """
    fragment_properties = []
    # the sign as a plain int for the arithmetic (numpy scalars are slow), as numpy's for the charges reported
    modifier, sign = np.sign(charge), 1 if charge > 0 else -1
    if sorted_frag_keys is None:
        sorted_frag_keys = sorted(subg_frags.keys())
    for z in range(1, abs(charge) + 1):
        charged_mass = (mass * z) - (z - 1) * PROTON_MASS * sign
        lo = bisect.bisect_left(sorted_frag_keys, charged_mass - mass_threshold)
        hi = bisect.bisect_right(sorted_frag_keys, charged_mass + mass_threshold)
        for frag_mass in sorted_frag_keys[lo:hi]:
            if mass_threshold_ppm is not None and abs(charged_mass - frag_mass) > frag_mass * mass_threshold_ppm / 1e6:
                continue
            for graph in subg_frags[frag_mass]:
                if z > 1 and z > max_fragment_charge(graph, peptide = peptide):
                    continue
                # metal cations beyond the fragment's charge each neutralize one of its acidic groups ([F+2Na-H]+ of a sialic acid salt)
                if (mod := graph.global_mod) and any(x in mod for x in METAL_GLOBAL_MODS) and \
                        (metals := sum(x in METAL_GLOBAL_MODS for x in parse_global_mod(mod))) > z and \
                        metals - z > sum(acidic_groups(d['string_labels']) for d in graph.nodes.values()):
                    continue
                fragment_properties.append((mass, frag_mass, abs(charged_mass - frag_mass), modifier * z, graph))
    if fragment_properties:
        return list(zip(*fragment_properties))
    return [[], [], [], [], []]


def simplify_fragments(dc_names, peptide = False, diffs = None, intensities = None, charge = -1, prior_weight = 1.0,
                       lability_scores = None, mass_threshold = 0.5, chain_lengths = None):
    """Sorts a list of possible fragments for each observed mass into a list of one fragment per observed mass\n
    | Arguments:
    | :-
    | dc_names (list): a list of Domon-Costello fragment names grouped by mass
    | peptide (bool): whether the input is a glycopeptide; default:False
    | diffs (list): a nested list of mass differences; default:None
    | intensities (list): intensities of the observed masses; the most intense are assigned first, so weaker ones favor cleavages already
    |                     chosen; default:None (in mass order)
    | charge (int): charge state of the precursor ion; default:-1
    | prior_weight (float): scaling factor for fragmentation prior contribution to scoring; default:1.0
    | lability_scores (list): a nested list of the lability score of each fragment option; default:None
    | mass_threshold (float): mass tolerance, which scales the mass difference term for glycopeptides; default:0.5
    | chain_lengths (dict): residues per chain rank of the glycan, as score_fragment_prior takes them; default:None\n
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
    lengths = None if chain_lengths is None else tuple(chain_lengths.items())
    for i in order:
        possible_frags = sorted(dc_names[i], key = lambda f: fragment_rank(tuple(f), charge, chain_lengths = lengths)[0])
        cleavages = [fragment_rank(tuple(f), charge, chain_lengths = lengths)[0] for f in possible_frags]
        if not possible_frags or len(possible_frags[0]) == 0:
            continue
        if cleavages[0] <= 1:
            observed_frags[i] = [possible_frags[0]]
        else:
            min_cleavages = cleavages[0]
            frag_options = [x for x, c in zip(possible_frags, cleavages) if c == min_cleavages]
            # a shared global modification is not evidence of a shared cleavage
            max_overlaps_seen = [sum(seen_cuts[c] for c in set(f) if c[0] != 'M') for f in frag_options]
            prior_scores = [fragment_rank(tuple(f), charge, chain_lengths = lengths)[1] for f in frag_options]
            if lability_scores and lability_scores[i]:
                option_lability = [l for c, l in zip(cleavages, lability_scores[i]) if c == min_cleavages]
            else:
                option_lability = [DEFAULT_LABILITY] * len(frag_options)
            if diffs and diffs[i]:
                option_diffs = [d for c, d in zip(cleavages, diffs[i]) if c == min_cleavages]
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


def get_initial_global_mods(nx_mono, charge, disable_global_mods = False, max_global_mods = 1, sample_prep = 'underivatized', hydrogen_shifts = False,
                            metal_cations = True):
    """Creates a list of global modifications dependent on the original structure, ion mode and sample preparation (methanol is lost by permethylated glycans
    only); a free glycan's fragment can carry as many metal cations as max_global_mods allows (none without metal_cations: fragments of a (de)protonated
    precursor), and with hydrogen_shifts (radical activation) every modification, and none, also comes with each of HYDROGEN_SHIFTS"""
    if disable_global_mods:
        return [None] + (list(HYDROGEN_SHIFTS) if hydrogen_shifts else []), []
    global_mods = [x for x in mono_attributes['Global']['mass'] if x not in ['CO2', 'SO4', 'PO4', *HYDROGEN_ATOMS] and (x not in METHANOL_LOSSES.values() or x == METHANOL_LOSSES.get(sample_prep))]
    charge_mods = {-1: ['+Na', '+K', '+Li'], 1: ['+Acetate']}
    metals = DEFAULT_METAL_CATIONS if metal_cations is True else set(metal_cations or ())
    global_mods = [mod for mod in global_mods if mod not in charge_mods[np.sign(charge)] and (mod in metals or mod not in METAL_GLOBAL_MODS)]
    labels = [v for v in nx.get_node_attributes(nx_mono, 'string_labels').values() if len(v) > 1]
    special_mod_residues = ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn', 'S', 'P']
    # residue by residue (the joined labels of Glc(a1-4)Ara contain GlcA), with every uronic acid (IdoA, GalA) as HexA and only phosphate
    # substituents (Man6P, GlcNAcOP) as P, not the P of Pen, Pse, Pyr or Prop; a derivatized carboxyl (derivatize_carboxyls) loses no CO2
    present_special_residues = [x for x in special_mod_residues if any(re.search(r'(\d|O)P(?![a-z])', v) if x == 'P' else x in v or (
        x == 'HexA' and map_to_basic(v, obfuscate_ptm = False).startswith(x)) for v in labels if v not in CARBOXYL_LABELS)]
    combos = sorted(global_mods)
    if max_global_mods > 1:
        peptide = isinstance(next(iter(nx_mono)), str)
        repeatable = [x for x in combos if x in REPEATABLE_GLOBAL_MODS or (x in METAL_GLOBAL_MODS and not peptide)]
        combos += sorted({combine_global_mods(c) for n in range(2, max_global_mods + 1)
                          for c in combinations_with_replacement(repeatable, n)})
    if hydrogen_shifts:
        combos += list(HYDROGEN_SHIFTS) + [combine_global_mods(parse_global_mod(c) + parse_global_mod(s)) for c in combos for s in HYDROGEN_SHIFTS]
    return [None] + combos, present_special_residues


def infer_glycosites(peptide_seq, glycan_class = None):
    """Infers potential N- and O-glycosylation sites from a peptide sequence\n
    | Arguments:
    | :-
    | peptide_seq (string): amino acid sequence
    | glycan_class (string): 'N', 'O', or None (both); default:None\n
    | Returns:
    | :-
    | Returns a list of 0-indexed positions: N-X-S/T (X not P) and S/T in sequence order, then N-X-C and an N-X ending the peptide
    """
    sites, other_n = [], []
    for i, aa in enumerate(peptide_seq):
        if glycan_class != 'O' and aa == 'N' and i + 2 < len(peptide_seq) and peptide_seq[i + 1] != 'P' and peptide_seq[i + 2] in 'ST':
            sites.append(i)
        elif glycan_class != 'N' and aa in 'ST':
            sites.append(i)
        # N-X-C sequons (also with Carbamidomethyl Cys, c) and an N-X at the C-terminus, whose S/T lies past the cleavage, come after these sites, so that callers
        # taking the first ones still pick a canonical site where there is one
        elif glycan_class != 'O' and aa == 'N' and peptide_seq[i + 1:i + 2] not in ('', 'P') and (peptide_seq[i + 2:i + 3] in ('', 'C', 'c')):
            other_n.append(i)
    return sites + other_n


def peptide_to_glycowork(peptide):
    """Writes a peptide in CandyCrunch's one-letter code (c, j, k, m, n, q for modified residues) in glycowork's notation, for glycowork's glycopeptide masses and formulas\n
    | Arguments:
    | :-
    | peptide (string): amino acid sequence, e.g., 'NcSK'\n
    | Returns:
    | :-
    | Returns the peptide with named modifications, e.g., 'NC[Carbamidomethyl]SK', for composition_to_mass, glycan_to_mass or get_formula(..., peptide = ...)
    """
    return ''.join(f'{TOKEN_MODIFICATIONS[aa][0]}[{TOKEN_MODIFICATIONS[aa][1]}]' if aa in TOKEN_MODIFICATIONS else aa for aa in peptide)


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
        if not mod:
            continue
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
    red_nodes = [next(iter(nx.get_node_attributes(glyc, 'reducing_end'))) for glyc in glycan_graphs]
    prefixes = [f'{l}-' for l in range(len(glycan_graphs) + 1)]
    glyco_pep = nx.union_all([pep_gr] + glycan_graphs, rename = prefixes)
    for pref, gsite, red_node in zip(prefixes[1:], glycosites, red_nodes):
        glyco_pep.add_edge(f'0-{gsite}', f'{pref}{red_node}')
        glyco_pep[f'0-{gsite}'][f'{pref}{red_node}']['bond_label'] = 'glycosite'
    pep_gr = copy.deepcopy(glyco_pep.subgraph([x for x in glyco_pep.nodes() if x.split('-')[0] == '0']))
    return glyco_pep, pep_gr


def input_to_graph(input_dict):
    if not input_dict['peptide'] and not input_dict['glycosites']:
        if isinstance(input_dict['glycans'], str) or len(input_dict['glycans']) == 1:
            # a glycan's graph is only read, and the same structures come up for many spectra (supporting_ions compares each with all its placements)
            glycan = input_dict['glycans'] if isinstance(input_dict['glycans'], str) else input_dict['glycans'][0]
            # read once with get, as another thread (the app's annotation) may clear the store in between
            nx_mono = _GLYCAN_GRAPHS.get(glycan)
            if nx_mono is None:
                if len(_GLYCAN_GRAPHS) > 5000:
                    _GLYCAN_GRAPHS.clear()
                nx_mono = _GLYCAN_GRAPHS[glycan] = mono_graph_to_nx(glycan_to_graph_monos(glycan))
            return nx_mono, None
    if input_dict['peptide']:
        return create_glycopeptide_graph(input_dict['peptide'], input_dict['glycans'], input_dict['glycosites'])


def get_glycan_cleavages(gp, subgs, glycosites, chain_ranks):
    """Return Domon-Costello labels for all glycans on each glycopeptide fragment (Fragments or graphs; chain_ranks: rank_chains of each glycan, keyed by its
    graph prefix), each glycan's cleavages named for all fragments in one call"""
    all_mods = [[] for _ in subgs]
    atom_dicts = [dict(zip(subg.terminals, subg.atom_mods)) if isinstance(subg, Fragment) else nx.get_node_attributes(subg, 'atomic_mod_dict') for subg in subgs]
    for prefix, glycosite in enumerate(glycosites, 1):
        prefix = str(prefix)
        glyc = [x for x in gp.nodes() if x.split('-')[0] == prefix]
        holding, parts = [], []
        for k, subg in enumerate(subgs):
            if any(x.split('-')[0] == prefix for x in subg.nodes):
                holding.append(k)
                if isinstance(subg, Fragment):
                    # the glycan's part of the fragment, only to be named (without the dicts its graph would need)
                    terms = [i for i, t in enumerate(subg.terminals) if t.split('-')[0] == prefix]
                    parts.append(Fragment({x: d for x, d in subg.nodes.items() if x.split('-')[0] == prefix}, None, None, [subg.mono_mods[i] for i in terms],
                                          [subg.atom_mods[i] for i in terms], subg.global_mod, [subg.terminals[i] for i in terms]))
                else:
                    parts.append(subg.subgraph(glyc))
            elif f'0-{glycosite}' in atom_dicts[k] and atom_dicts[k][f'0-{glycosite}'][3]:
                all_mods[k].append([f'{cut_type_dict[atom_dicts[k][f"0-{glycosite}"][3]]}_0_Alpha'])
            else:
                all_mods[k].append([f'loss of glycan {prefix}'])
        for k, name in zip(holding, subgraphs_to_domon_costello(gp.subgraph(glyc), parts, chain_ranks[prefix]) if parts else []):
            all_mods[k].append(name if name else ['M'])
    return all_mods


def peptide_to_RF_nomenclature(peptide, pep_subg, allowed_ion_types = None, allow_internal = False):
    """Return Roepstorff and Fohlman peptide fragment nomenclature, or None if the fragment is rejected"""
    RF_cleavages = []
    # the peptide's residues are the nodes '0-0' to '0-{n - 1}' (create_peptide_graph), so a cut's site is read off its node
    peptide_node_set = set(peptide.nodes)
    if not any(x in peptide_node_set for x in pep_subg.nodes):
        return ['No Peptide']
    # a Fragment's terminal atom dictionaries and global modification are read directly, without building its graph
    if isinstance(pep_subg, Fragment):
        pep_atom_dict = {k: v for k, v in zip(pep_subg.terminals, pep_subg.atom_mods) if k in peptide_node_set}
        global_mods = {None: [pep_subg.global_mod]} if pep_subg.global_mod else {}
    else:
        pep_atom_dict = {k: v for k, v in nx.get_node_attributes(pep_subg, 'atomic_mod_dict').items() if
                         k in peptide_node_set}
        global_mods = nx.get_node_attributes(pep_subg, 'global_mod')
    n_term_cuts, c_term_cuts = 0, 0
    for node, cleavages in pep_atom_dict.items():
        for cleavage in [x for x in cleavages.values() if isinstance(x, str) and x.startswith('peptide_')]:
            ion_type = cleavage[len('peptide_'):]
            if allowed_ion_types is not None and ion_type not in allowed_ion_types:
                return None
            if ion_type in C_TERM_IONS:
                c_term_cuts += 1
                cut_site = len(peptide_node_set) - int(node.split('-')[1])
            elif ion_type in N_TERM_IONS:
                n_term_cuts += 1
                cut_site = int(node.split('-')[1]) + 1
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
        RF_cleavages.append(f"M_{next(iter(global_mods.values()))[0]}")
    return sorted(RF_cleavages)


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
    input_dict['peptide'] = ''.join(split_string[::2])
    input_dict['glycans'] = split_string[1::2]
    input_dict['glycosites'] = np.cumsum([len(x) for x in split_string[::2][:-1]]) - 1
    return input_dict


def get_derivatization_count(mono_type, fragment_type, sample_prep = 'permethylated'):
    """Returns the number of derivatized groups (methyls when (deutero)permethylated, acetyls when peracetylated) a fragment keeps"""
    return mono_attributes.get(mono_type, {}).get('permethylated' if sample_prep == 'deuteropermethylated' else sample_prep, {}).get(fragment_type, 0)


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
    # a sialic acid of known linkage (Neu5Ac_a2-6) is its residue, unless derivatize_carboxyls made it a residue of its own
    linked = {k: base for base, keys in SIALIC_KEYS.items() for k in keys}
    if not linked.keys().isdisjoint(composition):
        folded = {}
        for k, v in composition.items():
            folded[linked.get(k, k)] = folded.get(linked.get(k, k), 0) + v
        composition = folded
    mono_types = sorted(m for m in composition if (m in derivatization_sites['permethylated'] or m in CARBOXYL_LABELS) and composition[m] > 0)
    # a residue without fragment tables (Kdo, Hep, dHexNAc) would be missing from every fragment, the intact molecule included
    if not mono_types or any(v > 0 and m not in mono_types and m not in SUBSTITUENTS for m, v in composition.items()):
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


def add_charge_reduced_ions(hit_dict, molecule_mass, n_glycans, charge, mass_threshold, mass_threshold_ppm = None):
    """Annotates the peaks of an electron-transfer glycopeptide spectrum that no fragment explains as the charge-reduced intact molecule or its small losses
    (CHARGE_REDUCED_LOSSES), named as the intact glycopeptide with the hydrogen atoms and losses as global modifications ('M_2+H|NH3')\n
    | Arguments:
    | :-
    | hit_dict (dict): observed m/z : CandyCrumbs annotation or None, changed in place
    | molecule_mass (float): neutral monoisotopic mass of the glycopeptide
    | n_glycans (int): number of glycans on it
    | charge (int): precursor charge
    | mass_threshold (float): tolerance in Da of the singly charged mass, as in matching
    | mass_threshold_ppm (float): relative tolerance in ppm on top of it; default:None\n
    | Returns:
    | :-
    | Returns hit_dict
    """
    charge = abs(charge)
    empty = [m for m, hit in hit_dict.items() if hit is None]
    if charge < 2 or not empty:
        return hit_dict
    # an ion of charge q that kept all charge protons carries charge - q hydrogen atoms more than the protonated intact molecule
    options = [(q, mod) for q in range(1, charge) for losses, shifts in CHARGE_REDUCED_LOSSES
               for mod in (combine_global_mods(list(losses) + ['+H'] * (charge - q + shift)) for shift in shifts)]
    zs = np.array([q for q, _ in options])
    targets = np.array([molecule_mass + PROTON_MASS + global_mod_mass(mod) for _, mod in options])
    errors = np.abs(np.array(empty)[:, None] * zs[None, :] - (zs[None, :] - 1) * PROTON_MASS - targets[None, :])
    errors[errors > (np.minimum(mass_threshold, targets * mass_threshold_ppm / 1e6) if mass_threshold_ppm is not None else mass_threshold)] = np.inf
    for observed_mass, row in zip(empty, errors):
        k = int(np.argmin(row))
        if np.isfinite(row[k]):
            hit_dict[observed_mass] = {'Theoretical fragment masses': [round(float(targets[k]), 5)], 'Fragment charges': [int(zs[k])],
                                       'Domon-Costello nomenclatures': [[['Peptide'] + ([f'M_{options[k][1]}'] if options[k][1] else [])] + [['M'] for _ in range(n_glycans)]]}
    return hit_dict


def composition_to_fragments(composition, fragment_masses, mass_threshold, max_cleavages = 3,
                             charge = -1, mass_tag = None, simplify = True, disable_global_mods = False,
                             disable_X_cross_rings = None, sample_prep = 'underivatized',
                             peptide_seq = None, glycosites = None, glycan_class = None,
                             fragmentation_method = None, max_global_mods = 1, mass_threshold_ppm = None, hydrogen_shifts = False, metal_cations = True):
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
    | disable_X_cross_rings (bool): whether to disable X-type cross-ring cleavages; default:None (disabled in positive mode)
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | peptide_seq (string): amino acid sequence; when provided, generates glycopeptide fragments; default:None
    | glycosites (list): 0-indexed peptide positions for each glycan; inferred from sequence if None; default:None
    | glycan_class (string): 'N' or 'O', to infer glycosites; default:None (both)
    | fragmentation_method (string or set): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD', or a set of peptide ion types; default:None (all)
    | max_global_mods (int): how many global modifications (metal cations on a free glycan's fragment included) may co-occur on one fragment; default:1
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; default:None
    | hydrogen_shifts (bool): whether free-glycan fragments also come one or two hydrogen atoms off their mass (radical activation); default:False
    | metal_cations (bool or set): whether fragments can carry metal cations (positive mode; True: Na+ and K+), which those of a (de)protonated precursor
    |                              cannot, or the ones they can carry (e.g., {'+Li'} for a lithiated precursor); default:True\n
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
    all_comps = ([composition] if isinstance(composition, dict) else list(composition)) if is_glycopeptide else [composition]
    window_masses = extend_masses(fragment_masses, charge)
    # Global modifications (shared), set up first, as a glycopeptide's backbone fragments are only made near an observed mass, also after one of them
    if not disable_global_mods:
        charge_exclude = {-1: ['+Na', '+K', '+Li'], 1: ['+Acetate']}
        excluded = set(charge_exclude.get(np.sign(charge), [])) | (METAL_GLOBAL_MODS - (DEFAULT_METAL_CATIONS if metal_cations is True else set(metal_cations or ())))
        if not is_glycopeptide:
            excluded.add('NH3')
        global_mods_dict = {k: v for k, v in mono_attributes['Global']['mass'].items()
                            if k not in ('CO2', 'SO4', 'PO4', *HYDROGEN_ATOMS) and k not in excluded and (k not in METHANOL_LOSSES.values() or k == METHANOL_LOSSES.get(sample_prep))}
        if max_global_mods > 1:
            repeatable = [x for x in global_mods_dict if x in REPEATABLE_GLOBAL_MODS or (x in METAL_GLOBAL_MODS and not is_glycopeptide)]
            for n in range(2, max_global_mods + 1):
                for combo in combinations_with_replacement(repeatable, n):
                    global_mods_dict[combine_global_mods(combo)] = sum(global_mods_dict[x] for x in combo)
        # per residue, as the joined labels of {'Hex': 2, 'Ac': 1} contain HexA; a sialic acid of known linkage (Neu5Ac_a2-6) too
        if any(m.split('_')[0] in ['Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn'] for comp in all_comps for m in comp):
            global_mods_dict['CO2'] = mono_attributes['Global']['mass']['CO2']
        if any('S' in comp for comp in all_comps):
            global_mods_dict['SO4'] = mono_attributes['Global']['mass']['SO4']
        if any(m == 'P' or m.endswith('P') for comp in all_comps for m in comp):
            global_mods_dict['PO4'] = mono_attributes['Global']['mass']['PO4']
    if is_glycopeptide:
        compositions = all_comps
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
        all_glycan_frags = [_build_composition_fragments(comp, 0, sample_prep, disable_X_cross_rings) for comp in compositions]
        # as for a structure, a glycan without fragments (a residue without fragment tables) leaves nothing to annotate
        if not all(all_glycan_frags):
            return dict.fromkeys(fragment_masses)
        glycan_totals = [next((m for m, lb, nc in gf if nc == 0), 0) for gf in all_glycan_frags]
        total_glycan = sum(glycan_totals)
        # Peptide masses
        pep_res = [AA_masses.get(aa, 0) for aa in peptide_seq]
        n_aa = len(pep_res)
        full_pep = sum(pep_res) + WATER_MASS
        cum_n = list(np.cumsum(pep_res))
        cum_c = list(np.cumsum(pep_res[::-1]))
        allowed_ion_types = PEPTIDE_ION_TYPES[fragmentation_method] if isinstance(fragmentation_method, (str, type(None))) else set(fragmentation_method)
        pep_ions = [('b', lambda i: cum_n[i - 1], 'n'),
                    ('a', lambda i: cum_n[i - 1] - 27.994915, 'n'),
                    ('c', lambda i: cum_n[i - 1] + 17.026549, 'n'),
                    ('y', lambda i: cum_c[i - 1] + WATER_MASS, 'c'),
                    ('z', lambda i: cum_c[i - 1] + WATER_MASS - 17.026549 + HYDROGEN_MASS, 'c'),
                    ('c-1', lambda i: cum_n[i - 1] + 17.026549 - HYDROGEN_MASS, 'n'),
                    ('z+1', lambda i: cum_c[i - 1] + WATER_MASS - 17.026549 + 2 * HYDROGEN_MASS, 'c'),
                    ('w', lambda i: cum_c[i - 1] + WATER_MASS - 17.026549 + HYDROGEN_MASS - W_SIDE_CHAIN_LOSSES[peptide_seq[n_aa - i]]
                     if peptide_seq[n_aa - i] in W_SIDE_CHAIN_LOSSES else None, 'c')]
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
        # options on EEQYNSTYR), while O-sites also carry Man, Fuc, Glc, or Xyl. X cross-rings only on an N-site: the peptide + 0,2X/0,4X HexNAc of
        # N-glycopeptides were real, X cross-rings on O-glycopeptides as frequent as in the same spectra shifted out of tolerance
        glycan_options = []
        for g_idx in range(n_glycans):
            opts = [(['M'], glycan_totals[g_idx], 0), ([f'loss of glycan {g_idx + 1}'], 0.0, 1)]
            opts += [([gl], gm, gc) for gm, gl, gc in all_glycan_frags[g_idx] if gc > 0 and (
                    gl.split(' ')[0] in ('Y', 'Z') or (gl.split(' ')[0] in X_cross_rings and peptide_seq[glycosites[g_idx]] == 'N')) and (
                    peptide_seq[glycosites[g_idx]] != 'N' or 'HexNAc' in [c.split('(')[0] for c in gl.split(' ')[1].split('/')])]
            glycan_options.append(opts)
        # A backbone ion keeps each glycan on its side whole, loses it, or keeps the core HexNAc of an N-glycan (y + HexNAc in HCD): in spectra of
        # identified glycopeptides, backbone ions with any other part of a glycan were annotated no more often than in the same spectra shifted out of
        # tolerance, while the intact peptide's Y/Z and X ions were real
        backbone_options = [[o for o in opts if o[2] == 0 or o[0][0].startswith('loss of') or (peptide_seq[glycosites[g_idx]] == 'N' and o[0][0] == 'Y HexNAc(1)')]
                            for g_idx, opts in enumerate(glycan_options)]
        # Intact peptide with any combination of glycan states
        for combo in product(*glycan_options):
            nc = sum(c[2] for c in combo)
            if not 0 < nc <= max_cleavages:
                continue
            add(full_pep + sum(c[1] for c in combo) + mode_mass, [['Peptide']] + [list(c[0]) for c in combo], nc)
        # Peptide backbone fragments; glycans outside the fragment are lost without costing a cleavage. They are most of the fragments (each backbone
        # ion with each glycan state), and few come near an observed mass, so with one glycan the masses are summed in numpy and only those near an
        # observed mass, as they are or after a global modification, are made (with a margin, a superset of what matching and the global modifications use)
        offsets = [0.0] + ([global_mod_mass(g, mode_mass = mode_mass) for g in global_mods_dict if g in GLYCOPEPTIDE_GLOBAL_MODS['backbone']] if not disable_global_mods else [])
        targets = (np.asarray(window_masses, dtype = float)[:, None] - np.array(offsets)[None, :]).ravel()
        opt_masses = np.array([c[1] for c in backbone_options[0]], dtype = float)
        opt_cleavages = np.array([c[2] for c in backbone_options[0]])
        for ion_name, mass_fn, terminus in pep_ions:
            for i in range(first_i.get(ion_name, 1), n_aa):
                pmass = mass_fn(i)
                if pmass is None:
                    continue
                plabel = f'{ion_name}_{i}'
                opts = [backbone_options[j] if (gs < i if terminus == 'n' else gs >= n_aa - i)
                        else [([f'loss of glycan {j + 1}'], 0.0, 0)] for j, gs in enumerate(glycosites)]
                if n_glycans == 1 and len(targets) and opts[0] is backbone_options[0]:
                    keys = np.round(pmass + opt_masses + mode_mass, 5)
                    for k in np.flatnonzero((opt_cleavages < max_cleavages) & check_masses(targets, keys, mass_threshold + 2e-4)):
                        frag_dict.setdefault(keys[k], []).append(([[plabel], list(backbone_options[0][k][0])], 1 + int(opt_cleavages[k])))
                    continue
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
            # the HexNAc oxonium ions 138.055 and 126.055 (their ratio to 144.065 tells GlcNAc from GalNAc) lose more than max_global_mods combines
            if 'HexNAc' in comp and charge > 0 and sample_prep == 'underivatized':
                glyc = [[f'loss of glycan {j + 1}'] for j in range(n_glycans)]
                glyc[g_idx] = ['B HexNAc(1)']
                for loss in ('CH2O|2H2O', 'C2H4O2|H2O'):
                    add(mono_attributes['HexNAc']['mass']['HexNAc'] + global_mod_mass(loss) + mode_mass, [['No Peptide', f'M_{loss}']] + glyc, 2)
    else:
        # Pure composition (existing behavior)
        # a derivatized reducing end carries one more group, and an alditol (mass_tag of 2 H) one more again
        re_bonus = WATER_MASS + mass_tag + DERIVATIZATION_MASSES.get(sample_prep, 0) * (1 + (abs(mass_tag - 2 * HYDROGEN_MASS) < 0.01))
        glycan_frags = _build_composition_fragments(composition, re_bonus, sample_prep, disable_X_cross_rings)
        if not glycan_frags:
            return dict.fromkeys(fragment_masses)
        for gm, gl, gc in glycan_frags:
            frag_dict.setdefault(round(gm + mode_mass, 5), []).append(([gl], gc))
    if not disable_global_mods:
        # snapshot the label lists, else a variant landing on an existing mass is modified again by later global mods
        base_entries = [(base_mass, list(entries)) for base_mass, entries in frag_dict.items()]
        base_masses = np.array([base_mass for base_mass, _ in base_entries])
        for gmod in global_mods_dict:
            # every metal cation of a combination (2+Na) takes a charge carrier's place
            adj_mass = global_mod_mass(gmod, mode_mass)
            # only variants that can match an observed mass are materialized; the margin covers the 5-decimal rounding
            in_window = check_masses(window_masses, base_masses + adj_mass, mass_threshold + 1e-4) if len(window_masses) else []
            for j in np.flatnonzero(in_window):
                base_mass, entries = base_entries[j]
                new_mass = round(base_mass + adj_mass, 5)
                for label, nc in entries:
                    if is_glycopeptide and gmod not in GLYCOPEPTIDE_GLOBAL_MODS.get(label[0][0], GLYCOPEPTIDE_GLOBAL_MODS['backbone']):
                        continue
                    if gmod in ('CO2', 'SO4', 'PO4'):
                        # as update_global_mods does for structures, a residue's own loss needs that residue on the fragment: in the glycans an ion keeps
                        # whole ('M' on a peptide) or in part ('B Hex(1)/Neu5Ac(1)', '02A Neu5Ac'), not in those it lost
                        residues = {r.split('(')[0].split('_')[0] for i, (piece, *_) in enumerate(label[1:] if is_glycopeptide else [label])
                                    if not piece.startswith('loss of') for r in (all_comps[i] if piece == 'M' else piece.split(' ')[1].split('/'))}
                        if not (residues & {'Neu5Ac', 'Neu5Gc', 'GlcA', 'HexA', 'Kdn'} if gmod == 'CO2' else 'S' in residues if gmod == 'SO4' else
                                any(r == 'P' or r.endswith('P') for r in residues)):
                            continue
                    if nc + 1 <= max_cleavages:
                        if is_glycopeptide:
                            new_label = [sub[:] for sub in label]
                            new_label[0] = new_label[0] + [f'M_{gmod}']
                        else:
                            new_label = label + [f'M_{gmod}']
                        frag_dict.setdefault(new_mass, []).append((new_label, nc + 1))
    if hydrogen_shifts and not is_glycopeptide:
        # every fragment, also one with a global modification, can be one or two hydrogen atoms off, which costs no cleavage
        base_entries = [(base_mass, list(entries)) for base_mass, entries in frag_dict.items()]
        base_masses = np.array([base_mass for base_mass, _ in base_entries])
        for shift in HYDROGEN_SHIFTS:
            in_window = check_masses(window_masses, base_masses + global_mod_mass(shift), mass_threshold + 1e-4) if len(window_masses) else []
            for j in np.flatnonzero(in_window):
                base_mass, entries = base_entries[j]
                for label, nc in entries:
                    mods = [parse_global_mod(x[2:]) for x in label if x.startswith('M_')]
                    new_label = [x for x in label if not x.startswith('M_')] + [f"M_{combine_global_mods((mods[0] if mods else []) + parse_global_mod(shift))}"]
                    frag_dict.setdefault(round(base_mass + global_mod_mass(shift), 5), []).append((new_label, nc))
    # an ion without a residue (HSO4-, a label's reporter ion) matches as itself, singly charged, ahead of any fragment
    small = small_ions(composition, charge, mass_tag) if not is_glycopeptide and not disable_global_mods else {}
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
                if mass_threshold_ppm is not None and abs(charged_mass - frag_mass) > frag_mass * mass_threshold_ppm / 1e6:
                    continue
                for label, nc in frag_dict[frag_mass]:
                    if nc > max_cleavages:
                        continue
                    # as in match_fragment_properties, metal cations beyond the charge each neutralize an acidic group of the fragment
                    if not is_glycopeptide and (metals := sum(x in METAL_GLOBAL_MODS for t in label if t.startswith('M_') for x in parse_global_mod(t[2:]))) > z and \
                            metals - z > sum((int(r.split('(')[1][:-1]) if '(' in r else 1) * acidic_groups(r.split('(')[0]) for r in label[0].split(' ')[1].split('/')):
                        continue
                    if z > 1:
                        # capped as max_fragment_charge caps a structure fragment: one charge per glycan residue of a glycan-only ion; on a peptide one at the
                        # N-terminus and each basic residue, plus one for glycan on a backbone fragment, or one per four glycan residues on the intact peptide
                        n_glycan = sum(sum(v for k, v in all_comps[i].items() if k not in SUBSTITUENTS) if piece == 'M' else
                                       sum(int(r.split('(')[1][:-1]) if '(' in r else 1 for r in piece.split(' ')[1].split('/') if r.split('(')[0] not in SUBSTITUENTS)
                                       for i, (piece, *_) in enumerate(label[1:] if is_glycopeptide else [label]) if not piece.startswith('loss of'))
                        backbone = label[0][0] if is_glycopeptide else 'No Peptide'
                        if backbone == 'No Peptide':
                            # glycan-only ions of a glycopeptide were annotated at 2+ and above no more often than by chance
                            z_max = 1 if is_glycopeptide else n_glycan
                        else:
                            ion, _, pos = backbone.partition('_')
                            pep_part = peptide_seq if backbone == 'Peptide' else peptide_seq[:int(pos)] if ion in N_TERM_IONS else peptide_seq[n_aa - int(pos):]
                            z_max = 1 + sum(aa in BASIC_RESIDUES for aa in pep_part) + (math.ceil(n_glycan / 4) if backbone == 'Peptide' else min(n_glycan, 1))
                        if z > z_max:
                            continue
                    matches.append((frag_mass, label, modifier * z, abs(charged_mass - frag_mass), nc))
        small_hits = [(m, [t], modifier, abs(observed_mass - m), 1) for t, m in small.items() if abs(observed_mass - m) <= mass_threshold and (
            mass_threshold_ppm is None or abs(observed_mass - m) <= m * mass_threshold_ppm / 1e6)]
        if matches or small_hits:
            # equal-cleavage glycopeptide options are often isobaric (a5 plus a 1,5X HexNAc is b5 that lost its glycan), so, as in the
            # structure path, the fragmentation prior of the ion types ranks them before the mass error does; a composition fragment
            # ('Y HexNAc(1)', '02X HexNAc') is scored as its cleavage type, an X cross-ring on the peptide being on the reducing-end residue
            matches.sort(key = lambda x: (x[4], -score_gp_prior([[re.sub(r'^([BCYZ]|\d\d[AX]) .*', r'\1_1', y) for y in sub] for sub in x[1]], charge), x[3])
                         if is_glycopeptide else (x[4], x[3]))
            matches = small_hits + matches
            matches = matches[:1] if simplify else matches[:5]
            hit_dict[observed_mass] = {
                'Theoretical fragment masses': [m[0] for m in matches],
                'Domon-Costello nomenclatures': [m[1] for m in matches],
                'Fragment charges': [m[2] for m in matches],
            }
        else:
            hit_dict[observed_mass] = None
    if is_glycopeptide and isinstance(fragmentation_method, str) and fragmentation_method in CHARGE_REDUCING_METHODS:
        add_charge_reduced_ions(hit_dict, full_pep + total_glycan, n_glycans, charge, mass_threshold, mass_threshold_ppm = mass_threshold_ppm)
    return hit_dict


@rescue_glycans
def CandyCrumbs(input_string, fragment_masses, mass_threshold = None,
                max_cleavages = 3, simplify = True, charge = -1, mass_tag = None,
                iupac = False, intensities = None, disable_global_mods = False, disable_X_cross_rings = None,
                disable_A_cross_rings = None, sample_prep = 'underivatized', prior_weight = 1.0, glycan_class = None,
                fragmentation_method = None, allow_internal_peptide_fragments = False, mass_threshold_ppm = None,
                max_global_mods = None, ms3_precursor = None, sialic_derivatization = None, metal_cations = True):
    """Basic wrapper for the annotation of observed masses with correct nomenclature given a glycan\n
    | Arguments:
    | :-
    | input_string (string or dict): glycan in IUPAC-condensed format, composition as dict/string, or glycopeptide as 'PEPT*glycan*IDE' or a dict with 'peptide', 'glycans', 'glycosites'
    | fragment_masses (list): all masses which are to be annotated with a fragment name
    | mass_threshold (float): the maximum tolerated mass difference around each observed mass at which to include fragments; default:None (0.5 Da, or 10 ppm for glycopeptides)
    | max_cleavages (int): maximum number of allowed concurrent fragmentations per mass; default:3
    | simplify (bool): whether to try condensing fragment options to the most likely option; default:True
    | charge (int): the charge state of the precursor ion (singly-charged, doubly-charged); default:-1
    | mass_tag (float): mass of the glycan label or reducing end modification; default:None (alditol, 2 H; none for glycopeptide compositions)
    | iupac (bool): whether to add the fragment sequence in IUPAC-condensed nomenclature to the annotations; default:False
    | intensities (list): intensities of fragment_masses; with simplify, the most intense peaks are assigned first; default:None
    | disable_global_mods (bool): whether to disable global modifications (neutral losses, adducts) and the ions without a residue (HSO4-, SO3-, H2PO4-,
    |                             PO3- of sulfates and phosphates in negative mode, the reporter ions of a 2AA/2AB/procainamide label in positive mode); default:False
    | disable_X_cross_rings (bool): whether to strip out any X-type cross-rings, the open-chain ions of a labeled reducing end (CHAIN_IONS) included;
    |                               default:None (kept)
    | disable_A_cross_rings (bool or string): whether to strip out any A-type cross-rings, or 'non_reducing' for all but the A_R ions (cross-rings of the
    |                                         reducing-terminal residue) of fragments with a metal cation; default:None ('non_reducing' in positive mode for
    |                                         unreduced glycans (mass_tag 0), stripped for other positive-mode glycans)
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | prior_weight (float): weighting of prior-informed scoring in simplify=True
    | glycan_class (string): "N" or "O" if relevant (only used to assign candidate sites in glycopeptides, nowhere else)
    | fragmentation_method (string): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD'/'EED'/'EDD'/'NETD'/'UVPD'; restricts peptide backbone ion types (not for
    |                                EED/EDD/NETD/UVPD; c-1 and z+1 only with ETD/ECD/EThcD/ETciD), and the radical methods (all but CID/HCD) give free-glycan
    |                                fragments that are one (odd-electron, M_+H) or two hydrogen atoms (M_2-H) off, at no cleavage but half (a quarter) of their
    |                                fragmentation prior; default:None (all but c-1 and z+1)
    | allow_internal_peptide_fragments (bool): whether to allow peptide fragments cleaved at both termini; default:False
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; default:None
    | max_global_mods (int): how many global modifications may co-occur on one fragment; 2 captures the
    |                        sequential water losses of the oxonium series, and a free glycan's fragments with two metal cations
    |                        ([F+2Na-H]+ of a sialic acid salt, [F+2Na]2+). Counts as one cleavage either way;
    |                        default:None (2 for glycopeptides, 1 for free glycans)
    | ms3_precursor (float): m/z of the MS2 fragment that was isolated for an MS3 spectrum whose peaks are fragment_masses; these are then only
    |                        annotated as fragments of what that MS2 fragment can be in this glycan (within max_cleavages), with up to
    |                        max_cleavages cleavages that fragment does not have (at most one X cross-ring), and all are None if it cannot be any
    |                        fragment of it; glycan structures only; default:None
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization of an underivatized sample (or of one permethylated after it), 'EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a
    |                        pair of glycowork SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids; every sialic acid then needs its linkage (a2-6 or a2-3/8/9 in a
    |                        structure, E/L in a composition), and fragments keeping its C1 carry the product; default:None
    | metal_cations (bool or set): whether fragments can carry metal cations (positive mode; True: Na+ and K+), which those of a (de)protonated precursor
    |                              cannot, or the ones they can carry (e.g., {'+Li'} for a lithiated precursor); default:True\n
    | Returns:
    | :-
    | Returns a dict keyed by observed mass, each pointing to None or a dict of 'Theoretical fragment masses', 'Domon-Costello nomenclatures' and
    | 'Fragment charges' (the best option, or up to 5 without simplify; plus 'Fragment IUPAC' with iupac)
    """
    if not len(fragment_masses):
        return {}
    # Any other sample_prep would be annotated at underivatized masses
    if sample_prep not in ('underivatized', 'permethylated', 'deuteropermethylated', 'peracetylated'):
        raise ValueError(f"Unknown sample_prep {sample_prep!r}; CandyCrumbs annotates 'underivatized', 'permethylated', 'deuteropermethylated' or 'peracetylated' glycans.")
    if sialic_derivatization is not None:
        if sample_prep not in ('underivatized', 'permethylated', 'deuteropermethylated'):
            raise ValueError(f"sialic_derivatization is a derivatization of underivatized glycans or of glycans (deutero)permethylated after it, not of {sample_prep} ones.")
        if sample_prep in ('permethylated', 'deuteropermethylated') and 'lactone' in (SIALIC_DERIVATIZATIONS.get(sialic_derivatization, ()) if isinstance(sialic_derivatization, str) else sialic_derivatization):
            raise ValueError(f"A lactone does not survive permethylation; give the product its sialic acids end up as, e.g., ('dimethylamide', 'methyl ester') instead of {sialic_derivatization!r}.")
        if isinstance(input_string, dict) and 'peptide' in input_string:
            input_string = input_string | {'glycans': [derivatize_carboxyls(g, sialic_derivatization) for g in input_string.get('glycans', [])]}
        elif isinstance(input_string, str) and '*' in input_string:
            input_string = re.sub(r'\*([^*]+)\*', lambda m: f'*{derivatize_carboxyls(m[1], sialic_derivatization)}*', input_string)
        else:
            input_string = derivatize_carboxyls(input_string, sialic_derivatization)
    # Glycans in the order of their sites, as a peptide*glycan* string numbers them
    if isinstance(input_string, dict) and input_string.get('peptide') and None not in (input_string.get('glycosites') or [None]) and len(input_string['glycosites']) == len(input_string.get('glycans', [])) > 1:
        sites, glycans = zip(*sorted(zip(input_string['glycosites'], input_string['glycans']), key = lambda x: x[0]))
        input_string = input_string | {'glycosites': list(sites), 'glycans': list(glycans)}
    if isinstance(input_string, dict) and 'peptide' in input_string and not input_string['peptide'] and len(input_string.get('glycans', [])) == 1:
        input_string = input_string['glycans'][0]
    glycopeptide_input = (isinstance(input_string, dict) and bool(input_string.get('peptide'))) or (
            isinstance(input_string, str) and '*' in input_string)
    if max_global_mods is None:
        # Glycopeptide oxonium series routinely lose two waters; free glycans rarely need a second modification
        max_global_mods = 2 if glycopeptide_input else 1
    hydrogen_shifts = isinstance(fragmentation_method, str) and fragmentation_method in RADICAL_METHODS and not glycopeptide_input
    if mass_threshold_ppm is None and mass_threshold is None and glycopeptide_input:
        # Intact glycopeptides are only measurable on FT instruments, where a fixed Da window is far too
        # loose; an explicitly given mass_threshold is always respected instead
        mass_threshold_ppm = 10
    if mass_threshold is None:
        mass_threshold = 0.5
    if mass_threshold_ppm is not None:
        mass_threshold = min(mass_threshold, max(fragment_masses) * abs(charge) * mass_threshold_ppm / 1e6)
    if disable_A_cross_rings is None:
        # a metal cation opens the free reducing end of an unreduced glycan, so its A_R ions are the A cross-rings positive mode keeps
        disable_A_cross_rings = ('non_reducing' if mass_tag is not None and abs(mass_tag) < 1e-6 else True) if charge > 0 else False
        # a warning shows once per session, while a print repeated for every glycan CandyCrunch scores in positive mode
        if disable_A_cross_rings:
            warnings.warn(
                "A-type cross-ring fragmentation auto-disabled for positive mode, except the A_R ions of metal-adducted unreduced glycans; reducing-end X-type cross-rings kept (Na+/CID diagnostic). Override A with disable_A_cross_rings=False")
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
                    glycosites = list(input_string['glycosites']) if input_string.get('glycosites') is not None and len(input_string['glycosites']) else None,
                    glycan_class = glycan_class, metal_cations = metal_cations)
            else:
                glycans = input_string.get('glycans', [])
                if not glycans:
                    return dict.fromkeys(fragment_masses)
                peptide = input_string['peptide']
                gc = glycan_class if glycan_class else get_class(glycans[0])
                sites = list(input_string['glycosites']) if input_string.get('glycosites') is not None and len(input_string['glycosites']) else infer_glycosites(peptide, gc)[:len(glycans)]
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
            return dict.fromkeys(fragment_masses)
    if composition is not None:
        return composition_to_fragments(composition, sorted(fragment_masses), mass_threshold,
                                        max_cleavages = max_cleavages, charge = charge, mass_tag = mass_tag,
                                        simplify = simplify,
                                        disable_global_mods = disable_global_mods,
                                        disable_X_cross_rings = disable_X_cross_rings,
                                        sample_prep = sample_prep, glycan_class = glycan_class,
                                        max_global_mods = max_global_mods, mass_threshold_ppm = mass_threshold_ppm, hydrogen_shifts = hydrogen_shifts,
                                        metal_cations = metal_cations)
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
                glycosites = list(input_dict['glycosites']), glycan_class = glycan_class, metal_cations = metal_cations)
    if intensities is not None:
        fragment_masses, intensities = map(list, zip(*sorted(zip(fragment_masses, intensities))))
    else:
        fragment_masses = sorted(fragment_masses)
    nx_mono, pep_gr = input_to_graph(input_dict)
    node_labels = nx.get_node_attributes(nx_mono, 'string_labels')
    if any(map_to_basic(v, obfuscate_ptm = False) not in mono_attributes for v in node_labels.values() if len(v) > 1):
        return dict.fromkeys(fragment_masses)
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge,
                                                            disable_global_mods = disable_global_mods,
                                                            max_global_mods = max_global_mods, sample_prep = sample_prep,
                                                            hydrogen_shifts = hydrogen_shifts, metal_cations = metal_cations)
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
            return dict.fromkeys(fragment_masses)
        cuts = [count_cleavages(x) for x in subgraphs_to_domon_costello(nx_mono, [p[3] for p in parents])]
        parents = [p for p, c in zip(parents, cuts) if c == min(cuts)]
        subgraphs = [s for s in enumerate_subgraphs(nx_mono) + [set(nx_mono)] if any(s <= set(p[3]) for p in parents)]
        node_basic = {k: map_to_basic(v, obfuscate_ptm = False) for k, v in node_labels.items()}

        def fragment_cuts(g):
            # a fragment's cleavages as (node, atom, bond type), (node, cross-ring) and (None, global modification), so the cleavages an MS3 fragment
            # shares with the isolated fragment can be told apart from the ones that happened in MS3
            return {(n, a, v) for n, d in zip(g.terminals, g.atom_mods) for a, v in d.items() if isinstance(v, str)} | {
                (n, m) for n, m in zip(g.terminals, g.mono_mods) if m in A_cross_rings or m in X_cross_rings} | ({(None, g.global_mod)} if g.global_mod else set())

        parent_cuts = [fragment_cuts(p[3]) for p in parents]
        # what an MS3 fragment is checked against: each isolated fragment's mass, charge, nodes, cross-ring per node and cleavages; and whether each
        # fragment (at each charge) passes, as one fragment matches several MS3 peaks
        parent_info = [(p_mass, abs(p_z), set(p_g), dict.fromkeys(p_g.nodes) | dict(zip(p_g.terminals, p_g.mono_mods)), p_cuts) for (p_mass, _, p_z, p_g), p_cuts
                       in zip(parents, parent_cuts)]
        ms3_keep = {}
        # cleavages are counted from the isolated fragment, so an MS3 fragment keeping all of its cleavages may have max_cleavages more
        ms3_cleavages, max_cleavages = max_cleavages, max(map(len, parent_cuts)) + max_cleavages
    # a glycopeptide's backbone ions are only made in the types its fragmentation method gives, as only those are named
    allowed_ion_types = (PEPTIDE_ION_TYPES[fragmentation_method] if isinstance(fragmentation_method, (str, type(None))) else set(fragmentation_method)) if (
        input_dict['peptide']) else None
    subg_frags = generate_atomic_frags(nx_mono, global_mods, special_residues, allowed_X_cleavages,
                                       max_cleavages = max_cleavages, fragment_masses = fragment_masses,
                                       subgraphs = subgraphs, threshold = mass_threshold, mass_tag = mass_tag,
                                       charge = charge, sample_prep = sample_prep,
                                       disable_A_cross_rings = disable_A_cross_rings, threshold_ppm = mass_threshold_ppm, peptide_ion_types = allowed_ion_types)
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
        # read once with get, as another thread (the app's annotation) may clear the store in between
        chain_rank = _CHAIN_RANKS.get(input_dict['glycans'][0])
        if chain_rank is None:
            if len(_CHAIN_RANKS) > 5000:
                _CHAIN_RANKS.clear()
            chain_rank = _CHAIN_RANKS[input_dict['glycans'][0]] = list(rank_chains(nx_mono))
    # the lability of every glycosidic bond is looked up once, not for every candidate fragment
    edge_lability = {(u, v): linkage_lability.get((map_to_basic(node_labels[u], obfuscate_ptm = False), d['bond_label'][1:] if
                                                   d['bond_label'][0].isalpha() else d['bond_label']), DEFAULT_LABILITY)
                     for u, v, d in nx_mono.edges(data = True) if d['bond_label'] not in ('glycosite', 'peptide')}
    downstream_values = []
    if input_dict['peptide']:
        peptide = True
        matches = [match_fragment_properties(subg_frags, observed_mass, mass_threshold, charge, sorted_frag_keys, peptide = True,
                                             mass_threshold_ppm = mass_threshold_ppm) for observed_mass in fragment_masses]
        # every matched fragment is named once (one fragment can match several observed masses; each gets its own copy of the names), each glycan's
        # cleavages for all of them in one call; one without an allowed peptide ion is dropped
        frags = list({id(g): g for m in matches for g in m[-1]}.values())
        rf_names = [peptide_to_RF_nomenclature(pep_gr, g, allowed_ion_types = allowed_ion_types, allow_internal = allow_internal_peptide_fragments) for g in frags]
        frags, rf_names = [g for g, rf in zip(frags, rf_names) if rf is not None], [rf for rf in rf_names if rf is not None]
        names = {id(g): merge_gp_global_mods([rf] + dc) for g, rf, dc in zip(frags, rf_names, get_glycan_cleavages(nx_mono, frags, input_dict['glycosites'], chain_rank))}
        lability = {id(g): compute_fragment_lability(edge_lability, g) for g in frags}
        n_sites = [input_dict['peptide'][site] == 'N' for site in input_dict['glycosites']]

        def plausible(name, z, frag):
            # As for glycopeptide compositions: in spectra of identified glycopeptides, glycan-only ions above 1+, backbone ions with part of a glycan (a
            # glycan lost whole is a Y_0/Z_0 piece; an N-glycan's core HexNAc, Y_1 cuts of all its chains, was real), X cross-rings on the intact peptide
            # other than one of an N-glycan's core HexNAc alone (peptide + 0,2X), and global modifications outside GLYCOPEPTIDE_GLOBAL_MODS of their kind
            # of ion were annotated no more often than in the same spectra shifted out of tolerance; the glycan-only ions keep all of theirs, which were real
            backbone = next(x for x in name[0] if not x.startswith('M_'))
            if backbone == 'No Peptide':
                return abs(z) == 1
            kind = 'Peptide' if backbone == 'Peptide' else 'backbone'
            if not {x[2:] for sub in name for x in sub if x.startswith('M_')} <= GLYCOPEPTIDE_GLOBAL_MODS[kind]:
                return False
            for prefix, (sub, n_site) in enumerate(zip(name[1:], n_sites), 1):
                pieces = [x for x in sub if not x.startswith('M_') and x != 'M' and not x.startswith('loss of')]
                if kind == 'Peptide':
                    if any(re.match(r'\d\dX', x) for x in pieces) and not (n_site and len(pieces) == 1 and re.fullmatch(r'\d\dX_1_[A-Za-z]+', pieces[0])):
                        return False
                elif pieces and not all(re.fullmatch(r'[YZ]_0_[A-Za-z]+', x) for x in pieces) and not (n_site and all(re.fullmatch(r'Y_1_[A-Za-z]+', x) for x in pieces) and
                                                                                                        sum(x.split('-')[0] == str(prefix) for x in frag) == 1):
                    return False
            return True

        for fragment_properties in matches:
            keep = [i for i, g in enumerate(fragment_properties[-1]) if id(g) in names and plausible(names[id(g)], fragment_properties[3][i], g)]
            fragment_properties = [[v[i] for i in keep] for v in fragment_properties]
            downstream_values.append((*fragment_properties, [[list(x) for x in names[id(g)]] for g in fragment_properties[-1]],
                                      [lability[id(g)] for g in fragment_properties[-1]]))
    else:
        peptide = False
        small = {} if disable_global_mods or parents is not None else small_ions(node_labels.values(), charge, mass_tag)
        # the open chain of a labeled reducing end gives X cross-rings of its own, also without global modifications
        if parents is None and allowed_X_cleavages:
            small |= chain_ions(node_labels[next(v for v, d in nx_mono.out_degree() if d == 0)], charge, mass_tag, sample_prep)
        # every fragment is named once, as it can match several observed masses (each gets its own copy of the names)
        graphs = [g for gs in subg_frags.values() for g in gs]
        names = dict(zip(map(id, graphs), subgraphs_to_domon_costello(nx_mono, graphs, chain_rank)))
        rings = {id(g): sum(m in A_cross_rings or m in X_cross_rings for m in g.mono_mods) for g in graphs}
        # a fragment's lability only depends on its node set
        set_lability = {id(g.nodes): g for g in graphs}
        set_lability = {k: compute_fragment_lability(edge_lability, g) for k, g in set_lability.items()}
        for observed_mass in fragment_masses:
            fragment_properties = match_fragment_properties(subg_frags, observed_mass, mass_threshold, charge,
                                                            sorted_frag_keys, mass_threshold_ppm = mass_threshold_ppm)
            if parents is not None:
                # at most ms3_cleavages cleavages the isolated fragment does not have, of which at most one X cross-ring
                keep = []
                for i, (mass, z, g) in enumerate(zip(fragment_properties[1], fragment_properties[3], fragment_properties[4])):
                    if (id(g), z) not in ms3_keep:
                        mods, cuts, ms3_keep[(id(g), z)] = dict.fromkeys(g.nodes) | dict(zip(g.terminals, g.mono_mods)), None, False
                        for p_mass, p_z, p_nodes, p_mods, p_cuts in parent_info:
                            if mass < p_mass and abs(z) <= p_z and mods.keys() <= p_nodes and all(p_mods[n] in (None, node_basic[n], m) for n, m in mods.items()):
                                cuts = fragment_cuts(g) if cuts is None else cuts
                                if len(new := cuts - p_cuts) <= ms3_cleavages and sum(len(c) == 2 and c[1] in X_cross_rings for c in new) <= 1:
                                    ms3_keep[(id(g), z)] = True
                                    break
                    if ms3_keep[(id(g), z)]:
                        keep.append(i)
                fragment_properties = [[v[i] for i in keep] for v in fragment_properties]
            else:
                # at most one ring cleavage per fragment: an A on its root plus an X on a leaf (or two X) has fewer cuts than a B/Y internal ion with a
                # neutral loss, so such readings outranked it (246.99 of TEST_DICTS[3] as 02X + 24A instead of B + Y - C2H4O2)
                keep = [i for i, g in enumerate(fragment_properties[4]) if rings[id(g)] <= 1]
                fragment_properties = [[v[i] for i in keep] for v in fragment_properties]
            dc_names = [list(names[id(sg)]) for sg in fragment_properties[-1]]
            lability = [set_lability[id(sg.nodes)] for sg in fragment_properties[-1]]
            # an ion named without a graph: one without a residue, which its fragmentation prior puts ahead of any single cleavage, or an open-chain ion
            hits = [(t, m) for t, m in small.items() if abs(observed_mass - m) <= mass_threshold and (
                mass_threshold_ppm is None or abs(observed_mass - m) <= m * mass_threshold_ppm / 1e6)]
            if hits:
                fragment_properties = [[observed_mass] * len(hits) + list(fragment_properties[0]), [m for _, m in hits] + list(fragment_properties[1]),
                                       [abs(observed_mass - m) for _, m in hits] + list(fragment_properties[2]),
                                       [int(np.sign(charge))] * len(hits) + list(fragment_properties[3]), [None] * len(hits) + list(fragment_properties[4])]
                dc_names, lability = [[t] for t, _ in hits] + dc_names, [DEFAULT_LABILITY] * len(hits) + lability
            downstream_values.append((*fragment_properties, dc_names, lability))
    chain_lengths = None if peptide else {rank: len(chain) for rank, chain in chain_rank}
    filtered_results = [priority_filter(x[5], x[2], peptide = peptide, charge = charge, lability = x[6], chain_lengths = chain_lengths) if x[0]
                        else ([], [], []) for x in downstream_values]
    filtered_dc_names = [r[0] for r in filtered_results]
    if simplify:
        filtered_diffs = [r[1] for r in filtered_results]
        filtered_lability = [r[2] for r in filtered_results]
        filtered_dc_names = simplify_fragments(filtered_dc_names, peptide = peptide, diffs = filtered_diffs,
                                               intensities = intensities, charge = charge, prior_weight = prior_weight,
                                               lability_scores = filtered_lability, mass_threshold = mass_threshold, chain_lengths = chain_lengths)
    for i, frag_dc_names in enumerate(filtered_dc_names):
        if frag_dc_names:
            filtered_properties = list(zip(*downstream_values[i]))
            final_hits = [next(y for y in filtered_properties if y[5] == x) for x in frag_dc_names[:5]]
            final_hits = [list(x) for x in zip(*final_hits)]
            hit_dict[fragment_masses[i]] = {'Theoretical fragment masses': final_hits[1],
                                            'Domon-Costello nomenclatures': final_hits[5],
                                            'Fragment charges': final_hits[3]}
            if iupac:
                hit_dict[fragment_masses[i]]['Fragment IUPAC'] = [glycopeptide_frag_to_string(pep_gr, x.graph) if peptide else None if x is None else
                                                                  mono_frag_to_string(x.graph) for x in final_hits[4]]
        else:
            hit_dict[fragment_masses[i]] = None
    if peptide and isinstance(fragmentation_method, str) and fragmentation_method in CHARGE_REDUCING_METHODS:
        add_charge_reduced_ions(hit_dict, sum(AA_masses[aa] for aa in input_dict['peptide']) + WATER_MASS + sum(
            mono_attributes[b]['mass'][b] for b in (map_to_basic(v, obfuscate_ptm = False) for v in node_labels.values() if len(v) > 1)), len(input_dict['glycans']),
                                charge, mass_threshold, mass_threshold_ppm = mass_threshold_ppm)
    return hit_dict


_PLACEMENTS = {}
_RESIDUES = {}
_EXPLAINED = {}
_CHAIN_RANKS = {}
_POOL = {}


def _run_job(job):
    func, args, kwargs = job
    return func(*args, **kwargs)


def _watch_parent():
    # A pool process ends with the process that started it; terminated (e.g., the app's worker on Cancel), it would otherwise wait for jobs forever
    parent = multiprocessing.parent_process()
    threading.Thread(target = lambda: (parent.join(), os._exit(1)), daemon = True).start()


def run_jobs(jobs, n_jobs = 1):
    """Runs (function, args, kwargs) jobs and returns their results in order; with n_jobs > 1 in a pool of that many spawned processes, which stays up for
    later calls (CandyCrumbs and supporting_ions are pure Python, so threads would only take turns, and starting the processes costs a few seconds).
    A script calling this with n_jobs > 1 needs if __name__ == '__main__' on Windows and macOS, as every pool process imports it again\n
    | Arguments:
    | :-
    | jobs (list): (function, args tuple, kwargs dict) per job; the functions have to be module-level so that they can be sent to the pool
    | n_jobs (int): number of processes; default:1 (no pool)\n
    | Returns:
    | :-
    | Returns a list of the results of the jobs
    """
    if n_jobs <= 1 or len(jobs) < 2:
        return [func(*args, **kwargs) for func, args, kwargs in jobs]
    if _POOL.get('n_jobs') != n_jobs:
        if 'pool' in _POOL:
            _POOL['pool'].shutdown()
        _POOL.update(n_jobs = n_jobs, pool = ProcessPoolExecutor(max_workers = n_jobs, mp_context = multiprocessing.get_context('spawn'), initializer = _watch_parent))
    try:
        return list(_POOL['pool'].map(_run_job, jobs))
    except BrokenProcessPool:
        _POOL.clear()
        raise


def supporting_ions(glycan, peaks, charge = -1, candidates = (), mass_tolerance = 0.3, top_n_peaks = 20, mass_tag = None, sample_prep = 'underivatized',
                    reference_glycans = None, sialic_derivatization = None):
    """Ties a glycan structure to the MS2 peaks that support it, so a prediction can be checked instead of taken at face value: for each residue, the
    intense peaks that a single glycosidic cleavage of the structure explains and that no other placement of that residue could explain with up to
    two cleavages (other placements: the residue with everything attached to it on any other residue of the structure that it is linked to in known
    glycans of the same class, at a free position); and for each competing candidate structure, the peaks only one of the two explains. On the curated
    test spectra, such peaks point to the true placement in about 85% of cases, and 43% of residues with other placements get at least one\n
    | Arguments:
    | :-
    | glycan (string): glycan in IUPAC-condensed nomenclature
    | peaks (dict): MS2 spectrum as m/z : intensity
    | charge (int): signed precursor charge; default:-1
    | candidates (list): competing structures (e.g., the other predictions) to compare the glycan with; default:()
    | mass_tolerance (float): fragment tolerance in Da; default:0.3
    | top_n_peaks (int): how many of the most intense peaks can support a residue; default:20
    | mass_tag (float): mass of the glycan label or reducing end modification, as in CandyCrumbs; default:None (alditol)
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | reference_glycans (list): known structures that define where a donor can attach; default:None (the CandyCrunch vocabulary)
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization, as in CandyCrumbs; the precursor mass then fixes how many sialic
    |                                          acids are a2-6 and how many a2-3/8/9, so placements and candidates with other counts are left out; default:None\n
    | Returns:
    | :-
    | Returns a dict with 'residues', one dict per residue with a fully specified linkage: 'residue' (index among the monosaccharides of the glycan
    | string, as GlycoDraw's per_residue), 'name' (donor, linkage and the residue it sits on), 'support' (peaks as (m/z, relative intensity, fragment
    | names, fragment IUPAC, the placements it rules out, indices of the residues in the fragment, index of the residue whose linkage was cleaved (as
    | GlycoDraw's highlight_linkages))), 'against' (peaks an alternative placement explains by one glycosidic cleavage and the
    | glycan cannot, as (m/z, relative intensity, fragment names, fragment IUPAC, placement)), 'open' (placements no peak rules out), and 'linkage'
    | (for the only sialic acid of an alditol with up to two antennae, Neu5Ac a2-3 or a2-6 on Gal, from a [M-2H]2- precursor: the singly charged
    | [M-H-Neu5Ac]- after charge separation as (m/z of its most intense peak, or of the ion if there is none, intensity relative to the base peak, the
    | linkage it reads as, whether that is the glycan's); measured on ion-trap CID only; else None); and
    | 'candidates', one dict per competing structure: 'structure', 'support', and 'against' (peaks only the glycan or only the candidate explains, by a
    | single glycosidic cleavage, as (m/z, relative intensity, fragment names, fragment IUPAC, residues in the fragment, residue of the cleaved
    | linkage), residue indices referring to the candidate for 'against')
    """
    if not isinstance(glycan, str) or '{' in glycan or not peaks:
        return {'residues': [], 'candidates': []}
    # (a pair of products as a list, e.g. from JSON settings, would not fit the cache key below)
    sialic_derivatization = tuple(sialic_derivatization) if isinstance(sialic_derivatization, list) else sialic_derivatization
    glycan_class = get_class(glycan)
    key = ('default' if reference_glycans is None else hash(tuple(reference_glycans)), glycan_class)
    if key not in _PLACEMENTS:
        if reference_glycans is None:
            with open(os.path.join(os.path.dirname(__file__), 'glycans.pkl'), 'rb') as file:
                reference_glycans = pickle.load(file)
        triples = set()
        for g in reference_glycans:
            try:
                gr = glycan_to_nxGraph(g) if '{' not in g and (not glycan_class or get_class(g) == glycan_class) else None
            except Exception:
                gr = None
            if gr is not None:
                labels = nx.get_node_attributes(gr, 'string_labels')
                triples.update((labels[next(iter(gr.succ[n]))], labels[n], labels[next(iter(gr.pred[n]))]) for n in gr if n % 2)
        _PLACEMENTS[key] = sorted(triples)
    # A glycan's residues and their other placements depend only on it and the reference glycans, and the same structures come up for many spectra
    residues = _RESIDUES.get((glycan, key))
    if residues is None:
        triples = _PLACEMENTS[key]
        gr = glycan_to_nxGraph(glycan)
        labels, paths, canonical = nx.get_node_attributes(gr, 'string_labels'), {}, graph_to_string(gr)
        for n in sorted(gr, reverse = True):
            paths[n] = paths.get(next(iter(gr.pred[n]), None), ()) + (labels[n],)
        monos, root = [p for k, p in paths.items() if k % 2 == 0], max(gr)
        def node_name(q):
            # A residue is named by its path towards the reducing end, as far as it takes to tell it apart from every other residue
            if q == root:
                return f'reducing-end {labels[q]}'
            path, d = paths[q], 1
            while d < len(path) and sum(p[-d:] == path[-d:] for p in monos) > 1:
                d += 2
            return path[-1] + ''.join(f'({path[-k]}){path[-k - 1]}' for k in range(2, d + 1, 2))
        residues = []
        for n in sorted(gr):
            if not n % 2 or '?' in labels[n] or '/' in labels[n]:
                continue
            child, parent, donor = next(iter(gr.succ[n])), next(iter(gr.pred[n])), labels[n].split('-')[0]
            placements, subtree = {}, nx.descendants(gr, n)
            for q in sorted(gr):
                if q % 2 or q in subtree:
                    continue
                used = {labels[s].split('-')[1] for s in gr.succ[q] if s != n}
                # Glycosidic fragments do not depend on the position on the new parent, so one free position this donor takes there is enough
                link = next((link for c, link, p in triples if c == labels[child] and p == labels[q] and link.startswith(donor + '-') and '?' not in link and
                             '/' not in link and link.split('-')[1] not in used), None)
                if link is None or q == parent:
                    continue
                variant = gr.copy()
                variant.remove_edge(parent, n)
                variant.add_edge(q, n)
                variant.nodes[n]['string_labels'] = link
                variant = graph_to_string(variant)
                if variant != canonical:
                    placements[variant] = f'{labels[child]} on {node_name(q)}'
            residues.append((child // 2, f'{labels[child]}({labels[n]}) on {node_name(parent)}', placements, (labels[child], labels[n], labels[parent])))
        if len(_RESIDUES) > 5000:
            _RESIDUES.clear()
        _RESIDUES[(glycan, key)] = residues
    # What explains each of the top peaks in every structure: the single glycosidic cleavages (names, IUPAC) and whether any fragment with up to two
    # cleavages (a cross-ring only alone) does. The other structures only matter at peaks the glycan explains by a single cleavage (could they
    # explain it too?) or not at all (do they?), so only those are searched, and the first fragment settles the former
    top = sorted(peaks.items(), key = lambda x: -x[1])[:top_n_peaks]
    observed, explained, alternatives = [(float(mz), intensity / (top[0][1] or 1)) for mz, intensity in top], {}, {v for _, _, p, _ in residues for v in p}
    for g in [glycan] + sorted(alternatives | {c for c in candidates if isinstance(c, str) and '{' not in c and c != glycan}):
        if glycan not in explained and g != glycan:
            break
        relevant = range(len(observed)) if g == glycan else [i for i, (single, any_fragment) in explained[glycan].items() if single or not any_fragment]
        # What a structure explains only depends on it, the masses asked for and the settings, and about a quarter of these come up again within a run
        # (the isomer override and finalise_predictions look at the same spectra, and placements and candidates of related structures overlap)
        needed = tuple(g == glycan or not explained[glycan][i][1] for i in relevant)
        key = (g, glycan if sialic_derivatization is not None else None, charge, mass_tolerance, mass_tag, sample_prep, sialic_derivatization,
               g in alternatives, tuple(observed[i][0] for i in relevant), needed)
        found = _EXPLAINED.get(key, False)
        if found is False:
            if len(_EXPLAINED) > 20000:
                _EXPLAINED.clear()
            try:
                if sialic_derivatization is not None and g != glycan and glycan_to_composition(g, sialic_linkage = True) != glycan_to_composition(glycan, sialic_linkage = True):
                    raise ValueError('other sialic acid linkages than the precursor mass allows')
                structure = g if sialic_derivatization is None else derivatize_carboxyls(g, sialic_derivatization)
                nx_mono, _ = input_to_graph(glycopeptide_string_to_input(structure))
                chain_rank = _CHAIN_RANKS.get(structure)
                if chain_rank is None:
                    if len(_CHAIN_RANKS) > 5000:
                        _CHAIN_RANKS.clear()
                    chain_rank = _CHAIN_RANKS[structure] = list(rank_chains(nx_mono))
                global_mods, special_residues = get_initial_global_mods(nx_mono, charge, disable_global_mods = True)
                subg_frags = generate_atomic_frags(nx_mono, global_mods, special_residues, [], max_cleavages = 2, fragment_masses = [observed[i][0] for i in relevant],
                                                   threshold = mass_tolerance, mass_tag = mass_tag, charge = charge, sample_prep = sample_prep,
                                                   disable_A_cross_rings = charge > 0 or g in alternatives) if relevant else {}
            except Exception:
                _EXPLAINED[key] = None
                continue
            frag_keys, found, names = sorted(subg_frags), [], None
            for i, names_needed in zip(relevant, needed):
                single, any_fragment = [], False
                for z in range(1, abs(charge) + 1):
                    target = observed[i][0] * z - (z - 1) * PROTON_MASS * (1 if charge > 0 else -1)
                    for mass in frag_keys[bisect.bisect_left(frag_keys, target - mass_tolerance):bisect.bisect_right(frag_keys, target + mass_tolerance)]:
                        if names is None:
                            # every fragment is named once, at the first that is needed
                            graphs = [x for gs in subg_frags.values() for x in gs]
                            names = dict(zip(map(id, graphs), subgraphs_to_domon_costello(nx_mono, graphs, chain_rank)))
                        for frag in subg_frags[mass]:
                            if z > frag.number_of_nodes():
                                continue
                            types = [x.split('_')[0] for x in names[id(frag)]]
                            if not types or (len(types) > 1 and any(t[-1] in 'AX' for t in types)):
                                continue
                            any_fragment = True
                            if names_needed and len(types) == 1 and types[0] in ('B', 'C', 'Y', 'Z'):
                                single.append((tuple(names[id(frag)]), mono_frag_to_string(frag.graph), tuple(sorted(frag)),
                                               next((u for u, v in nx_mono.edges() if (u in frag) != (v in frag)), None)))
                        if any_fragment and not names_needed:
                            break
                found.append((tuple(single), any_fragment))
            found = _EXPLAINED[key] = tuple(found)
        if found is None:
            continue
        explained[g] = dict(zip(relevant, found))
    if glycan not in explained:
        return {'residues': [], 'candidates': []}
    out = {'residues': [], 'candidates': []}
    # The linkage of a lone Neu5Ac on Gal of an alditol, read from a [M-2H]2- precursor in ion-trap CID: after charge separation the singly charged
    # [M-H-Neu5Ac]- reaches 15% of the base peak for a2-3 and stays below it for a2-6 (ion-trap CID, 25 monosialylated isomer pairs: 44 of 50 structures,
    # 25 of 27 projects with both isomers in that direction; curated test GTs 52/60 rows on Thermo, 58/63 on Bruker traps). It fails on free reducing
    # ends, Neu5Gc, and three or more antennae, and HCD has no labeled data; an intensity reading, so it is reported beside the peaks and never counted as support
    sialic = sum(v for k, v in glycan_to_composition(glycan).items() if k in ('Neu5Ac', 'Neu5Gc', 'Kdn')) if charge == -2 and (mass_tag is None or abs(mass_tag - 2 * HYDROGEN_MASS) < 1e-3) and sample_prep == 'underivatized' and sialic_derivatization is None else 0
    for residue, name, placements, link in residues:
        linkage = None
        if sialic == 1 and link[0] == 'Neu5Ac' and link[1] in ('a2-3', 'a2-6') and link[2] == 'Gal':
            gr = glycan_to_nxGraph(glycan)
            target = get_fragment_mass(glycan, 'M', charge = -1) - composition_to_mass({'Neu5Ac': 1}) + WATER_MASS
            # antennae as non-reducing ends other than Fuc; a scan that stops below the ion cannot read it
            if sum(gr.out_degree(n) == 0 and gr.nodes[n]['string_labels'] != 'Fuc' for n in gr if n % 2 == 0) <= 2 and max(peaks) >= target - mass_tolerance:
                intensity, mz = max(((i, mz) for mz, i in peaks.items() if abs(mz - target) <= mass_tolerance), default = (0, target))
                rel = intensity / (max(peaks.values()) or 1)
                linkage = (round(float(mz), 2), round(rel, 3), 'a2-3' if rel >= 0.15 else 'a2-6', (rel >= 0.15) == (link[1] == 'a2-3'))
        placements = {v: p for v, p in placements.items() if v in explained}
        support, against, ruled_out = [], [], set()
        for i, (mz, rel) in enumerate(observed):
            beaten = [p for v, p in placements.items() if not explained[v][i][1]] if explained[glycan][i][0] else []
            # A fragment counts once, at its most intense peak
            if beaten and explained[glycan][i][0][0][0] not in [s[2] for s in support]:
                support.append((round(mz, 2), round(rel, 3)) + explained[glycan][i][0][0][:2] + (beaten,) + explained[glycan][i][0][0][2:])
            ruled_out.update(beaten)
            if not explained[glycan][i][1]:
                against += [(round(mz, 2), round(rel, 3), explained[v][i][0][0][0], explained[v][i][0][0][1], p) for v, p in placements.items() if explained[v][i][0]]
        out['residues'].append({'residue': residue, 'name': name, 'support': support, 'against': against,
                                'open': [p for p in placements.values() if p not in ruled_out], 'linkage': linkage})
    for c in candidates:
        if c in explained and c != glycan:
            out['candidates'].append({'structure': c, 'support': [(round(mz, 2), round(rel, 3)) + explained[glycan][i][0][0] for i, (mz, rel) in enumerate(observed)
                                                               if explained[glycan][i][0] and not explained[c][i][1]],
                                      'against': [(round(mz, 2), round(rel, 3)) + explained[c][i][0][0] for i, (mz, rel) in enumerate(observed)
                                                  if not explained[glycan][i][1] and explained[c][i][0]]})
    return out


def rank_glycopeptide_structures(peptide, modification_str, fragment_masses, intensities = None, charge = 2, structures = None,
                                 fragmentation_method = None, mass_threshold = 0.2, mass_threshold_ppm = 10, kingdom = 'Animalia',
                                 top_n_peaks = 150, max_candidates = 100, o_glycan_root = 'GalNAc', hexnac_penalty = 0.5, **kwargs):
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
    | mass_threshold (float): fragment mass tolerance in Da, the tolerance of ion-trap spectra; default:0.2
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; None for low-resolution spectra; default:10
    | kingdom (string): taxonomic kingdom of the database structures; default:'Animalia'
    | top_n_peaks (int): how many of the most intense peaks are scored; default:150
    | max_candidates (int): most candidates (structure combinations, for several glycans) to score, at ~1 s each for N-glycans; default:100
    | o_glycan_root (string): reducing-end residue the database candidates of an O-site need (GalNAc: mucin-type); a composition without such a candidate
    |                         keeps all; default:'GalNAc' (None for O-GlcNAc, O-Man or O-Fuc glycopeptides)
    | hexnac_penalty (float): factor on the score of candidates the HexNAc oxonium ions contradict (see 'oxonium_ratio'); default:0.5 (1 to ignore them)
    | **kwargs: passed on to CandyCrumbs, e.g., max_cleavages\n
    | Returns:
    | :-
    | Returns a dataframe with one row per candidate ('structures', one per glycan; candidates with residues CandyCrumbs has no tables for are left out),
    | best first, with 'score' (the product of the next three), 'rank' (candidates with identical fragment masses, such as linkage isomers, tie), 'explained'
    | (share of square-root intensity that CandyCrumbs annotates, weighted by the fragmentation prior of each annotation), 'direct_fragments' (share of the
    | candidate's single-cleavage fragments observed), 'hexnac_evidence' (hexnac_penalty for a candidate with GlcNAc when the oxonium ratio is below 1 and
    | for one without when it is above 2, among candidates that differ in having GlcNAc, else 1), 'oxonium_ratio' ((138 + 168) / (126 + 144) of the HexNAc
    | oxonium ions, None without 126 and 144), and 'missing_fragments' (singly protonated m/z of those the spectrum lacks at every charge)
    """
    if not isinstance(fragment_masses, dict):
        fragment_masses = dict(zip(fragment_masses, [1.0] * len(fragment_masses) if intensities is None else intensities))
    if not fragment_masses:
        raise ValueError("No peaks to score")
    # The HexNAc oxonium ions of GlcNAc and GalNAc differ in their ring fragments: (138 + 168) / (126 + 144) was below 1 in 85% of the collisional spectra
    # of one-site O-glycopeptides with one HexNAc (GalNAc, median 0.56) and in 0.9% of N-glycopeptides (GlcNAc, median 3.14), above 2 in 77% of these and
    # 2% of those; electron-transfer spectra rarely record these ions
    mzs, ints = np.array(list(fragment_masses), dtype = float), np.array(list(fragment_masses.values()), dtype = float)
    ox = [ints[np.abs(mzs - m) <= (min(mass_threshold, m * mass_threshold_ppm / 1e6) if mass_threshold_ppm else mass_threshold)].sum()
          for m in (OXONIUM_IONS[k] for k in ('HexNAc-C2H4O2-H2O', 'HexNAc-CH2O-2H2O', 'HexNAc-C2H4O2', 'HexNAc-2H2O'))]
    oxonium_ratio = float((ox[1] + ox[3]) / (ox[0] + ox[2])) if ox[0] + ox[2] > 0 else None
    hexnac = None if oxonium_ratio is None else 'GalNAc' if oxonium_ratio < 1 else 'GlcNAc' if oxonium_ratio > 2 else None
    # An accurate spectrum is scored deisotoped, as the app annotates it: the 13C peaks of the oxonium and Y ions are no fragments of their own, but took
    # chance labels from candidates with more residue kinds (ETD runs: database structures of every composition within 10 ppm and -1 to +2
    # isotopes of 516 spectra, the Byonic glycan ranked first 0.576 -> 0.584; CandyCrumbs told it from its isobar Neu5Gc + Fuc in HCD 0.45 -> 0.67)
    if mass_threshold_ppm:
        fragment_masses = deisotope_ms2(fragment_masses, max(abs(charge), 1), min(mass_threshold, 0.05))
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
            # mucin-type O-glycans start with GalNAc; the database's other O-site candidates (Gal-rooted glycolipid-like ones, O-Man, O-Fuc, O-GlcNAc) were 7%
            # of those of the compositions of O-glycoprotease digests
            if o_glycan_root and peptide[site] != 'N':
                candidates = [g for g in candidates if re.sub(r'^.*[)\]]', '', g).startswith(o_glycan_root)] or candidates
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
        explained = sum(weights[mz] * score_gp_prior(dc if 'No Peptide' in dc[0] else
                                                     [[y for y in sub if not re.fullmatch(r'[YZ]_\d+_[A-Za-z]+', y)] or ['M'] for sub in dc], charge)
                        for mz, dc in annotations.items()) / sum(weights.values())
        direct = np.mean(found) if found else 1.0
        rows.append([list(combo), explained * direct, explained, direct, sorted({round(float(mzs[0]), 4) for mzs, f in zip(expected, found) if not f})])
    if not rows:
        raise ValueError("None of the candidate structures consists of residues CandyCrumbs can fragment")
    # a generic HexNAc could be either, so the ratio only weighs candidates whose HexNAc are named
    glcnac = [None if any(re.search(r'HexNAc(?![a-z])', g) for g in row[0]) else any('GlcNAc' in g for g in row[0]) for row in rows]
    weighable = {x for x in glcnac if x is not None} == {True, False} and hexnac is not None
    for row, has_glcnac in zip(rows, glcnac):
        factor = hexnac_penalty if weighable and has_glcnac is not None and has_glcnac == (hexnac == 'GalNAc') else 1.0
        row[1] *= factor
        row[4:4] = [factor, oxonium_ratio]
    df_out = pd.DataFrame(rows, columns = ['structures', 'score', 'explained', 'direct_fragments', 'hexnac_evidence', 'oxonium_ratio', 'missing_fragments'])
    df_out = df_out.sort_values('score', ascending = False, kind = 'stable').reset_index(drop = True)
    df_out.insert(2, 'rank', df_out['score'].round(10).rank(ascending = False, method = 'min').astype(int))
    return df_out


def rank_glycopeptide_compositions(peptide, compositions, fragment_masses, intensities = None, charge = 2, glycosite = None, isotope_errors = None,
                                   fragmentation_method = None, mass_threshold = 0.2, mass_threshold_ppm = 10, top_n_peaks = 100, isotope_penalty = 0.2, **kwargs):
    """Ranks the glycan compositions that fit a glycopeptide's precursor (isobars such as Neu5Ac + Hex and Neu5Gc + Fuc, or compositions at other isotope
    errors of the precursor) by their fragment evidence in its MS2 spectrum\n
    | Arguments:
    | :-
    | peptide (string): amino acid sequence, in CandyCrunch's one-letter code (c, j, k, m, n, q for modified residues)
    | compositions (list): candidate compositions of the peptide's one glycan, as dicts or strings ('H5N4F1A1', 'Hex5HexNAc4Neu5Ac1')
    | fragment_masses (list or dict): observed m/z values, or a peak dictionary of m/z : intensity such as the peak_d of load_spectra_filepath
    | intensities (list): intensities of the m/z values if fragment_masses is a list; default:None (all equal)
    | charge (int): precursor charge state; default:2
    | glycosite (int): 0-based position of the glycosylated residue; default:None (the first site infer_glycosites finds). A peptide with several glycans
    |                  is ranked by its total composition on one of its sites: on 701 multi-site O-glycopeptide spectra that ranked the search engine's
    |                  composition first as often as scoring each candidate by its best distribution over the identified sites (two sites: ion-trap ETD
    |                  0.838 vs 0.848, EThcD 0.954 vs 0.949; three or four: equal), which costs one CandyCrumbs call per distribution (median 31, up to 1,620)
    | isotope_errors (list): per composition, how many isotope steps the precursor m/z is off for it (the m/z the instrument picked minus the composition's,
    |                        in 13C spacings); default:None (all 0)
    | fragmentation_method (string): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD', e.g., the activation column of load_spectra_filepath; default:None (all peptide ion types)
    | mass_threshold (float): fragment mass tolerance in Da, the tolerance of ion-trap spectra (at 0.5 Da a candidate one isotope heavier explains their
    |                         13C peaks and the 2+ ions); default:0.2
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; None for low-resolution spectra; default:10
    | top_n_peaks (int): how many of the most intense peaks are scored; default:100
    | isotope_penalty (float): what each isotope step of isotope_errors costs a candidate's score; default:0.2
    | **kwargs: passed on to CandyCrumbs; default max_cleavages 2\n
    | Returns:
    | :-
    | Returns a dataframe with one row per candidate, best first: 'composition', 'isotope_error', 'score' (explained x sialic_evidence, minus isotope_penalty
    | per isotope step), 'rank' (equal scores tie), 'explained' (share of the square-root intensity CandyCrumbs annotates), and 'sialic_evidence' (share of the candidate's
    | sialic acid kinds whose oxonium ion, B1 or B1 - H2O, is a peak; 1 without sialic acids)
    """
    if not isinstance(fragment_masses, dict):
        fragment_masses = dict(zip(fragment_masses, [1.0] * len(fragment_masses) if intensities is None else intensities))
    if not fragment_masses:
        raise ValueError("No peaks to score")
    # Accurate spectra are scored deisotoped, as the app annotates them: the 13C peaks of the oxonium and Y ions took chance labels from candidates with more
    # residue kinds; in ion-trap spectra deisotoping cost more than it gained
    if mass_threshold_ppm:
        fragment_masses = deisotope_ms2(fragment_masses, max(abs(charge), 1), min(mass_threshold, 0.05))
    peaks = sorted(fragment_masses.items(), key = lambda x: -x[1])[:top_n_peaks]
    weights = np.sqrt(np.maximum([i for _, i in peaks], 0)) if peaks[0][1] > 0 else np.ones(len(peaks))
    observed = np.array([m for m, _ in peaks])
    isotope_errors = [0] * len(compositions) if isotope_errors is None else list(isotope_errors)
    kwargs.setdefault('max_cleavages', 2)
    rows = []
    for comp, isotope_error in zip(compositions, isotope_errors):
        comp = canonicalize_composition(comp) if isinstance(comp, str) else dict(comp)
        hits = CandyCrumbs({'peptide': peptide, 'glycans': [comp], 'glycosites': [] if glycosite is None else [glycosite]}, list(observed), mass_threshold,
                           charge = abs(charge), intensities = [i for _, i in peaks], fragmentation_method = fragmentation_method,
                           mass_threshold_ppm = mass_threshold_ppm, **kwargs)
        explained = float(weights[[hits.get(m) is not None for m in observed]].sum() / weights.sum())
        # The explained share alone rewards compositions with more residue kinds (more fragments, more chance matches) and knows no missing ion: in
        # glycopeptide runs with search-engine identifications it chose Neu5Gc + Fuc over the identified glycan's isobar Neu5Ac + Hex in 55% (HCD) to 74% (EThcD) of spectra, while every
        # sialic acid kind there showed its oxonium ion (Neu5Ac 292/274 in 97% of HCD, 56% of EThcD and 85% of ion-trap ETD spectra of sialylated
        # glycopeptides, 1% of the others; Neu5Gc's never). Weighted by it, the identified glycan won 0.98 (HCD), 0.77 (EThcD) and 0.93 (ion-trap ETD) of
        # these isobars; Fuc and Hex oxonium ions are too rare (0 and 64% in HCD) to count. With candidates from the reader's precursor m/z at isotope errors
        # -1..+2 (1,206 spectra, 51 of them still on a 13C peak) the identified glycan came first in 0.948, against 0.890 by the isotope error alone and 0.865 by
        # explained share alone; the isotope penalty scanned 0-0.3 (0.05: 0.911, 0.1: 0.931, 0.2: 0.948, 0.3: 0.953, the 13C-picked spectra 0.745 from 0.1 on)
        sialic = [r for r in ('Neu5Ac', 'Neu5Gc') if comp.get(r)]
        found = [check_masses(observed, np.array([mono_attributes[r]['mass'][r] + PROTON_MASS - d for d in (0, WATER_MASS)]), np.minimum(
            mass_threshold, (mono_attributes[r]['mass'][r] + PROTON_MASS) * mass_threshold_ppm / 1e6) if mass_threshold_ppm else mass_threshold).any() for r in sialic]
        evidence = float(np.mean(found)) if sialic else 1.0
        rows.append([comp, isotope_error, explained * evidence - isotope_penalty * abs(isotope_error), explained, evidence])
    df_out = pd.DataFrame(rows, columns = ['composition', 'isotope_error', 'score', 'explained', 'sialic_evidence'])
    df_out = df_out.sort_values('score', ascending = False, kind = 'stable').reset_index(drop = True)
    df_out.insert(3, 'rank', df_out['score'].round(10).rank(ascending = False, method = 'min').astype(int))
    return df_out


# O-glycoproteases that cut N-terminal to a glycosylated Ser/Thr, so that their peptides carry a glycan on the first residue
N_TERMINAL_O_GLYCOPROTEASES = {'IMPa', 'OgpA', 'OpeRATOR', 'SmE'}


def rank_glycopeptide_sites(peptide, glycans, fragment_masses, intensities = None, charge = 2, candidate_sites = None, glycan_class = None,
                            fragmentation_method = None, mass_threshold = 0.2, mass_threshold_ppm = 10, top_n_peaks = 100, protease = None, prior_weight = 0.1,
                            max_placements = 200, **kwargs):
    """Ranks the placements of a glycopeptide's glycans on its candidate sites by the site-determining ions of its MS2 spectrum, the backbone ions (c/z with
    electron transfer, b/y with collisional activation) that carry every glycan whose site they cover, and optionally by the cleavage rule of the protease\n
    | Arguments:
    | :-
    | peptide (string): amino acid sequence, in CandyCrunch's one-letter code (c, j, k, m, n, q for modified residues)
    | glycans (list): the glycans to place, one per occupied site, as compositions (dicts or strings, 'H1N1A1') or IUPAC-condensed structures
    | fragment_masses (list or dict): observed m/z values, or a peak dictionary of m/z : intensity such as the peak_d of load_spectra_filepath
    | intensities (list): intensities of the m/z values if fragment_masses is a list; default:None (all equal)
    | charge (int): precursor charge state; default:2
    | candidate_sites (list): 0-based positions the glycans can occupy; default:None (those infer_glycosites finds for glycan_class)
    | glycan_class (string): 'N' or 'O', for the default candidate sites; default:None (both)
    | fragmentation_method (string): 'CID'/'HCD'/'ETD'/'ECD'/'EThcD'/'ETciD', e.g., the activation column of load_spectra_filepath; default:None (all peptide ion types)
    | mass_threshold (float): fragment mass tolerance in Da, the tolerance of ion-trap spectra; default:0.2
    | mass_threshold_ppm (float): relative tolerance in ppm, applied on top of mass_threshold; None for low-resolution spectra; default:10
    | top_n_peaks (int): how many of the most intense peaks are scored; default:100
    | protease (string): the protease that made the peptide; with one of N_TERMINAL_O_GLYCOPROTEASES (IMPa, OgpA/OpeRATOR, SmE), a placement with a glycan
    |                    on the first residue gains prior_weight; default:None (no prior)
    | prior_weight (float): what the protease prior adds to a placement's score; default:0.1
    | max_placements (int): most placements (distinct assignments of the glycans to sites) to score, at one CandyCrumbs call each; default:200
    | **kwargs: passed on to CandyCrumbs; default max_cleavages 2\n
    | Returns:
    | :-
    | Returns a dataframe with one row per placement, best first: 'glycosites' (0-based positions), 'glycans' (in the order of glycosites), 'score' (site_ions
    | plus prior), 'rank' (equal scores tie), 'site_ions' (share of the square-root intensity annotated as backbone ions carrying every glycan whose site
    | they cover), and 'prior'
    """
    # Every placement of the identified glycans of 1,640 fully localized O-glycopeptide spectra with more Ser/Thr than glycans: the search engine's placement
    # ranked first in 0.65 (ion-trap ETD) and 0.68 (EThcD) of O-glycoprotease peptides by these site ions, 0.80/0.82 of tryptic ones (chance 0.35/0.44;
    # any backbone ion 0.54/0.58 and 0.72/0.70, as one that lost a glycan fits every placement; the annotated share of all peaks 0.52/0.58 and 0.67/0.70).
    # HCD ranked O-glycoprotease peptides at chance (0.42) and tryptic ones at 0.59. The search engine put the glycan on the first residue of 72-80% of IMPa
    # and SmE peptides; that prior alone ranked 0.64/0.63, site ions plus a prior of 0.1 0.82/0.82 (0.05: 0.81/0.81, 0.2 as 0.1); StcE, which does not cut
    # at the glycosite, gets none (a prior of 0.2 ranked its peptides worse)
    if not isinstance(fragment_masses, dict):
        fragment_masses = dict(zip(fragment_masses, [1.0] * len(fragment_masses) if intensities is None else intensities))
    if not fragment_masses:
        raise ValueError("No peaks to score")
    if not glycans:
        raise ValueError("No glycans to place")
    sites = sorted(infer_glycosites(peptide, glycan_class = glycan_class) if candidate_sites is None else candidate_sites)
    if len(sites) < len(glycans):
        raise ValueError(f"{len(glycans)} glycans do not fit on the {len(sites)} candidate sites of {peptide}")
    # a composition is placed by its content, so that equal glycans swapped between sites are one placement
    keys = [str(sorted((g if isinstance(g, dict) else canonicalize_composition(g)).items())) if isinstance(g, dict) or is_composition(g) else g for g in glycans]
    by_key = dict(zip(keys, glycans))
    n_placements = math.comb(len(sites), len(glycans)) * math.factorial(len(glycans)) // math.prod(math.factorial(n) for n in Counter(keys).values())
    if n_placements > max_placements:
        raise ValueError(f"{n_placements} placements exceed max_placements = {max_placements}; pass fewer candidate_sites or raise max_placements")
    placements = sorted({tuple(sorted(zip(pos, [keys[k] for k in perm]))) for pos in combinations(sites, len(glycans)) for perm in permutations(range(len(glycans)))})
    # Accurate spectra are scored deisotoped, as the app annotates them; in ion-trap spectra deisotoping cost more than it gained
    if mass_threshold_ppm:
        fragment_masses = deisotope_ms2(fragment_masses, max(abs(charge), 1), mass_tolerance = min(mass_threshold, 0.05))
    peaks = sorted(fragment_masses.items(), key = lambda x: -x[1])[:top_n_peaks]
    weights = np.sqrt(np.maximum([i for _, i in peaks], 0)) if peaks[0][1] > 0 else np.ones(len(peaks))
    observed = [m for m, _ in peaks]
    kwargs.setdefault('max_cleavages', 2)
    rows = []
    for placement in placements:
        sites_of = [s for s, _ in placement]
        hits = CandyCrumbs({'peptide': peptide, 'glycans': [by_key[k] for _, k in placement], 'glycosites': sites_of}, observed, mass_threshold = mass_threshold,
                           charge = abs(charge), intensities = [i for _, i in peaks], fragmentation_method = fragmentation_method,
                           mass_threshold_ppm = mass_threshold_ppm, **kwargs)
        retained = np.zeros(len(observed), dtype = bool)
        for k, m in enumerate(observed):
            if hits.get(m) is None:
                continue
            name = hits[m]['Domon-Costello nomenclatures'][0]
            ion = re.fullmatch(r'(.+)_(\d+)', next(t for t in name[0] if not t.startswith('M_')))
            if ion is None:
                continue
            # a backbone ion that covers a glycan's site is site evidence when it carries that glycan; one that lost it (a loss of the glycan, or a Y_0/Z_0 piece of
            # a structure) fits every placement, as a glycan outside the ion is named the same way
            covered = [j for j, s in enumerate(sites_of) if (s < int(ion[2]) if ion[1] in N_TERM_IONS else s >= len(peptide) - int(ion[2]))]
            retained[k] = not any(name[1 + j][0].startswith('loss of') or re.fullmatch(r'[YZ]_0_[A-Za-z]+', name[1 + j][0]) for j in covered)
        site_ions = float(weights[retained].sum() / weights.sum())
        prior = prior_weight if protease in N_TERMINAL_O_GLYCOPROTEASES and placement[0][0] == 0 else 0.0
        rows.append([sites_of, [by_key[k] for _, k in placement], site_ions + prior, site_ions, prior])
    df_out = pd.DataFrame(rows, columns = ['glycosites', 'glycans', 'score', 'site_ions', 'prior'])
    df_out = df_out.sort_values('score', ascending = False, kind = 'stable').reset_index(drop = True)
    df_out.insert(3, 'rank', df_out['score'].round(10).rank(ascending = False, method = 'min').astype(int))
    return df_out


def get_fragment_mass(glycan, fragment, charge = -1, mass_tag = None, sample_prep = 'underivatized', max_cleavages = 3, sialic_derivatization = None):
    """Calculates the theoretical m/z of a named fragment; the inverse of CandyCrumbs\n
    | Arguments:
    | :-
    | glycan (string): glycan in IUPAC-condensed format
    | fragment (string/list): fragment in Domon-Costello nomenclature, either compact ("0,2A5a", "B1a/M-H2O", "Y1a/M+H", "M+2Na", "HSO4") or as the list
    |                         CandyCrumbs returns (['02A_5_Alpha'], ['Y_1_Alpha', 'M_+H'], ['I_HSO4'])
    | charge (int): charge state of the fragment ion, sign sets the ion mode; default:-1
    | mass_tag (float): mass of the glycan label or reducing end modification; default:None (alditol, 2 H)
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated
    | max_cleavages (int): maximum number of allowed concurrent fragmentations; default:3
    | sialic_derivatization (string or tuple): linkage-specific sialic acid derivatization, as in CandyCrumbs; default:None\n
    | Returns:
    | :-
    | Returns the m/z of the fragment, or None if it cannot exist on this glycan
    """
    if sialic_derivatization is not None:
        if sample_prep in ('permethylated', 'deuteropermethylated') and 'lactone' in (SIALIC_DERIVATIZATIONS.get(sialic_derivatization, ()) if isinstance(sialic_derivatization, str) else sialic_derivatization):
            raise ValueError(f"A lactone does not survive permethylation; give the product its sialic acids end up as, e.g., ('dimethylamide', 'methyl ester') instead of {sialic_derivatization!r}.")
        glycan = derivatize_carboxyls(glycan, sialic_derivatization)
    if isinstance(fragment, str):
        cuts = []
        # a + separates fragment names, except in front of an adduct (M+Na, M+2Na, or M_+Na as CandyCrumbs writes it) or a hydrogen atom (M+H)
        for part in re.split(r'[/;]|\+(?!\d*(?:' + '|'.join(x[1:] for x in ADDUCT_GLOBAL_MODS | HYDROGEN_ATOMS) + r')(?![a-z]))', fragment.replace(' ', '')):
            if part.removeprefix('I_') in SMALL_IONS:
                cuts.append('I_' + part.removeprefix('I_'))
                continue
            if part[0] in 'Mm':
                # a compact modification (M-H2O, M+2Na, M-H) becomes CandyCrumbs' token (H2O, 2+Na, -H), and one written as that token (M_2+Na) stays
                terms = re.findall(r'([+-])(\d*)([A-Z][A-Za-z0-9]*)', part[1:])
                if part[1:2] == '_' or not terms:
                    mod = part[2:] if part[1:2] == '_' else part[1:].lstrip('-')
                else:
                    mod = combine_global_mods([sign + name if sign == '+' or name == 'H' else name for sign, count, name in terms for _ in range(int(count or 1))])
                if mod:
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
    if any(c.startswith('I_') for c in fragment):
        # an ion named without a graph (HSO4-, a label's reporter or open-chain ion) stands alone, if the glycan gives it
        labels = nx.get_node_attributes(nx_mono, 'string_labels')
        root_label = labels[next(v for v, d in nx_mono.out_degree() if d == 0)]
        return (small_ions(labels.values(), charge, mass_tag) | chain_ions(root_label, charge, mass_tag, sample_prep)).get(fragment[0]) if len(fragment) == 1 else None
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
            retained_atoms = mono_attributes[map_to_basic(node_dict[node], obfuscate_ptm = False)]['atoms'].get(cut_type)
            # a cross-ring this residue has no table entry for (0,4A of Fuc) cannot exist on it
            if retained_atoms is None:
                return None
            for child, _ in nx_mono.in_edges(node):
                if (pos := bond_labels[(child, node)][-1]).isdigit() and int(pos) not in retained_atoms:
                    keep -= nx.ancestors(nx_mono, child) | {child}
        elif cut_type[-1] in {'B', 'C'}:
            keep &= nx.ancestors(nx_mono, post_mono) | {post_mono}
        elif cut_type[-1] == 'X':
            retained_atoms = mono_attributes[map_to_basic(node_dict[node], obfuscate_ptm = False)]['atoms'].get(cut_type)
            if retained_atoms is None:
                return None
            # an X cross-ring keeps the children on the ring atoms it retains, unless a cleavage of their own takes them
            for child, _ in nx_mono.in_edges(node):
                if not ((pos := bond_labels[(child, node)][-1]).isdigit() and int(pos) in retained_atoms):
                    keep -= nx.ancestors(nx_mono, child) | {child}
        else:
            keep -= nx.ancestors(nx_mono, node) | {node}
    if not keep:
        return None
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge, disable_global_mods = global_mod is None,
                                                            max_global_mods = 2, sample_prep = sample_prep,
                                                            hydrogen_shifts = bool(HYDROGEN_ATOMS & set(parse_global_mod(global_mod))))
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
    # This is the criterion used to match nodes (it can also be something more general, i.e., Hex, HexNAc etc)
    nm = iso.categorical_node_match("string_labels", None)
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
        all_unique_graphs1.extend(g for x, g in enumerate(first_graphs) if x not in graphs1)
        all_unique_graphs2.extend(g for x, g in enumerate(second_graphs) if x not in graphs2)
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
    | Returns a dictionary of form glycan : averaged intensities, after plotting them (the second glycan mirrored below the axis)
    """
    out_dic = {}
    for g in glycan_list:
        out_dic[g] = [sum(col) / len(col) for col in
                      zip(*df_sub[df_sub.Prediction == g].binned_intensities.values.tolist())]
        # a glycan without spectra (in this confidence bracket) gets an empty line, which keeps its legend entry
        if len(glycan_list) == 2 and glycan_list.index(g) == 1:
            plt.plot(mz_range[:len(out_dic[g])], list(map(neg, out_dic[g]))[:num_bins_plot])
        else:
            plt.plot(mz_range[:len(out_dic[g])], out_dic[g][:num_bins_plot])
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
        if not len(df_a2) or not len(df_b2):
            # a confidence bracket without spectra of one of the glycans has no significant bins
            for k in bin_keys:
                bins[k].append(0)
            continue
        cohensd = [cohen_d(df_a2[:, k], df_b2[:, k]) for k in range(max_bin)]
        pvals = np.array([ttest_ind(df_a2[:, k], df_b2[:, k], equal_var = False)[1] for k in range(max_bin)])
        tested = ~np.isnan(pvals)
        pvals[tested] = correct_multiple_testing(pvals[tested], 0.05)[0]
        for c, cd in enumerate(cohensd):
            bins[bin_keys[c]].append(cd[0] if pvals[c] < 0.05 else 0)
    bins = {k: v for k, v in bins.items() if
            max(abs(v2) for v2 in v) >= thresh and not any(math.isinf(v2) for v2 in v)}
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
        elif len(parts) == 2 and parts[0] in N_TERM_IONS | C_TERM_IONS:
            mpl_parts.append(f"${parts[0]}_{{{parts[1]}}}$")
        elif len(parts) == 2 and parts[0] == 'I':
            mpl_parts.append(SMALL_IONS[parts[1]][2])
        elif len(parts) >= 2 and parts[0] == 'M':
            # Adducts (+Na) and hydrogen atoms (+H) are gains, everything else a loss; element counts become subscripts, a radical a superscript bullet
            terms = ''.join(f" {sign} {term}" for sign, term in global_mod_terms('_'.join(parts[1:])))
            mpl_parts.append('$M' + re.sub(r'(?<=[A-Za-z])(\d+)', r'_{\1}', terms).replace('\u2022', r'^{\bullet}') + '$')
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
DIAGNOSTIC_COLOR = '#8c5bb5'


def identify_oxonium(mz, tolerance_ppm = 20):
    """Returns the conventional name of a diagnostic oxonium ion at this m/z, or None"""
    for name, theoretical in OXONIUM_IONS.items():
        if abs(mz - theoretical) <= theoretical * tolerance_ppm / 1e6:
            return name
    return None


def resolve_spectrum_input(input_string):
    """Splits any CandyCrumbs input into its peptide, glycan and glycosite components"""
    if isinstance(input_string, dict) and 'peptide' in input_string:
        glycans, sites = list(input_string.get('glycans', [])), list(input_string.get('glycosites', []))
        # in site order, as CandyCrumbs numbers them
        if None not in sites and len(sites) == len(glycans) > 1:
            sites, glycans = map(list, zip(*sorted(zip(sites, glycans), key = lambda x: x[0])))
        return input_string['peptide'], glycans, sites
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
    backbone = [x[0] for x in peptide_part if re.fullmatch(r'[abcwxyz](?:-1|\+1)?_\d+', x)]
    if backbone:
        return 'backbone_n' if backbone[0] in N_TERM_IONS else 'backbone_c'
    return 'glycopeptide'


_FRAG_IUPAC = {}


def fragment_cartoon(glycans, dc_name):
    """The IUPAC-condensed fragment a glycan-only ion's cartoon shows, or None; glycans as CandyCrumbs named the fragments (derivatized with
    derivatize_carboxyls, as heavier sialic acids or lighter lactones can reorder the chains), and an oxonium ion of a peptide with several
    glycans comes from the one glycan it does not lose, whose chains its cleavages name"""
    own = [(g, sub) for g, sub in zip(glycans, dc_name[1:]) if not str(sub[0]).startswith('loss of')] if len(glycans) > 1 else [(glycans[0], dc_name)]
    if len(own) != 1:
        return None
    flat = [y for sub in dc_name for y in sub] if dc_name and isinstance(dc_name[0], list) else list(dc_name)
    # a sulfate or phosphate the ion lost as SO3 or HPO3 (M-SO4, M-PO4) is not on it any more
    losses = Counter(m for cut in flat if str(cut).startswith('M_') for m in parse_global_mod(str(cut)[2:]))
    glycan, sub = own[0]
    if isinstance(glycan, str) and not is_composition(glycan):
        frag_iupac = fragment_to_fragIUPAC(glycan, sub)
        for loss, group in (('SO4', 'S'), ('PO4', 'P')):
            if frag_iupac is not None and losses[loss]:
                # with more of these groups on the fragment than it lost, which ones are left is unknown, so it gets no cartoon rather than a guessed one
                site = rf'\d?{group}(?=\d|[(\[\]]|$)'
                frag_iupac = re.sub(site, '', frag_iupac) if len(re.findall(site, frag_iupac)) <= losses[loss] else None
        return frag_iupac
    # an ion of a composition names its composition ('Y Hex(1)/HexNAc(2)/S(1)'), which GlycoDraw draws; a cross-ring ('02A Hex') keeps only part of a residue
    pieces = [str(p).split(' ', 1)[1] for p in flat if re.fullmatch(r'[BCYZM] \S+', str(p))]
    if len(pieces) != 1:
        return None
    composition = Counter({m: int(n) for m, n in re.findall(r'([^/(]+)\((\d+)\)', pieces[0])})
    composition.subtract({'S': losses['SO4'], 'P': losses['PO4']})
    return canonicalize_composition({m: n for m, n in composition.items() if n > 0}, as_string = True)


def fragment_to_fragIUPAC(glycan_string, dc_name):
    """Reduces the glycan part of a fragment label to the IUPAC-condensed structure it describes; cached, as every spectrum drawn asks for its cartoons"""
    key = (glycan_string, repr(dc_name))
    if key in _FRAG_IUPAC:
        return _FRAG_IUPAC[key]
    if len(_FRAG_IUPAC) > 5000:
        _FRAG_IUPAC.clear()
    flat = [y for sub in dc_name for y in sub] if dc_name and isinstance(dc_name[0], list) else list(dc_name)
    # an ion without a residue (HSO4-) has no structure to draw
    if any(str(cut).startswith('I_') for cut in flat):
        return None
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
        _FRAG_IUPAC[key] = domon_costello_to_fragIUPAC(glycan_string, cuts) if cuts else glycan_string
    except Exception:
        _FRAG_IUPAC[key] = None
    return _FRAG_IUPAC[key]


_CARTOONS = {}


def fragIUPAC_to_image(frag_iupac, padding = 4, scale = 1.0):
    """Renders a glycan fragment as a cropped SNFG cartoon at scale pixels per point (padding in points), or None if the drawing stack is
    unavailable; cached, as spectra are drawn again for every candidate, panel size, and saved image"""
    if (frag_iupac, padding, scale) in _CARTOONS:
        return _CARTOONS[(frag_iupac, padding, scale)]
    try:
        from io import BytesIO
        from glycowork.motif.draw import GlycoDraw
        from glycorender.render import convert_svg_to_png
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(BytesIO(convert_svg_to_png(GlycoDraw(frag_iupac, suppress = True, compact = True,
                                                                dim = 30).as_svg(), None, return_bytes = True, scale = scale)))
    except Exception:
        return None
    box, pad = image.convert('RGBA').getbbox(), round(padding * scale)
    if box:
        image = image.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(image.width, box[2] + pad), min(image.height, box[3] + pad)))
    if len(_CARTOONS) > 2000:
        _CARTOONS.clear()
    _CARTOONS[(frag_iupac, padding, scale)] = image
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
            match = re.fullmatch(r'([abcwxyz](?:-1|\+1)?)_(\d+)', str(token))
            if not match:
                continue
            series, index = match.group(1), int(match.group(2))
            target = n_cuts if series in N_TERM_IONS else c_cuts
            target.setdefault(index if series in N_TERM_IONS else len(peptide) - index, set()).add(f'{series[0]}{index}{series[1:]}')
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


def deisotope_ms2(peaks, precursor_charge, mass_tolerance = 0.2, sum_intensities = True, min_intensity = 0.0, min_isotope_count = 2,
                  validate_pattern = True, relabel = False):
    """collapses isotope patterns of an MS2 spectrum onto their monoisotopic peak\n
   | Arguments:
   | :-
   | peaks (dict): dictionary of form (fragment) m/z : intensity
   | precursor_charge (int): highest fragment charge whose isotope spacing is tested
   | mass_tolerance (float): m/z tolerance for isotope spacing and for peaks consumed by a pattern; default:0.2
   | sum_intensities (bool): whether a pattern's monoisotopic peak carries the summed intensity of the pattern; default:True
   | min_intensity (float): peaks below this intensity are ignored; default:0.0
   | min_isotope_count (int): minimum number of peaks of a pattern; default:2
   | validate_pattern (bool): whether to check each isotope's intensity against the previous one; default:True
   | relabel (bool): whether a pattern whose first peak has less than half the intensity of its most intense peak is named by the latter; default:False\n
   | Returns:
   | :-
   | Returns a dictionary of form (fragment) m/z : intensity, sorted by descending intensity
   """
    sorted_peaks = sorted([(m, i) for m, i in peaks.items() if i >= min_intensity])
    # plain lists and bisect, as indexing numpy arrays element by element in these loops was most of the time
    masses = [m for m, _ in sorted_peaks]
    consumed = [False] * len(sorted_peaks)
    deisotoped = {}
    spacings = [1.0034 / z for z in range(1, precursor_charge + 1)]
    # Whether any peak lies one isotope spacing of a tested charge above a peak, for all peaks at once (with the bounds the search below computes): a peak
    # without one starts no pattern, which spares most peaks of a pooled orbitrap or TOF spectrum the search
    arr = np.array(masses, dtype = float)
    partner = np.zeros(len(arr), dtype = bool) if not np.isnan(arr).any() else np.ones(len(arr), dtype = bool)
    for spacing in spacings if not partner.all() else []:
        start = np.maximum(np.searchsorted(arr, arr + spacing - mass_tolerance), np.arange(1, len(arr) + 1))
        partner |= (start < len(arr)) & (arr[np.minimum(start, len(arr) - 1)] <= arr + spacing + mass_tolerance)
    partner = partner.tolist()
    for i, (current_mass, current_intensity) in enumerate(sorted_peaks):
        # Skip if already processed
        if consumed[i]:
            continue
        lo, hi = bisect.bisect_left(masses, current_mass - mass_tolerance), bisect.bisect_right(masses, current_mass + mass_tolerance)
        consumed[lo:hi] = [True] * (hi - lo)
        best_pattern = [(current_mass, current_intensity)]
        # Check all charge states
        for spacing in spacings if partner[i] else []:
            next_mass = current_mass
            charge_pattern = [(current_mass, current_intensity)]
            # Fast-forward to relevant mass range
            start_idx = bisect.bisect_left(masses, next_mass + spacing - mass_tolerance, i + 1)
            for j in range(start_idx, len(sorted_peaks)):
                candidate_mass, candidate_intensity = sorted_peaks[j]
                if candidate_mass > next_mass + spacing + mass_tolerance:
                    break
                if consumed[j]:
                    continue
                if abs(candidate_mass - next_mass - spacing) <= mass_tolerance:
                    if validate_pattern and not (candidate_intensity <= charge_pattern[-1][1] * 1.20 or
                                                 candidate_intensity >= charge_pattern[-1][1] * 0.05):
                        continue
                    charge_pattern.append((candidate_mass, candidate_intensity))
                    next_mass = candidate_mass
            if len(charge_pattern) > len(best_pattern):
                best_pattern = charge_pattern
        if len(best_pattern) >= min_isotope_count:
            mono_mass = best_pattern[0][0]
            apex = max(best_pattern, key = lambda x: x[1])
            # With relabel, a pattern whose first peak is less than half its most intense one is named by the most intense: in spectra pooled from many
            # MS2, a noise peak 1-3 Da below a fragment starts its 'pattern'
            if relabel and apex[1] > 2 * best_pattern[0][1]:
                mono_mass = apex[0]
            for peak_mass, _ in best_pattern[1:]:
                lo, hi = bisect.bisect_left(masses, peak_mass - mass_tolerance), bisect.bisect_right(masses, peak_mass + mass_tolerance)
                consumed[lo:hi] = [True] * (hi - lo)
            deisotoped[mono_mass] = sum(i for _, i in best_pattern) if sum_intensities else best_pattern[0][1]
        else:
            deisotoped[current_mass] = current_intensity
    return dict(sorted(deisotoped.items(), key = lambda x: x[1], reverse = True))


def plot_annotated_spectrum(input_string, spectrum, intensities = None, mass_threshold = None,
                            max_cleavages = 3, mass_tag = None, sample_prep = 'underivatized',
                            disable_global_mods = False, prior_weight = 1.0, charge = None, ax = None,
                            annotate_top_n = None, annotation_threshold = 0.05, figsize = None,
                            draw_glycans = True, glycan_zoom = 0.25, max_glycan_cartoons = None,
                            label_fontsize = 6, max_label_levels = 6, filepath = '', hit_dict = None, mz_labels = False, diagnostic = None, cartoons = None,
                            cartoon_dpi = None, **kwargs):
    """Plots an MS2 spectrum annotated with CandyCrumbs fragment assignments\n
    | Arguments:
    | :-
    | input_string (string/dict): anything CandyCrumbs accepts, i.e., a glycan, a peptide*glycan* string, or an input dict
    | spectrum (dataframe/list): either the prediction dataframe from wrap_inference, or the observed m/z values
    | intensities (list): the peak_d list from wrap_inference when spectrum is a dataframe, else the intensities
    | mass_threshold (float): maximum tolerated mass difference for fragment matching; default:None (see CandyCrumbs)
    | max_cleavages (int): maximum concurrent fragmentations per mass; default:3
    | mass_tag (float): mass of the glycan label or reducing end modification; default:None
    | sample_prep (string): underivatized/permethylated/deuteropermethylated/peracetylated; default:'underivatized'
    | disable_global_mods (bool): whether to disable global modifications; default:False
    | prior_weight (float): weighting of prior-informed scoring; default:1.0
    | charge (int): charge state of the precursor ion; taken from the dataframe when available; default:None
    | ax (matplotlib axis): axis to plot on, creates a new figure if None; default:None
    | annotate_top_n (int): only label the N most intense annotated peaks; default:None (label all)
    | annotation_threshold (float): minimum relative intensity (0-1) to annotate a peak; default:0.05
    | figsize (tuple): figure size if creating a new figure; default:None (chosen for the input type)
    | draw_glycans (bool): whether to add SNFG cartoons for glycan fragments; default:True
    | glycan_zoom (float): scale of the SNFG cartoons; default:0.25
    | cartoon_dpi (float): resolution the figure will be shown or saved at, which the cartoons are rendered for, as an image shrunk or enlarged on
    |                      drawing turns grainy or blurry; default:None (1 px per point, as for about 300 dpi)
    | max_glycan_cartoons (int): the most cartoons to draw; they are placed in this order wherever they fit without overlapping each other, the
    |                            title, or another cartoon's leader, near their peak: fragments of diagnostic peaks first, then single glycosidic
    |                            cleavages, then the rest, most intense first within each; default:None (as many as fit)
    | label_fontsize (int): font size of the peak labels; default:6
    | max_label_levels (int): how many times to push a label clear of its neighbors before dropping it; default:6
    | filepath (string): where to save the figure; default:'' (not saved)
    | hit_dict (dict): CandyCrumbs output for these peaks and settings if already computed (e.g., outside a drawing lock); default:None (computed here)
    | mz_labels (bool): whether each label also gives the peak's observed m/z, and unannotated peaks above annotation_threshold get their m/z in gray where
    |                   there is room; default:False
    | diagnostic (list): m/z (to 2 decimals or better) of peaks that tell this structure apart from its alternatives (e.g., the supporting peaks of
    |                    supporting_ions); they are labeled whatever their intensity, in lilac, and their fragments are drawn as cartoons first; default:None
    | cartoons (dict): fragment IUPAC -> True to draw that fragment's cartoon before all others (beyond max_glycan_cartoons and max shift), False to leave
    |                  it out; the axis' cartoon_fragments maps every labeled peak with a cartoon fragment to (fragment IUPAC, drawn); default:None
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
    hit_dict = hit_dict if hit_dict is not None else CandyCrumbs(input_string, mz_values.tolist(), mass_threshold, max_cleavages = max_cleavages,
                           simplify = True, charge = charge, mass_tag = mass_tag, sample_prep = sample_prep, intensities = peak_intensities.tolist(),
                           disable_global_mods = disable_global_mods, prior_weight = prior_weight, **kwargs)
    peptide, glycans, glycosites = resolve_spectrum_input(input_string)
    glycan_string = glycans[0] if glycans and isinstance(glycans[0], str) and not is_composition(glycans[0]) else None
    ladder_ax = None
    cartoon_room = 0.26 if draw_glycans and glycans else 0.02
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
    # Final limits before anything is placed in pixels: labels and cartoons are laid out for this data-to-pixel mapping
    ax.set_xlim(mz_values.min() - 20, mz_values.max() + 20)
    ax.set_ylim(0, 104)
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
        rank = 0 if diagnostic is not None and np.any(np.isclose(np.asarray(diagnostic, dtype = float), mz, rtol = 0, atol = 0.0051)) else 1 if len(shown) == 1 and re.match(r'[BCYZ]_\d', str(shown[0])) else 2
        peaks.append((mz, rel_int, label, kind, dc_name, rank))
    if annotate_top_n is not None:
        keep = {id(p) for p in sorted(peaks, key = lambda x: -x[1])[:annotate_top_n]}
        peaks = [p for p in peaks if id(p) in keep]
    # one collection for all annotated peaks (a vlines call per peak took ~80 ms of every spectrum the app draws), drawn in the same order
    if peaks:
        ax.vlines([p[0] for p in peaks], 0, [p[1] for p in peaks], colors = [DIAGNOSTIC_COLOR if p[5] == 0 else PEAK_COLORS.get(p[3], 'tab:red') for p in peaks],
                  linewidth = 1.3)
    # Each label sits directly above its peak, lifted only as far as its neighbors require, with a dotted
    # leader whenever it had to move; anything that cannot be placed at all is dropped rather than overlaid
    figure = ax.figure
    # Lays out ticks and limits without rasterizing; the figure is drawn when it is shown or saved
    figure.draw_without_rendering()
    renderer = figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    placed, label_tops = [], {}
    # Annotated peaks are labeled first, most intense first; with mz_labels, unannotated peaks then get their m/z in gray wherever room is left
    labels = [(mz, rel_int, f'{mz:.2f}  {label}' if mz_labels else label, DIAGNOSTIC_COLOR if rank == 0 else PEAK_COLORS.get(kind, 'tab:red'))
              for mz, rel_int, label, kind, _, rank in sorted(peaks, key = lambda x: (x[5] > 0, -x[1]))]
    if mz_labels:
        labels += [(mz, rel_int, f'{mz:.2f}', 'grey') for mz, rel_int in sorted(zip(mz_values, rel_intensities), key = lambda x: -x[1]) if not hit_dict.get(mz)]
    for mz, rel_int, label, color in labels:
        if rel_int < annotation_threshold * 100 and color != DIAGNOSTIC_COLOR:
            continue
        text = place_peak_label(ax, renderer, mz, rel_int, label, color, placed,
                                (axes_box.y0, axes_box.y1), label_fontsize, max_label_levels)
        if text is not None and color == 'grey':
            # A bare m/z is only worth it where no taller peak runs through it
            (x0, y0), (x1, _) = ax.transData.inverted().transform(text.get_window_extent(renderer).get_points())
            if np.any((mz_values >= x0) & (mz_values <= x1) & (rel_intensities > y0) & (mz_values != mz)):
                text.remove()
                placed.pop()
                continue
        if text is not None:
            label_tops[mz] = text.get_window_extent(renderer).y1
        if text is not None and abs(text.xyann[1]) > 8:
            ax.annotate('', xy = (mz, rel_int),
                        xytext = (text.xyann[0], text.xyann[1] - np.sign(text.xyann[1])),
                        textcoords = 'offset points',
                        arrowprops = dict(arrowstyle = '-', lw = 0.4, ls = ':', color = color,
                                          shrinkA = 0, shrinkB = 1))
    cartoons_drawn = False
    if draw_glycans and glycans:
        blended = ax.get_xaxis_transform()
        # Only glycan-only ions get a cartoon; on a peptide-bearing fragment the picture would just repeat
        # the intact glycan. One cartoon per distinct structure, anchored on its most intense member with a
        # leader from every peak it explains: an oxonium series shares a structure and differs only by a
        # neutral loss, so a box per loss repeats the same picture and drags them away from their peaks
        # cartoons come from the glycans as CandyCrumbs named their fragments: derivatized (heavier sialic acids, or lighter lactones, can reorder the chains)
        drawn = [derivatize_carboxyls(g, kwargs['sialic_derivatization']) if kwargs.get('sialic_derivatization') is not None and isinstance(g, str) and
                 not is_composition(g) else g for g in glycans]
        groups, cartoons, shown = {}, cartoons or {}, set()
        for mz, rel_int, _, kind, dc_name, rank in peaks:
            # A leader to a noise-level peak is a line across the plot for nothing, so a cartoon only
            # claims the peaks that were worth labeling in the first place
            if kind not in ('oxonium', 'glycan') or (rel_int < annotation_threshold * 100 and rank > 0):
                continue
            frag_iupac = fragment_cartoon(drawn, dc_name)
            if frag_iupac is not None:
                groups.setdefault(frag_iupac, []).append((rel_int, mz, rank))
        # Cartoons the caller switched on first, then diagnostic fragments, then single glycosidic cleavages (the cartoons that say something), then
        # cross-rings and multiple cleavages; switched off ones never
        ranked = sorted([g for g in groups.items() if cartoons.get(g[0]) is not False], key = lambda group: (min(m[2] for m in group[1]), -max(group[1])[0]))
        ranked = [g for g in ranked if cartoons.get(g[0])] + [g for g in ranked if not cartoons.get(g[0])][:max_glycan_cartoons]
        # In display pixels: cartoons fill the band between the axes and the title (which sits at 1.4 axes heights) or the peptide ladder, each at
        # the free spot closest to its peak, lower rows preferred; a cartoon is left out rather than drawn more than max_shift from its peak, over
        # another one, or where a leader (box to the top of its peak's label) would cross a box, a label, or another leader
        gap = 3
        band = (axes_box.y1 + gap, min(axes_box.y0 + 1.4 * axes_box.height, figure.bbox.y1) - gap)
        if ladder_ax is not None and ladder_ax.get_window_extent(renderer).y0 > axes_box.y1:
            band = (band[0], min(band[1], ladder_ax.get_window_extent(renderer).y0 - gap))
        max_shift = 0.08 * axes_box.width
        boxes, leaders = [], []
        # Boxes and labels block leaders through their diagonals, which any segment passing through a box with both ends outside it crosses
        blocking = [segment for b in placed for segment in (((b.x0, b.y0), (b.x1, b.y1)), ((b.x0, b.y1), (b.x1, b.y0)))]
        to_blended = blended.inverted()

        def crosses(a, b):
            # Whether any segment of a properly intersects any segment of b
            if not len(a) or not len(b):
                return False
            a, b = np.asarray(a, dtype = float)[:, None], np.asarray(b, dtype = float)[None]
            side = lambda p, q, r: (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
            return bool(np.any((side(b[..., 0, :], b[..., 1, :], a[..., 0, :]) * side(b[..., 0, :], b[..., 1, :], a[..., 1, :]) < 0) &
                               (side(a[..., 0, :], a[..., 1, :], b[..., 0, :]) * side(a[..., 0, :], a[..., 1, :], b[..., 1, :]) < 0)))

        # Rendered at the pixels per point of the shown or saved figure, so that matplotlib draws them one to one
        scale = glycan_zoom * cartoon_dpi / 72 if cartoon_dpi else 1.0
        for frag_iupac, members in ranked:
            image = fragIUPAC_to_image(frag_iupac, scale = scale)
            if image is None:
                continue
            forced = bool(cartoons.get(frag_iupac))
            box = AnnotationBbox(OffsetImage(np.array(image), zoom = glycan_zoom / scale), (0, 0), xycoords = blended, frameon = True, annotation_clip = False,
                                 bboxprops = dict(boxstyle = 'round,pad=0.15', fc = 'white', ec = '#cccccc', lw = 0.5))
            ax.add_artist(box)
            extent = box.get_window_extent(renderer)
            width, height = extent.width, extent.height
            # Leaders end at the top of a peak's label (or of the peak), most intense peak first
            tops = []
            for rel_int, mz, _ in sorted(members, reverse = True):
                px, py = ax.transData.transform((mz, rel_int))
                tops.append((px, max(py, label_tops.get(mz, -np.inf))))
            anchor, lo, hi = tops[0][0], axes_box.x0 + width / 2, axes_box.x1 - width / 2
            candidates = []
            for y in sorted({band[0]} | {b[3] + gap for b in boxes}):
                if y + height > band[1]:
                    continue
                # Box centers that would overlap a box already placed at this height
                blocked = [(b[0] - width / 2 - gap, b[2] + width / 2 + gap) for b in boxes if b[1] < y + height + gap and b[3] + gap > y]
                for x in {min(max(x, lo), hi) for x in [anchor, lo, hi] + [edge for interval in blocked for edge in interval]}:
                    if (forced or abs(x - anchor) <= max_shift) and not any(a < x < b for a, b in blocked):
                        candidates.append((abs(x - anchor) + 0.5 * (y - band[0]), x, y))
            # The leader to the cartoon's most intense peak must be clear; those to its other peaks (neutral losses, other charges) are drawn where clear
            for _, x, y in sorted(candidates):
                rect = (x - width / 2 - gap, y - gap, x + width / 2 + gap, y + height + gap)
                diagonals = [((rect[0], rect[1]), (rect[2], rect[3])), ((rect[0], rect[3]), (rect[2], rect[1]))]
                if not crosses([((x, y), tops[0])], blocking) and not crosses(leaders, diagonals):
                    break
            else:
                box.remove()
                continue
            box.xy = box.xybox = tuple(to_blended.transform((x, y + height / 2)))
            for top in tops:
                if top is tops[0] or not crosses([((x, y), top)], blocking):
                    ax.plot(*zip(to_blended.transform((x, y)), to_blended.transform(top)), transform = blended, color = '#cccccc', lw = 0.5, ls = '--', clip_on = False)
                    leaders.append(((x, y), top))
                    blocking.append(((x, y), top))
            boxes.append((x - width / 2, y, x + width / 2, y + height))
            blocking += diagonals
            cartoons_drawn = True
            shown.add(frag_iupac)
        ax.cartoon_fragments = {mz: (frag_iupac, frag_iupac in shown) for frag_iupac, members in groups.items() for _, mz, _ in members if mz in label_tops}
    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity (%)')
    stage = f'MS\u00b3 spectrum of m/z {kwargs["ms3_precursor"]:.2f}' if kwargs.get('ms3_precursor') is not None else 'MS\u00b2 spectrum'
    title = f'Annotated {stage}: {peptide + "*" + str(glycan_string) if peptide else input_string}'
    title_size = 9 if len(title) > 90 else 'large'
    title_text = ladder_ax.set_title(title, fontsize = title_size) if ladder_ax is not None else ax.set_title(title, fontsize = title_size, y = 1.40 if cartoons_drawn else 1.0)
    # A long structure made the title (centered on its axes, which may be an inset) wider than the figure, which cut it off at both ends, so it breaks
    # after a bracket where it would leave the figure; beyond two lines (more ran off the top of the figure) its middle is left out
    extent = title_text.get_window_extent(ax.figure._get_renderer())
    room = 2 * min(extent.x0 + extent.width / 2, ax.figure.bbox.width - extent.x0 - extent.width / 2)
    if extent.width > 0.95 * room:
        chars, lines = int(len(title) * 0.95 * room / extent.width), ['']
        for part in re.findall(r'[^)\]]*[)\]]|[^)\]]+', title):
            lines[-1:] = [lines[-1], part] if lines[-1] and len(lines[-1]) + len(part) > chars else [lines[-1] + part]
        title_text.set_text('\n'.join(lines if len(lines) < 3 else [lines[0], '\u2026' + title[1 - chars:]]))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ladder_ax is not None:
        draw_peptide_ladder(peptide, hit_dict, glycosites, ax = ladder_ax)
    if filepath:
        plt.savefig(filepath, dpi = 300, bbox_inches = 'tight')
    return hit_dict, ax


def plot_fragments_excel(df: pd.DataFrame | str | Path, # DataFrame or filepath with fragments
                         folder_filepath: str | Path, # Output folder path
                         glycan: str, # Glycan the fragments come from
                         fragment_col_num: int | str = 0, # Fragment column index, or its name
                         scaling_factor: float | None = None, # Image scaling; defaults to 0.2 if compact else 1.2
                         compact: bool = False, # Use compact style
                         **kwargs # Passed on to GlycoDraw, e.g., vertical, show_linkage
                         ) -> None:
    "Creates Excel file with SNFG images of glycan fragments in a new column"
    if Path(folder_filepath).suffix:
        raise ValueError(f"folder_filepath has to be a directory (got '{folder_filepath}').")
    if isinstance(df, (str, Path)):
        filename, suffix = Path(df).stem, Path(df).suffix.lower()
        df = pd.read_csv(df) if suffix == ".csv" else pd.read_csv(df, sep = "\t") if suffix == ".tsv" else pd.read_excel(df)
    else:
        filename, df = "fragments", df.copy()
    column = df[fragment_col_num] if isinstance(fragment_col_num, str) else df.iloc[:, fragment_col_num]
    iupacs = []
    for row, fragment in enumerate(column, start = 2):  # Excel row: 1-indexed plus a header row
        if isinstance(fragment, str):
            fragment = fragment.strip()
            if fragment.startswith("["):
                try:
                    fragment = ast.literal_eval(fragment)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Could not parse fragment list in row {row}: {fragment!r}") from e
            elif fragment:
                fragment = [fragment]
        # Empty cells (None, NaN, '', []) get no image
        if not isinstance(fragment, (list, tuple)) or not fragment:
            if not (fragment is None or fragment == '' or isinstance(fragment, (list, tuple)) or pd.isna(fragment)):
                raise ValueError(f"Unexpected fragment format in row {row}: {fragment!r}")
            iupacs.append(None)
            continue
        iupac = fragment_to_fragIUPAC(glycan, list(fragment))
        # Ions without a residue (I_...) correctly have no image
        flat = [y for sub in fragment for y in (sub if isinstance(sub, (list, tuple)) else [sub])]
        if iupac is None and not any(str(cut).startswith('I_') for cut in flat):
            warnings.warn(f"Could not convert the fragment in row {row}: {fragment!r}; it is left without an image")
        iupacs.append(iupac)
    df["fragment_IUPAC"] = iupacs
    plot_glycans_excel(df, 
                       Path(folder_filepath) / f"{filename}_SNFG.xlsx", 
                       glycan_col_num = "fragment_IUPAC",
                       scaling_factor = scaling_factor if scaling_factor is not None else (0.2 if compact else 1.2),
                       compact = compact, **kwargs)
