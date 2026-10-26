import pytest
import re
import random
import time
from collections import Counter
import numpy as np
import networkx as nx
from pyteomics import mass
from candycrunch.analysis import CandyCrumbs, get_fragment_mass, glycan_to_graph_monos, derivatization_sites, DERIVATIZATION_MASSES, \
    build_glycopeptide_input, rank_glycopeptide_structures, rank_glycopeptide_compositions, rank_glycopeptide_sites, supporting_ions, WATER_MASS, mono_attributes, smiles_mono_attributes, \
    CHAIN_IONS, domon_costello_to_mpl, \
    peptide_to_glycowork, infer_glycosites, AA_masses, generate_atomic_frags, get_initial_global_mods, input_to_graph, glycopeptide_string_to_input, extend_masses, \
    subgraphs_to_domon_costello, rank_chains, X_cross_rings, peptide_to_RF_nomenclature, get_glycan_cleavages, max_fragment_charge
from glycowork.motif.tokenization import glycan_to_mass, composition_to_mass, calculate_adduct_mass, HYDROGEN_MASS, PROTON_MASS, modification_formula_dict

TEST_DICTS = [{'glycan_string':'GalNAc(b1-4)GlcNAc(b1-3)[GalNAc(b1-4)GlcNAc(b1-6)]Gal(b1-4)Glc',
'charge': -2,
'label_mass':2.0156,
'masses': [405.07,423.05,465.11,528.14,549.15,567.17,713.21,731.22,749.24,892.27,934.28,952.28,973.25,1095.31,1113.32],
'annotations': [['B_2_Alpha'],['C_2_Alpha'],['24A_3_Alpha'],['Z_2_Alpha','Y_3_Beta'],['C_3_Alpha','Z_2_Alpha'],['C_3_Alpha','Y_2_Alpha'],['Z_3_Alpha','Z_3_Beta'],['Z_2_Alpha'],['Y_2_Alpha'],[],['Z_3_Alpha'],['Y_3_Alpha'],['C_3_Alpha'],['M_C2H4O2'],['M_C2H2O']],
'ref': 'JC_Pygmy_Hippo_milk_(neutral)',
'max_cleavages':4
},
{'glycan_string': 'Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc',
'charge': -2,
'label_mass':2.0156,
'masses': [405.11,425.07,443.07,528.17,546.19,586.18,670.19,688.20,731.24,749.25,852.25,870.26,892.28,934.30,952.30,1013.27,1055.28,1073.30,1094.30,1114.31,1216.32,1234.33],
'annotations': [['B_2_Beta'],['02A_3_Alpha','M_H2O'],['02A_3_Alpha'],['Z_2_Alpha','Y_3_Beta'],['Y_2_Alpha','Y_3_Beta'],['04A_4_Alpha'],['C_3_Alpha','Z_2_Beta'],['C_3_Alpha','Y_2_Beta'],['Z_2_Alpha'],['Y_2_Alpha'],['Z_2_Beta'],['Y_2_Beta'],[],['Z_3_Alpha'],['Y_3_Alpha'],[],['Z_3_Beta'],['Y_3_Beta'],['C_4_Alpha'],['Y_4_Alpha'],['M_C2H4O2'],['M_C2H2O']],
'ref': 'JC_Pygmy_Hippo_milk_(neutral)'
},
{'glycan_string': 'Fuc(a1-2)Gal(b1-3)[Gal(a1-3)[Fuc(a1-2)]Gal(b1-4)GlcNAc(b1-6)]GalNAc',
'charge': -1,
'label_mass':2.0156,
'masses': [389.15,407.14,503.22,529.15,571.49,655.16,697.20,715.20,733.24,859.35,877.26,895.24,1005.26,1023.27,1041.33,1058.21,1143.45,1161.34],
'annotations': [['Z_2_Alpha','Z_1_Gamma'],['Y_2_Alpha','Z_1_Gamma'],['Z_3_Alpha','Z_3_Beta','Z_1_Gamma','M_H2O'],['24A_3_Alpha'],['02A_3_Alpha','M_H2O'],[],['Z_3_Alpha','Z_1_Gamma'],['Y_3_Alpha','Z_1_Gamma'],['Y_2_Alpha'],[],['Z_1_Gamma'],['Y_1_Gamma'],[],['Z_3_Alpha'],['Y_3_Alpha'],['Y_2_Gamma'],['M_C2H4O2'],['M_C2H2O']],
'ref': '10.1074/mcp.M116.067983'
},
{'glycan_string': 'Fuc(a1-2)[GalNAc(a1-3)]Gal(b1-3)[GalNAc(a1-3)[Fuc(a1-2)]Gal(b1-4)[Fuc(a1-2)]GlcNAc(b1-6)]GalNAc',
'charge': -2,
'label_mass':2.0156,
'masses': [246.99,306.87,510.14,540.75,553.26,694.70,733.22,766.26,861.36,900.13,919.09,1064.16,1082.30,1330.37,1372.21,1390.18],
'annotations': [['Y_3_Alpha','B_2_Alpha','M_C2H4O2'],['Y_3_Alpha','B_2_Alpha'],['B_2_Alpha'],['Y_1_Gamma'],['Y_2_Alpha','Z_1_Gamma'],['Y_2_Gamma'],['Y_1_Alpha'],[],['Y_3_Alpha','Z_1_Gamma'],['Z_2_Alpha','Z_2_Beta'],['04A_0_Alpha'],['Z_1_Gamma'],['Y_1_Gamma'],[],['Z_2_Gamma'],['Y_3_Gamma']],
'ref': '10.1074/mcp.M116.067983'
},
{'glycan_string': 'Neu5Ac(a2-6)GalNAc',
'charge': -1,
'label_mass':2.0156,
'masses': [170.03,204.07,222.12,276.10,290.11,308.12],
'annotations': [[],['Z_1_Alpha'],['Y_1_Alpha'],[],['B_1_Alpha'],['C_1_Alpha']],
'ref': '10.1074/mcp.M116.067983'
},
{'glycan_string': 'Fuc(a1-2)[Gal(b1-3)]Gal(b1-3)[Neu5Ac(a2-3)Gal(?1-?)[Fuc(?1-?)]GlcNAc(b1-6)]GalNAc',
'charge': -1,
'label_mass':2.0156,
'masses': [553.20,571.18,674.17,692.36,715.49,733.28,829.06,859.36,1023.38,1041.20,1057.28,1185.35,1203.34,1333.37,1416.40,1434.44],
'annotations': [['Y_2_Alpha','Z_1_Gamma'],['Y_2_Alpha','Y_1_Gamma'],['Z_1_Alpha'],['Y_1_Alpha'],['Y_3_Alpha','Z_1_Gamma'],[],['Z_2_Alpha','Z_2_Beta','M_CH2O'],['Y_2_Alpha','Z_2_Beta'],['Z_2_Alpha'],['Y_2_Alpha'],['Y_3_Alpha','Y_3_Beta'],['Z_3_Alpha'],['Y_3_Alpha'],['Y_2_Gamma'],[],['M_C2H4O2']],
'ref': '10.1074/mcp.M116.067983'
},
{'glycan_string':"AVAVT*Neu5Ac(a2-3)Gal(a1-3)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)]GalNAc*LQSH",
'charge':+3,
'label_mass':0,
'masses':[156.076, 204.086, 228.09, 243.108, 259.176, 274.092, 292.102, 355.148, 366.14, 425.178, 454.155, 468.232, 564.797, 657.234, 666.339, 747.365, 791.372, 828.393, 860.311, 892.912, 925.51, 973.939, 990.897, 1022.361, 1026.430, 1075.959, 1128.589, 1236.557, 1267.642, 1290.641, 1313.451, 1771.752, 1884.831, 1981.806],
'annotations':[[['y_1'],['loss of glycan 1']],[['No Peptide'], ['C_4_Alpha', 'Z_1_Beta', 'Z_1_Alpha']],[['z_2'],['loss of glycan 1']],[['y_2'],['loss of glycan 1']],[['c_3'],['loss of glycan 1']],[['No Peptide'], ['B_1_Beta', 'M_H2O']],[['No Peptide'], ['B_1_Beta']],[['z_3'], ['loss of glycan 1']],[['No Peptide'], ['Y_3_Alpha', 'B_3_Alpha']],[['w_4'], ['loss of glycan 1']],[['No Peptide'], ['B_2_Alpha']],[['z_4'], ['loss of glycan 1']],[['Peptide'], ['Y_1_Beta', 'Y_1_Alpha']],[['No Peptide'], ['B_3_Alpha']],[['Peptide'], ['Y_1_Beta', 'Y_2_Alpha']],[['Peptide'], ['Y_1_Beta', 'Y_3_Alpha']],[['Peptide'], ['Y_1_Alpha']],[['Peptide'], ['Y_2_Beta', 'Y_3_Alpha']],[['No Peptide'], ['Y_1_Beta']],[['Peptide'], ['Y_1_Beta']], [['Peptide'], ['Y_0_Alpha']],[['Peptide'], ['Y_3_Alpha']], [['z_6'], ['M']], [['No Peptide'], ['Y_3_Alpha']], [['z_7'], ['M']], [['z_8'], ['M']], [['Peptide'], ['Y_1_Beta', 'Y_1_Alpha']], [], [], [['Peptide'], ['Y_1_Alpha', 'Y_2_Beta']], [['No Peptide'], ['B_4_Alpha']], [['c_5'], ['M']], [['c_6'], ['M']], [['z_6'], ['M']]],
'ref':'https://doi.org/10.1007/s13361-018-1945-7'
}
]
# Correctly annotated peaks (top-1, top-5) of each TEST_DICTS spectrum as of 2026-10-19: fewer fails the accuracy tests, so that a lost annotation shows
# instead of passing under a common threshold; when a change annotates more (or deliberately fewer), these are updated with it
REFERENCE_CORRECT = [(8, 11), (16, 17), (10, 13), (7, 13), (4, 4), (9, 11), (25, 27)]
CHAIN_RANKS = {'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu'}

def strip_chain_rank(name):
    parts = name.split('_')
    if len(parts) == 3 and parts[2] in CHAIN_RANKS:
        return '_'.join(parts[:2])
    return name

def normalize_annotations(annotations):
    if not annotations:
        return []
    if isinstance(annotations[0], str):
        return sorted(strip_chain_rank(x) for x in annotations)
    return sorted(tuple(sorted(strip_chain_rank(y) for y in x)) for x in annotations)

@pytest.mark.parametrize("test_dict", TEST_DICTS)
def test_candycrumbs_accuracy(test_dict):
    result = CandyCrumbs(test_dict['glycan_string'], test_dict['masses'], mass_threshold = 0.4, charge = test_dict['charge'], mass_tag = test_dict['label_mass'], max_cleavages = test_dict.get('max_cleavages', 3))
    total_annotations = len(test_dict['annotations'])
    correct_annotations = 0
    assert len(result) == total_annotations
    for mz, expected_annotations in zip(test_dict['masses'], test_dict['annotations']):
        if not expected_annotations:
            total_annotations=total_annotations-1
            continue
        if mz in result:
            if result[mz]:
                predicted_annotations = result[mz]['Domon-Costello nomenclatures'][0]
                if normalize_annotations(predicted_annotations) == normalize_annotations(expected_annotations):
                    correct_annotations += 1
    score = correct_annotations / total_annotations
    floor = REFERENCE_CORRECT[TEST_DICTS.index(test_dict)][0]
    print(f"Score: {score:.2f} ({correct_annotations}/{total_annotations}), reference: {floor}/{total_annotations}")
    assert correct_annotations >= floor

@pytest.mark.parametrize("test_dict", TEST_DICTS)
def test_candycrumbs_accuracy_top5(test_dict):
    result = CandyCrumbs(test_dict['glycan_string'], test_dict['masses'], 0.4, charge = test_dict['charge'], mass_tag = test_dict['label_mass'], simplify = False)
    total_annotations = len(test_dict['annotations'])
    correct_annotations = 0
    assert len(result) == total_annotations
    for mz, expected_annotations in zip(test_dict['masses'], test_dict['annotations']):
        if not expected_annotations:
            total_annotations=total_annotations-1
            continue
        if mz in result:
            if result[mz]:
                predicted_annotations = result[mz]['Domon-Costello nomenclatures']
                exp_norm = normalize_annotations(expected_annotations)
                if any(normalize_annotations(pred) == exp_norm for pred in predicted_annotations):
                    correct_annotations += 1
    score = correct_annotations / total_annotations
    floor = REFERENCE_CORRECT[TEST_DICTS.index(test_dict)][1]
    print(f"Score: {score:.2f} ({correct_annotations}/{total_annotations}), reference: {floor}/{total_annotations}")
    assert correct_annotations >= floor


DERIVATIZATION_GLYCANS = ['Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
                          'Neu5Gc(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', 'Kdn(a2-3)Gal(b1-4)Glc',
                          'Man6P(a1-2)Man(a1-2)Man',
                          'GalOS(b1-3)GalNAc', 'Neu5Ac9Ac(a2-3)Gal(b1-4)Glc', 'GlcNS6S(a1-4)IdoA2S(a1-4)GlcNS',
                          'Gal(b1-4)GlcN(a1-6)Glc', 'Xyl(b1-2)Man(b1-4)GlcNAc', 'Fuc(a1-2)Gal4S(b1-3)GalNAc',
                          'LDManHep(a1-3)LDManHep(a1-5)Kdo', 'Araf(a1-5)Araf(a1-5)Araf', 'QuiNAc(b1-3)GalNAc', 'Kdo4P(a2-4)Kdo']


@pytest.mark.parametrize("glycan", DERIVATIZATION_GLYCANS)
@pytest.mark.parametrize("sample_prep", ['underivatized', 'permethylated', 'deuteropermethylated', 'peracetylated'])
def test_candycrumbs_precursor_matches_glycowork(glycan, sample_prep):
    # The intact ion is the sum of every residue, substituent, derivatization and reducing-end table CandyCrumbs uses
    for mass_tag, modification in ((2 * HYDROGEN_MASS, 'reduced'), (0, None)):
        expected = glycan_to_mass(glycan, sample_prep = sample_prep, modification = modification) - PROTON_MASS
        assert abs(get_fragment_mass(glycan, 'M', charge = -1, mass_tag = mass_tag,
                                     sample_prep = sample_prep) - expected) < 1e-3


def test_derivatization_sites_match_glycowork():
    for prep, sites in derivatization_sites.items():
        for mono, positions in sites.items():
            # glycowork's residue masses assume one glycosidic bond taking a site
            n = (composition_to_mass({mono: 1}, sample_prep = prep) - composition_to_mass({}, sample_prep = prep) -
                 composition_to_mass({mono: 1}) + composition_to_mass({})) / DERIVATIZATION_MASSES[prep]
            assert len(positions) == round(n) + 1, (prep, mono)


def test_candycrumbs_permethylated_sialic_amides():
    # sialic acids amidated before permethylation end as dimethylamides, whatever their linkage: the B1 ion 13.03 above the methyl ester's 376.20
    out = CandyCrumbs('Neu5Ac(a2-?)Gal(b1-4)Glc', [389.23, 582.20], 0.5, charge = 1, mass_tag = calculate_adduct_mass('C6HNS'), sample_prep = 'permethylated',
                      sialic_derivatization = ('dimethylamide', 'dimethylamide'))
    assert 'B_1_Alpha' in out[389.23]['Domon-Costello nomenclatures'][0]
    assert abs(out[389.23]['Theoretical fragment masses'][0] - 389.2282) < 1e-3
    assert 'Y_2_Alpha' in out[582.20]['Domon-Costello nomenclatures'][0]


def test_candycrumbs_deuteropermethylated():
    # CD3 caps weigh 3.0188 Da more than CH3 ones and methanol leaves as CD3OH, while a native O-methyl stays CH3: the B1 ion of GalOMe is 228.18, three
    # caps above permethylated Gal's 219.12 and three below deuteropermethylated Gal's 231.20
    masses = [309.29, 359.26, 394.31, 607.47]
    out = CandyCrumbs('Neu5Ac(a2-3)Gal(b1-3)GalNAc', masses, 0.02, charge = 1, sample_prep = 'deuteropermethylated')
    assert [out[m]['Domon-Costello nomenclatures'][0] for m in masses] == [['Y_1_Alpha'], ['B_1_Alpha', 'M_CD3OH'], ['B_1_Alpha'], ['B_2_Alpha']]
    out = CandyCrumbs('GalOMe(b1-3)GalNAc', [228.18, 231.20], 0.02, charge = 1, sample_prep = 'deuteropermethylated')
    assert out[228.18]['Domon-Costello nomenclatures'][0] == ['B_1_Alpha'] and out[231.20] is None


def test_smiles_mono_attributes_reproduce_hand_tables():
    # The atom-level tables behind every residue without a hand table rebuild each hand table entry they name: masses (to the tables'
    # rounding), retained carbons and derivatized groups; 0,1A and 2,5X have no cut_type_dict name and 1,2X breaks adjacent ring bonds
    bases = {'Hex': ['Glc', 'Gal', 'Man'], 'HexNAc': ['GlcNAc', 'GalNAc', 'ManNAc'], 'HexN': ['GlcN', 'GalN', 'ManN'], 'dHex': ['Fuc', 'Rha', 'Qui'],
             'Pen': ['Xyl', 'Ara', 'Rib', 'Lyx'], 'HexA': ['GlcA', 'GalA', 'IdoA', 'ManA'], 'Neu5Ac': ['Neu5Ac'], 'Neu5Gc': ['Neu5Gc'], 'Kdn': ['Kdn']}
    for base, residues in bases.items():
        hand = mono_attributes[base]
        for residue in residues:
            generated = smiles_mono_attributes(residue)
            for fragment in set(hand['mass']) - {'01A', '12X', '25X'}:
                assert abs(generated['mass'][fragment] - hand['mass'][fragment]) < 1e-4, (residue, fragment)
                assert [generated[prop][fragment] for prop in ('atoms', 'permethylated', 'peracetylated')] == [
                    sorted(hand['atoms'][fragment]), hand['permethylated'][fragment], hand['peracetylated'][fragment]], (residue, fragment)


def test_candycrumbs_smiles_residues():
    # A residue without a hand table or a composition (Kdo, heptoses, furanoses) gets one from its atoms instead of leaving the glycan unannotated
    result = CandyCrumbs('LDManHep(a1-3)LDManHep(a1-5)Kdo', [191.06, 209.07, 429.13], 0.02, charge = -1, mass_tag = 0)
    assert [result[m]['Domon-Costello nomenclatures'][0] for m in (191.06, 209.07, 429.13)] == [['B_1_Alpha'], ['C_1_Alpha'], ['Y_2_Alpha']]
    # an anhydro bridge keeps the ring in one piece across it, so 3,6-anhydrogalactose has no cleavage separating C3 from C6
    assert set(smiles_mono_attributes('3,6-Anhydro-L-Gal')['mass']) == {'3,6-Anhydro-L-Gal', '02X', '02A', '15X', '15A', '25A', '25X'}


def test_candycrumbs_floating_parts():
    # Every floating part gets placed, so no linkage ends up as a monosaccharide node
    glycan = '{Fuc(a1-3)}{Neu5Ac(a2-3/6)}Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc'
    assert all(v in {'Fuc', 'Neu5Ac', 'Gal', 'GlcNAc', 'Man'} for v in glycan_to_graph_monos(glycan)[0].values())
    result = CandyCrumbs(glycan, [290.09], 0.1, charge = -2)
    assert result[290.09]['Domon-Costello nomenclatures'][0] == ['B_1_Alpha']


def test_candycrumbs_composition_substituents():
    # A sulfate of a composition is a sulfate, not a serine residue, and can sit on any fragment
    result = CandyCrumbs('Hex1HexNAc1S1', [241.0, 282.03, 464.11], 0.02, charge = -1)
    assert [result[m]['Domon-Costello nomenclatures'][0] for m in (241.0, 282.03)] == [['B Hex(1)/S(1)'],
                                                                                       ['B HexNAc(1)/S(1)']]
    assert result[464.11]['Domon-Costello nomenclatures'][0] == ['M Hex(1)/HexNAc(1)/S(1)']

IGG_G0F = 'GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'


def test_glycopeptide_fragments():
    # HCD ions of the IgG1 Fc glycopeptide EEQYNSTYR with G0F at 3+, from pyteomics peptide masses and glycowork residue masses: the peptide + 0,2X
    # HexNAc ion (+83), the intact precursor at 3+ (one basic residue), b5 that lost the glycan, and y8 that kept one HexNAc
    pep = mass.calculate_mass(sequence = 'EEQYNSTYR')
    hexnac = composition_to_mass({'HexNAc': 1}) - composition_to_mass({})
    ions = {pep + calculate_adduct_mass('C4H5NO') + PROTON_MASS: ([['Peptide'], ['02X_1_Alpha']], [['Peptide'], ['02X HexNAc']], 1),
            (pep + glycan_to_mass(IGG_G0F) - composition_to_mass({}) + 3 * PROTON_MASS) / 3: ([['Peptide'], ['M']], [['Peptide'], ['M']], 3),
            mass.fast_mass('EEQYN', ion_type = 'b', charge = 1): ([['b_5'], ['Y_0_Alpha']], [['b_5'], ['loss of glycan 1']], 1),
            mass.fast_mass('EQYNSTYR', ion_type = 'y', charge = 1) + hexnac: ([['y_8'], ['Y_1_Alpha', 'Y_1_Gamma']], [['y_8'], ['Y HexNAc(1)']], 1)}
    ions = {round(mz, 4): v for mz, v in ions.items()}
    for i, glycan in enumerate([IGG_G0F, 'Hex3HexNAc4dHex1']):
        result = CandyCrumbs({'peptide': 'EEQYNSTYR', 'glycans': [glycan], 'glycosites': [4]}, list(ions), charge = 3, fragmentation_method = 'HCD')
        for mz, (*labels, z) in ions.items():
            assert result[mz]['Domon-Costello nomenclatures'][0] == labels[i], (glycan, mz)
            assert result[mz]['Fragment charges'][0] == z
            assert abs(result[mz]['Theoretical fragment masses'][0] - (mz * z - (z - 1) * PROTON_MASS)) < 2e-3


def test_glycopeptide_ethcd_charges():
    # EThcD c ions keep the glycan; without a basic residue in ETQ the glycan carries the second proton of c3 2+
    glycan = 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'
    residues = glycan_to_mass(glycan) - composition_to_mass({})
    ions = {round((mass.fast_mass(seq, ion_type = 'c', charge = 1) + residues + PROTON_MASS) / 2, 4): f'c_{len(seq)}' for seq in ('ETQ', 'ETQPAT')}
    for g in (glycan, 'Hex1HexNAc1Neu5Ac1'):
        result = CandyCrumbs({'peptide': 'ETQPATSPAR', 'glycans': [g], 'glycosites': [2]}, list(ions), charge = 3, fragmentation_method = 'EThcD')
        for mz, name in ions.items():
            assert result[mz]['Domon-Costello nomenclatures'][0] == [[name], ['M']] and result[mz]['Fragment charges'][0] == 2


def test_glycopeptide_hydrogen_transfer_ions():
    # ETD-type spectra also hold c and z ions after a hydrogen atom moved between the two halves: c - H (c-1, as pyteomics names it) and z. + H (z+1, y - NH,
    # as CandyCrunch's z is the radical; pyteomics 4 calls it 'z+1', pyteomics 5 'z+2'), which only the ETD-type methods name
    glycan = 'Neu5Ac(a2-3)Gal(b1-3)GalNAc'
    ions = {round(mass.fast_mass('ETQ', ion_type = 'c-1', charge = 1) + glycan_to_mass(glycan) - composition_to_mass({}), 4): [['c-1_3'], ['M']],
            round(mass.fast_mass('PATSPAR', ion_type = 'y', charge = 1) - mass.calculate_mass(formula = 'NH'), 4): [['z+1_7'], ['loss of glycan 1']]}
    for g in (glycan, 'Hex1HexNAc1Neu5Ac1'):
        for method in ('ETD', 'EThcD', 'HCD', None):
            result = CandyCrumbs({'peptide': 'ETQPATSPAR', 'glycans': [g], 'glycosites': [2]}, list(ions), charge = 3, fragmentation_method = method)
            for mz, name in ions.items():
                assert (result[mz]['Domon-Costello nomenclatures'][0] if result[mz] else None) == (name if method in ('ETD', 'EThcD') else None), (g, method, mz)


def test_build_glycopeptide_input_modifications():
    # Met oxidation is a peptide residue, not a glycan; an unknown modification raises instead of becoming a massless glycan
    gp = build_glycopeptide_input('EMTSPGTPAR', 'M2(Oxidation);T3(HexNAc(1)Hex(1))')
    assert gp == {'peptide': 'EmTSPGTPAR', 'glycans': [{'HexNAc': 1, 'Hex': 1}], 'glycosites': [2]}
    y0 = round(mass.calculate_mass(sequence = 'EMTSPGTPAR') + calculate_adduct_mass('O') + PROTON_MASS, 4)
    assert CandyCrumbs(gp, [y0], charge = 2)[y0]['Domon-Costello nomenclatures'][0] == [['Peptide'], ['loss of glycan 1']]
    with pytest.raises(ValueError):
        build_glycopeptide_input('EMTSPGTPAR', 'S4(Phospho);T3(HexNAc(1)Hex(1))')
    # CandyCrunch's residue tokens in glycowork's notation weigh what AA_masses weighs
    assert peptide_to_glycowork('NcSmkj') == 'NC[Carbamidomethyl]SM[Oxidation]K[TMT6plex]K[Guanidinyl]'
    assert abs(composition_to_mass('H5N4', peptide = peptide_to_glycowork('EmNcSK')) - composition_to_mass('H5N4') - sum(AA_masses[aa] for aa in 'EmNcSK')) < 1e-9
    # N-X-C (also as c) and an N-X ending the peptide are sites too, after the canonical N-X-S/T
    assert infer_glycosites('NGcK', 'N') == [0] and infer_glycosites('NGTKNGcR', 'N') == [0, 4] and infer_glycosites('AANK', 'N') == [2] and infer_glycosites('ANPK', 'N') == []
    # deamidated Asn and Gln (PNGase F leaves released N-glycosites as Asp) are n and q, which weigh Asp and Glu and are no N-glycosite
    gp = build_glycopeptide_input('ITANTTDEP', 'N4(Deamidated);T5(HexNAc(1)Hex(1))')
    assert gp['peptide'] == 'ITAnTTDEP' and peptide_to_glycowork('nq') == 'N[Deamidated]Q[Deamidated]' and infer_glycosites('GLnSTK', 'N') == []
    assert abs(AA_masses['n'] - AA_masses['D']) < 1e-9 and abs(AA_masses['q'] - AA_masses['E']) < 1e-9


def test_rank_glycopeptide_structures():
    # HCD of a 3-sialyl T antigen: its direct fragments are the NeuAc-Hex oxonium ion (454) and the peptide with HexNAc-Hex; the 6-sialyl isomer
    # predicts the peptide with HexNAc-NeuAc (loss of Gal) instead, which is absent
    pep = mass.calculate_mass(sequence = 'TPSAAYPTDR')
    res = {k: composition_to_mass({k: 1}) - composition_to_mass({}) for k in ('Hex', 'HexNAc', 'Neu5Ac')}
    peaks = {res['HexNAc'] + PROTON_MASS: 100, res['HexNAc'] - WATER_MASS + PROTON_MASS: 30, res['Neu5Ac'] + PROTON_MASS: 80,
             res['Neu5Ac'] - WATER_MASS + PROTON_MASS: 60, res['Hex'] + res['HexNAc'] + PROTON_MASS: 50, res['Neu5Ac'] + res['Hex'] + PROTON_MASS: 40,
             pep + PROTON_MASS: 70, pep + res['HexNAc'] + PROTON_MASS: 90, pep + res['HexNAc'] + res['Hex'] + PROTON_MASS: 50,
             mass.fast_mass('AAYPTDR', ion_type = 'y', charge = 1): 20, mass.fast_mass('YPTDR', ion_type = 'y', charge = 1): 15}
    structures = ['Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', 'Neu5Ac(a2-3)Gal(b1-3)GalNAc']
    df = rank_glycopeptide_structures('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1)', peaks, charge = 2, structures = structures, fragmentation_method = 'HCD')
    assert df['structures'].tolist() == [[structures[1]], [structures[0]]] and df['rank'].tolist() == [1, 2]
    assert any(abs(mz - (pep + res['HexNAc'] + res['Neu5Ac'] + PROTON_MASS)) < 1e-3 for mz in df['missing_fragments'][1])
    # A list of m/z values with intensities is the same input as the peak dictionary
    df_list = rank_glycopeptide_structures('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1)', list(peaks), list(peaks.values()), charge = 2,
                                           structures = structures, fragmentation_method = 'HCD')
    assert df_list['score'].tolist() == df['score'].tolist()
    # All-zero intensities weigh the peaks equally, like a bare m/z list
    df_zero = rank_glycopeptide_structures('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1)', dict.fromkeys(peaks, 0), charge = 2, structures = structures,
                                           fragmentation_method = 'HCD')
    assert df_zero['score'].tolist() == rank_glycopeptide_structures('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1)', list(peaks), charge = 2,
                                                                      structures = structures, fragmentation_method = 'HCD')['score'].tolist()
    # No candidates, no glycan, or more structure combinations than max_candidates raise before anything is scored
    for pep_seq, mods, cands, message in (('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1)', [], 'candidate structures'),
                                          ('TPSAAYPMDR', 'M8(Oxidation)', None, 'no glycan'),
                                          ('TPSAAYPTDR', 'S3(Hex1HexNAc1NeuAc1);T8(Hex1HexNAc1NeuAc1)', [structures] * 2, 'max_candidates')):
        with pytest.raises(ValueError, match = message):
            rank_glycopeptide_structures(pep_seq, mods, peaks, charge = 2, structures = cands, max_candidates = 3)


def test_rank_glycopeptide_compositions():
    # Neu5Ac + Hex and Neu5Gc + Fuc weigh the same, so the sialic acid's oxonium ions (292/274 or 308/290) decide; and of two equal candidates, the one
    # that needs no isotope step off the precursor comes first
    pep = mass.calculate_mass(sequence = 'TPSAAYPTDR')
    res = {k: composition_to_mass({k: 1}) - composition_to_mass({}) for k in ('HexNAc', 'Neu5Ac', 'Neu5Gc')}
    peaks = {res['HexNAc'] + PROTON_MASS: 100, pep + PROTON_MASS: 70, pep + res['HexNAc'] + PROTON_MASS: 90}
    for sialic_acid in ('Neu5Ac', 'Neu5Gc'):
        spectrum = peaks | {res[sialic_acid] + PROTON_MASS: 80, res[sialic_acid] - WATER_MASS + PROTON_MASS: 60}
        df = rank_glycopeptide_compositions('TPSAAYPTDR', ['H1N1A1', 'N1F1G1'], spectrum, charge = 2, glycosite = 2, fragmentation_method = 'HCD')
        assert df['composition'][0].get(sialic_acid) == 1 and df['sialic_evidence'].tolist() == [1.0, 0.0] and df['rank'].tolist() == [1, 2]
    df = rank_glycopeptide_compositions('TPSAAYPTDR', ['H1N1', 'H1N1'], peaks, charge = 2, glycosite = 2, isotope_errors = [1, 0])
    assert df['isotope_error'].tolist() == [0, 1]


def test_glycopeptide_structure_chance_categories():
    # As for compositions, a glycopeptide structure's fragments of kinds that were annotated no more often than by chance are not matched: backbone ions
    # with part of a glycan other than an N-glycan's core HexNAc, glycan-only ions above 1+, metal adducts and NH3 on backbone ions
    pep, glycan = 'EEQYNSTYR', 'GlcNAc(b1-2)Man(a1-3)[GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc'
    hexnac, hexose = mono_attributes['HexNAc']['mass']['HexNAc'], mono_attributes['Hex']['mass']['Hex']
    y5 = sum(AA_masses[a] for a in pep[4:]) + WATER_MASS + PROTON_MASS
    b4 = sum(AA_masses[a] for a in pep[:4]) + PROTON_MASS
    peptide = sum(AA_masses[a] for a in pep) + WATER_MASS + PROTON_MASS
    peaks = {'y5 + HexNAc': y5 + hexnac, 'peptide + 0,2X HexNAc': peptide + 83.037114, 'y5 + HexNAc(2)': y5 + 2 * hexnac,
             'Hex(1)/HexNAc(1) 2+': (hexose + hexnac + 2 * PROTON_MASS) / 2, 'b4 - NH3': b4 - 17.026549, 'b4 + Na': b4 + 22.989218 - PROTON_MASS}
    hits = CandyCrumbs({'peptide': pep, 'glycans': [glycan], 'glycosites': [4]}, list(peaks.values()), charge = 3, fragmentation_method = 'HCD')
    found = {name: hits[m]['Domon-Costello nomenclatures'][0] if hits[m] else None for name, m in peaks.items()}
    assert found['y5 + HexNAc'][0] == ['y_5'] and found['y5 + HexNAc'][1][0].startswith('Y_1_')
    assert found['peptide + 0,2X HexNAc'][0] == ['Peptide'] and found['peptide + 0,2X HexNAc'][1][0].startswith('02X_1_')
    for name in ('y5 + HexNAc(2)', 'Hex(1)/HexNAc(1) 2+', 'b4 - NH3', 'b4 + Na'):
        assert found[name] is None, (name, found[name])


def test_rank_glycopeptide_sites():
    # c ions that carry the glycan only from its site on decide between the Ser/Thr; without them every placement ties, and an O-glycoprotease's prior puts
    # the glycan on the first residue
    pep, hexnac = 'TPSAAYPTDR', composition_to_mass({'HexNAc': 1}) - WATER_MASS
    cum = np.cumsum([AA_masses[a] for a in pep])
    c_ions = {cum[i - 1] + 17.026549 + PROTON_MASS + (hexnac if i > 7 else 0): 100.0 for i in range(2, 10)}
    df = rank_glycopeptide_sites(pep, ['HexNAc'], c_ions, charge = 2, glycan_class = 'O', fragmentation_method = 'ETD')
    assert df['glycosites'].tolist() == [[7], [2], [0]] and df['rank'].tolist() == [1, 2, 3]
    flat = rank_glycopeptide_sites(pep, ['HexNAc'], {hexnac + PROTON_MASS: 100.0}, charge = 2, glycan_class = 'O', fragmentation_method = 'ETD', protease = 'IMPa')
    assert flat['glycosites'][0] == [0] and flat['rank'].tolist() == [1, 2, 2] and flat['prior'].tolist() == [0.1, 0.0, 0.0]


def test_glycopeptide_nsite_pieces():
    # The reducing end of an N-glycan is HexNAc, so a piece of a composition still on the peptide has to contain one: the peptide + 0,2X Hex
    # mass is no longer the peptide with '02X Hex', and no Y/Z/X piece without HexNAc is offered
    m = round(mass.calculate_mass(sequence = 'EEQYNSTYR') + calculate_adduct_mass('C2H2O') + PROTON_MASS, 4)
    result = CandyCrumbs({'peptide': 'EEQYNSTYR', 'glycans': ['Hex3HexNAc4dHex1'], 'glycosites': [4]}, [m], charge = 3, simplify = False)
    pieces = [y for label in result[m]['Domon-Costello nomenclatures'] for y in label[1] if y[0] in 'YZ' or y[:3][-1] == 'X']
    assert pieces and all('HexNAc' in y for y in pieces)


def test_glycopeptide_chance_categories():
    # Fragment kinds that spectra of identified glycopeptides showed no more often than spectra shifted out of tolerance are not made for glycopeptide
    # compositions: backbone ions with part of a glycan other than an N-glycan's core HexNAc, glycan-only ions above 1+, metal adducts and NH3 on backbone ions
    pep, comp = 'EEQYNSTYR', {'Hex': 3, 'HexNAc': 4, 'dHex': 1}
    hexnac, hexose = mono_attributes['HexNAc']['mass']['HexNAc'], mono_attributes['Hex']['mass']['Hex']
    y5 = sum(AA_masses[a] for a in pep[4:]) + WATER_MASS + PROTON_MASS
    b4 = sum(AA_masses[a] for a in pep[:4]) + PROTON_MASS
    peptide = sum(AA_masses[a] for a in pep) + WATER_MASS + PROTON_MASS
    peaks = {'y5 + HexNAc': y5 + hexnac, 'peptide + 0,2X HexNAc': peptide + 83.037114, 'y5 + HexNAc(2)': y5 + 2 * hexnac,
             'Hex(1)/HexNAc(1) 2+': (hexose + hexnac + 2 * PROTON_MASS) / 2, 'b4 - NH3': b4 - 17.026549, 'b4 + Na': b4 + 22.989218 - PROTON_MASS}
    hits = CandyCrumbs({'peptide': pep, 'glycans': [comp], 'glycosites': [4]}, list(peaks.values()), charge = 3, fragmentation_method = 'HCD')
    found = {name: hits[m]['Domon-Costello nomenclatures'][0] if hits[m] else None for name, m in peaks.items()}
    assert found['y5 + HexNAc'] == [['y_5'], ['Y HexNAc(1)']]
    assert found['peptide + 0,2X HexNAc'] == [['Peptide'], ['02X HexNAc']]
    for name in ('y5 + HexNAc(2)', 'Hex(1)/HexNAc(1) 2+', 'b4 - NH3', 'b4 + Na'):
        assert found[name] is None, (name, found[name])


def test_glycopeptide_charge_reduced_ions():
    # Electron transfer leaves the intact glycopeptide charge-reduced with all its protons, and losing NH3 or acetamide from the radical; peaks no
    # fragment explains are named so for compositions and structures alike, and not with collisional activation
    pep, comp = 'TTNSATKI', {'Hex': 1, 'HexNAc': 1, 'Neu5Ac': 1}
    mass = sum(AA_masses[a] for a in pep) + WATER_MASS + 2 * (composition_to_mass(comp) - WATER_MASS)
    peaks = {(mass + 3 * PROTON_MASS) / 2: ('M_+H', 2), (mass + 3 * PROTON_MASS - 17.026549) / 2: ('M_+H|NH3', 2), mass + 3 * PROTON_MASS - 59.037114: ('M_2+H|C2H2O|NH3', 1)}
    for glycans in ([comp, comp], ['Neu5Ac(a2-3)Gal(b1-3)GalNAc'] * 2):
        for method in ('EThcD', 'HCD'):
            hits = CandyCrumbs({'peptide': pep, 'glycans': glycans, 'glycosites': [0, 1]}, list(peaks), charge = 3, fragmentation_method = method)
            for m, (mod, z) in peaks.items():
                if method == 'HCD':
                    assert hits[m] is None
                else:
                    assert hits[m]['Domon-Costello nomenclatures'][0] == [['Peptide', mod], ['M'], ['M']] and hits[m]['Fragment charges'][0] == z


def test_rank_glycopeptide_structures_published():
    # The published disialyl core 2 O-glycopeptide (doi:10.1007/s13361-018-1945-7) against every database structure of its composition: its
    # topology ranks first, tied only with its sialic acid linkage isomers, which have the same fragment masses
    gp = TEST_DICTS[6]
    df = rank_glycopeptide_structures('AVAVTLQSH', 'T5(Hex2HexNAc2NeuAc2)', gp['masses'], charge = gp['charge'], mass_threshold = 0.4,
                                      mass_threshold_ppm = None)
    top = [s[0] for s, r in zip(df['structures'], df['rank']) if r == 1]
    assert 'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)[Neu5Ac(a2-3)Gal(b1-3)]GalNAc' in top
    assert {re.sub(r'a2-[36]', 'a2-?', s) for s in top} == {
        'Neu5Ac(a2-?)Gal(b1-4)GlcNAc(b1-6)[Neu5Ac(a2-?)Gal(b1-3)]GalNAc'}


def test_rank_glycopeptide_structures_hexnac_evidence():
    # The HexNAc oxonium ions decide between O-glycan candidates with and without GlcNAc: a ratio (138 + 168) / (126 + 144) below 1 favours GalNAc-only
    # ones, above 2 those with GlcNAc, and without these ions nothing is weighed; O-site candidates start with GalNAc unless o_glycan_root is None
    base = {204.0867: 100.0, 366.1395: 60.0, 274.0921: 30.0, 292.1027: 40.0}
    galnac = rank_glycopeptide_structures('TTNSATKI', 'T1(Hex(2)HexNAc(2)NeuAc(1))', base | {126.0550: 30.0, 144.0656: 40.0, 138.0550: 10.0, 168.0656: 5.0},
                                          charge = 2, fragmentation_method = 'HCD')
    assert galnac['oxonium_ratio'][0] < 1 and not any('GlcNAc' in s[0] for s in galnac[galnac['rank'] == 1]['structures'])
    glcnac = rank_glycopeptide_structures('TTNSATKI', 'T1(Hex(2)HexNAc(2)NeuAc(1))', base | {126.0550: 5.0, 144.0656: 5.0, 138.0550: 60.0, 168.0656: 20.0},
                                          charge = 2, fragmentation_method = 'HCD')
    assert glcnac['oxonium_ratio'][0] > 2 and all('GlcNAc' in s[0] for s in glcnac[glcnac['rank'] == 1]['structures'])
    plain = rank_glycopeptide_structures('TTNSATKI', 'T1(Hex(2)HexNAc(2)NeuAc(1))', base, charge = 2, fragmentation_method = 'HCD')
    assert plain['oxonium_ratio'][0] is None and (plain['hexnac_evidence'] == 1).all()
    rooted = rank_glycopeptide_structures('TTNSATKI', 'T1(Hex(1)HexNAc(1)NeuAc(1))', base, charge = 2, fragmentation_method = 'HCD')
    every = rank_glycopeptide_structures('TTNSATKI', 'T1(Hex(1)HexNAc(1)NeuAc(1))', base, charge = 2, fragmentation_method = 'HCD', o_glycan_root = None)
    assert all(s[0].endswith('GalNAc') for s in rooted['structures']) and len(every) > len(rooted)


def test_candycrumbs_ms3_precursor():
    # MS3 of the Neu5Ac B1 ion (290.09) of Neu5Ac(a2-6)GalNAc-ol: its water loss is a fragment of that fragment, while the GalNAc-ol Y1/Z1 ions
    # and the 0,3A cross-ring of GalNAc (also 272.08) are not; MS3 of the Y1 ion keeps Z1 (its water loss), not the 2,4X/3,5A reading of 204
    g = 'Neu5Ac(a2-6)GalNAc'
    out = CandyCrumbs(g, [204.07, 222.12, 272.1], charge = -1, simplify = False, ms3_precursor = 290.11)
    assert out[204.07] is None and out[222.12] is None and out[272.1]['Domon-Costello nomenclatures'] == [
        ['B_1_Alpha', 'M_H2O']]
    out = CandyCrumbs(g, [204.07], charge = -1, simplify = False, ms3_precursor = 222.12)
    assert out[204.07]['Domon-Costello nomenclatures'] == [['Z_1_Alpha'], ['M_H2O', 'Y_1_Alpha']]
    # A precursor that is no fragment of the glycan leaves its MS3 spectrum unexplained
    assert all(v is None for v in CandyCrumbs(g, [222.12, 272.1], charge = -1, ms3_precursor = 500.0).values())


def test_supporting_ions():
    # The B3 ion Fuc(a1-2)Gal(b1-4)GlcNAc places the Fuc on the 6-arm Gal, as no other placement of it forms that mass, and is the peak that only this
    # structure (not its isomer with the Fuc on the core 1 Gal) explains; the Z1 ion of the 6-arm places its Gal and GlcNAc
    g, isomer = 'Fuc(a1-2)Gal(b1-4)GlcNAc(b1-6)[Gal(b1-3)]GalNAc', 'Fuc(a1-2)Gal(b1-3)[Gal(b1-4)GlcNAc(b1-6)]GalNAc'
    out = supporting_ions(g, {510.19: 30.0, 715.27: 100.0}, candidates = [isomer])
    residues = {e['name']: e for e in out['residues']}
    fuc = residues['Fuc(a1-2) on Gal(b1-4)GlcNAc']
    assert fuc['residue'] == 0 and [(s[0], s[2], s[3]) for s in fuc['support']] == [(510.19, ('B_3_Alpha',), 'Fuc(a1-2)Gal(b1-4)GlcNAc')]
    assert 'Fuc on Gal(b1-3)GalNAc' in fuc['support'][0][4] and not fuc['against']
    assert [s[2] for s in residues['GlcNAc(b1-6) on reducing-end GalNAc']['support']] == [('Z_1_Beta',)]
    # Nothing here tells where the core 1 Gal sits, so its alternatives stay open
    assert not residues['Gal(b1-3) on reducing-end GalNAc']['support'] and residues['Gal(b1-3) on reducing-end GalNAc']['open']
    assert [(c['structure'], [s[0] for s in c['support']], c['against']) for c in out['candidates']] == [(isomer, [510.19], [])]


def test_candycrumbs_alditol_cross_ring_alone():
    # 493.17 is the Neu5Ac-HexNAc B ion, not a fragment of the reduced GalNAc: a cross-ring of an alditol reducing end stands alone (0,4A of a
    # 6-linked branch, 919.09 in TEST_DICTS[3]), never next to another cleavage (1,5A + 1,5X, or 1,3A - C2H4O2, of Neu5Ac on GalNAc-ol)
    for g, a_ring in (('Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', '15A_2_Alpha'), ('Neu5Ac(a2-3)Gal(b1-3)GalNAc', '13A_3_Alpha')):
        out = CandyCrumbs(g, [493.17], charge = -1, simplify = False)[493.17]
        assert not out or not any(a_ring in x for x in out['Domon-Costello nomenclatures'])


@pytest.mark.parametrize('modification', ['reduced', 'unreduced', '2AA', '2AB', 'procainamide', 'PA'])
def test_candycrumbs_label_chain_ions(modification):
    # A reductively aminated reducing end gives X cross-rings of its open chain in negative mode (2AB N-glycans, GPST000651: 173.07 = 2AB + C3H2O - H+,
    # ^{0,3}X_0 - C2H5NO - H2O, in 89% of the HCD spectra); an alditol or a free reducing end does not, nor positive mode, permethylation or disable_X_cross_rings
    mass_tag = 0 if modification == 'unreduced' else calculate_adduct_mass(modification_formula_dict[modification])
    aminated = modification in ('2AA', '2AB', 'procainamide', 'PA')
    for glycan, pieces in (('Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', {'I_02X0': 'C4H7NO2', 'I_03X0-C2H5NO-H2O': 'C3H2O', 'I_04X0': 'C6H11NO4'}),
                           ('Fuc(a1-2)Gal(b1-4)Glc', {'I_02X0': 'C2H4O2', 'I_03X0-H2O': 'C3H4O2', 'I_04X0-2H2O': 'C4H4O2'})):
        expected = {name: mass_tag + calculate_adduct_mass(formula) - PROTON_MASS for name, formula in pieces.items()}
        out = CandyCrumbs(glycan, list(expected.values()), 0.003, charge = -1, mass_tag = mass_tag, simplify = False)
        found = {name for v in out.values() if v for d in v['Domon-Costello nomenclatures'] for name in d if name[2:] in CHAIN_IONS}
        assert found == (set(expected) if aminated else set())
        assert all((abs(get_fragment_mass(glycan, name, -1, mass_tag) - m) < 1e-6) if aminated else get_fragment_mass(glycan, name, -1, mass_tag) is None
                   for name, m in expected.items())
        # the residue has no N-acetyl group to lose, so Glc gives no acetamide loss
        assert glycan.endswith('GlcNAc') or get_fragment_mass(glycan, 'I_03X0-C2H5NO-H2O', -1, mass_tag) is None
        for kwargs, masses in ((dict(charge = 1), [m + 2 * PROTON_MASS for m in expected.values()]), (dict(charge = -1, disable_X_cross_rings = True), list(expected.values())),
                               (dict(charge = -1, sample_prep = 'permethylated'), list(expected.values()))):
            out = CandyCrumbs(glycan, masses, 0.003, mass_tag = mass_tag, simplify = False, **kwargs)
            assert not any(name[2:] in CHAIN_IONS for v in out.values() if v for d in v['Domon-Costello nomenclatures'] for name in d)
    if modification == '2AB':
        assert abs(get_fragment_mass('GlcNAc(b1-4)GlcNAc', 'I_03X0-C2H5NO-H2O', -1, mass_tag) - 173.0720) < 1e-3
        assert domon_costello_to_mpl(['I_03X0-C2H5NO-H2O']) == '$^{0,3}X_0 - C_{2}H_{5}NO - H_{2}O$'


# Settings that take generate_atomic_frags through its paths: cross-rings, global modifications (metal cation pairs, hydrogen shifts, CO2/SO4 of acidic
# residues), derivatization, mass tags (alditol, free reducing end, 2AB), multiply charged fragments, and positive mode's reducing-end A cross-rings
PRECHECK_CASES = [dict(glycan = 'Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', charge = -1, max_cleavages = 3),
                  dict(glycan = IGG_G0F, charge = -2, max_cleavages = 2, mass_tag = 0.0),
                  dict(glycan = 'Fuc(a1-2)Gal(b1-4)GlcNAc(b1-3)Gal(b1-4)Glc', charge = 1, max_cleavages = 3, sample_prep = 'permethylated'),
                  dict(glycan = 'Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)GlcNAc', charge = -2, max_cleavages = 2, x_rings = False,
                       mass_tag = calculate_adduct_mass(modification_formula_dict['2AB'])),
                  dict(glycan = 'GlcNS6S(a1-4)IdoA2S(a1-4)GlcNS', charge = -2, max_cleavages = 3),
                  dict(glycan = 'Neu5Ac(a2-3)Gal(b1-4)Glc', charge = 2, max_cleavages = 2, mass_tag = 0.0, max_global_mods = 2),
                  dict(glycan = 'Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', charge = -1, max_cleavages = 2, sample_prep = 'peracetylated', hydrogen_shifts = True)]


@pytest.mark.parametrize('case', PRECHECK_CASES)
def test_candycrumbs_precheck_matches_full_calculation(case):
    # generate_atomic_frags only makes the fragments near a wanted mass (all node sets checked at once in arrays, then the exact masses of the combinations
    # found), names them per terminal option and builds a fragment's graph only when asked: they have to be the fragments of the full calculation of every
    # fragment that are within the tolerance, with the names of their graphs
    glycan, charge, sample_prep, mass_tag = case['glycan'], case['charge'], case.get('sample_prep', 'underivatized'), case.get('mass_tag')
    nx_mono, _ = input_to_graph(glycopeptide_string_to_input(glycan))
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge, max_global_mods = case.get('max_global_mods', 1), sample_prep = sample_prep,
                                                            hydrogen_shifts = case.get('hydrogen_shifts', False))
    x_rings = X_cross_rings if case.get('x_rings', True) else []
    kwargs = dict(max_cleavages = case['max_cleavages'], mass_tag = mass_tag, charge = charge, sample_prep = sample_prep,
                  disable_A_cross_rings = ('non_reducing' if mass_tag == 0 else True) if charge > 0 else False)
    full = generate_atomic_frags(nx_mono, global_mods, special_residues, x_rings, fragment_masses = [], **kwargs)
    rng = random.Random(0)
    masses = sorted(rng.uniform(150, max(full)) for _ in range(60))
    found = generate_atomic_frags(nx_mono, global_mods, special_residues, x_rings, fragment_masses = masses, threshold = 0.3, **kwargs)
    wanted, chain_rank = extend_masses(masses, charge), list(rank_chains(nx_mono))

    def named(fragments):
        # fragments by mass (rounded to 5 decimals) and name, without those this close to the tolerance's edge, which rounding can put on the other side
        return Counter((m, tuple(name)) for m, frags in fragments.items() if abs(np.abs(wanted - m).min() - 0.3) > 1e-4
                       for name in subgraphs_to_domon_costello(nx_mono, frags, chain_rank = chain_rank))
    assert found and named(found) == named({m: frags for m, frags in full.items() if np.abs(wanted - m).min() < 0.3})
    fragments = [f for frags in found.values() for f in frags]
    assert subgraphs_to_domon_costello(nx_mono, fragments, chain_rank = chain_rank) == subgraphs_to_domon_costello(nx_mono, [f.graph for f in fragments],
                                                                                                               chain_rank = chain_rank)


# Glycopeptides through the same precheck: an N-glycan on a short peptide at z = 3 with two global modifications, a sialylated core 2 with X cross-rings,
# and two O-glycans (node sets holding both reducing ends take the full calculation)
GLYCOPEPTIDE_PRECHECK_CASES = [dict(glycopeptide = f'EEQYN*{IGG_G0F}*STYR', charge = 3, max_cleavages = 2, max_global_mods = 2),
                               dict(glycopeptide = 'AVAVT*Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-6)]GalNAc*LQSH', charge = 3, max_cleavages = 2, x_rings = True),
                               dict(glycopeptide = 'T*Gal(b1-3)GalNAc*TPPT*Neu5Ac(a2-3)Gal(b1-3)GalNAc*TATPIR', charge = 2, max_cleavages = 3)]


@pytest.mark.parametrize('case', GLYCOPEPTIDE_PRECHECK_CASES)
def test_candycrumbs_glycopeptide_precheck(case):
    # as test_candycrumbs_precheck_matches_full_calculation for glycopeptides, whose node sets have two roots (the full calculation takes the C-terminal
    # residue of the peptide part) and gain the water of the peptide's termini once, also with the ppm tolerance CandyCrumbs uses for them; and the names
    # of the peptide and glycan parts and the charge cap read off a Fragment are those of its graph
    input_dict = glycopeptide_string_to_input(case['glycopeptide'])
    nx_mono, pep_gr = input_to_graph(input_dict)
    charge = case['charge']
    global_mods, special_residues = get_initial_global_mods(nx_mono, charge, max_global_mods = case.get('max_global_mods', 1))
    x_rings = X_cross_rings if case.get('x_rings') else []
    kwargs = dict(max_cleavages = case['max_cleavages'], charge = charge, disable_A_cross_rings = True)
    full = generate_atomic_frags(nx_mono, global_mods, special_residues, x_rings, fragment_masses = [], **kwargs)
    rng = random.Random(0)
    masses = sorted([rng.choice(sorted(full)) * (1 + rng.gauss(0, 3e-6)) for _ in range(60)] + [rng.uniform(150, max(full)) for _ in range(60)])
    wanted = extend_masses(masses, charge)
    chain_rank = {}
    for prefix in sorted({x.split('-')[0] for x in nx_mono} - {'0'}, key = int):
        glyc = [x for x in nx_mono if x.split('-')[0] == prefix]
        chain_rank[prefix] = [(rank, [f'{prefix}-{n}' for n in chain]) for rank, chain in
                              rank_chains(nx.relabel_nodes(nx_mono.subgraph(glyc), {x: int(x.split('-')[1]) for x in glyc}))]
    for threshold_ppm in (None, 10):
        found = generate_atomic_frags(nx_mono, global_mods, special_residues, x_rings, fragment_masses = masses, threshold = 0.02, threshold_ppm = threshold_ppm,
                                      **kwargs)

        def described(fragments):
            # fragments within the tolerance by mass and what they are, without those this close to its edge, which rounding can put on the other side
            out = Counter()
            for m, frags in fragments.items():
                d = np.abs(wanted - m).min()
                limit = 0.02 if threshold_ppm is None else min(0.02, m * threshold_ppm / 1e6)
                if d < limit and abs(d - limit) > 1e-4:
                    out.update((m, tuple(f.nodes), tuple(f.terminals), tuple(f.mono_mods), tuple(tuple(a.items()) for a in f.atom_mods), f.global_mod) for f in frags)
            return out

        assert found and described(found) == described(full)
        fragments = [f for frags in found.values() for f in frags]
        assert [peptide_to_RF_nomenclature(pep_gr, f) for f in fragments] == [peptide_to_RF_nomenclature(pep_gr, f.graph) for f in fragments]
        assert get_glycan_cleavages(nx_mono, fragments, input_dict['glycosites'], chain_rank) == get_glycan_cleavages(nx_mono, [f.graph for f in fragments],
                                                                                                                    input_dict['glycosites'], chain_rank)
        assert [max_fragment_charge(f, peptide = True) for f in fragments] == [max_fragment_charge(f.graph, peptide = True) for f in fragments]


def test_candycrumbs_speed(request):
    # CandyCrumbs as the app annotates a spectrum (simplify, three cleavages, cross-rings and global modifications), as the pipeline scores a candidate
    # (two cleavages, neither) and supporting_ions, over the reference spectra and a 150-peak N-glycan spectrum, timed against the fastest run on this
    # machine (pytest's cache, which --cache-clear resets): printed like the scores, failing only beyond 4x, as test workers share the cores and what ran
    # before in the process moves these by up to 2x (after rank_glycopeptide_structures, which loads glycowork's df_glycan, numpy allocates slower)
    rng = random.Random(0)
    spectra = [(d['glycan_string'], d['masses'], d['charge'], d['label_mass']) for d in TEST_DICTS[:6]]
    spectra.append((IGG_G0F, sorted(round(rng.uniform(150, 1800), 2) for _ in range(150)), -2, 2.0156))
    runs = {'app': lambda g, masses, charge, mass_tag: CandyCrumbs(g, masses, mass_threshold = 0.5, charge = charge, mass_tag = mass_tag),
            'pipeline': lambda g, masses, charge, mass_tag: CandyCrumbs(g, masses, mass_threshold = 0.5, charge = charge, mass_tag = mass_tag, max_cleavages = 2,
                                                                        disable_X_cross_rings = True, disable_global_mods = True, simplify = False),
            'supporting_ions': lambda g, masses, charge, mass_tag: supporting_ions(g, {m: 100.0 - i for i, m in enumerate(masses)}, charge = charge,
                                                                                    mass_tag = mass_tag)}
    cache = getattr(request.config, 'cache', None)
    for name, run in runs.items():
        # after a run that sets up what is kept across calls (vocabulary, node sets, fragment options), as in a session, three timed ones with the peaks
        # shifted, so that no result kept for the same peaks (supporting_ions) is reused
        times = []
        for shift in (0.0, 0.001, 0.002, 0.003):
            start = time.perf_counter()
            for glycan, masses, charge, mass_tag in spectra:
                run(glycan, [m + shift for m in masses], charge, mass_tag)
            times.append(time.perf_counter() - start)
        times = times[1:]
        best = cache.get(f'candycrumbs/speed/{name}', None) if cache else None
        print(f"CandyCrumbs speed, {name}: {min(times) * 1000:.0f} ms" + (f", fastest on this machine {best * 1000:.0f} ms" if best else ''))
        if cache and (best is None or min(times) < best):
            cache.set(f'candycrumbs/speed/{name}', min(times))
        assert best is None or min(times) < 4 * best, f"CandyCrumbs ({name}) is {min(times) / best:.1f}x slower than this machine's fastest run"