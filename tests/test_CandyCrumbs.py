import pytest
import unittest
import re
from pyteomics import mass
from candycrunch.analysis import CandyCrumbs, get_fragment_mass, glycan_to_graph_monos, derivatization_sites, DERIVATIZATION_MASSES, \
    build_glycopeptide_input, rank_glycopeptide_structures, supporting_ions, WATER_MASS
from glycowork.motif.tokenization import glycan_to_mass, composition_to_mass, calculate_adduct_mass, HYDROGEN_MASS, PROTON_MASS

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
THRESHOLD = 0.4
TOP5_THRESHOLD = 0.7
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
    result = CandyCrumbs(test_dict['glycan_string'], test_dict['masses'], mass_threshold = 0.4, charge=test_dict['charge'],mass_tag=test_dict['label_mass'],max_cleavages=test_dict.get('max_cleavages',3))
    total_annotations = len(test_dict['annotations'])
    correct_annotations = 0
    assert len(result) == total_annotations
    for mass, expected_annotations in zip(test_dict['masses'], test_dict['annotations']):
        if not expected_annotations:
            total_annotations=total_annotations-1
            continue
        if mass in result:
            if result[mass]:
                predicted_annotations = result[mass]['Domon-Costello nomenclatures'][0]
                if normalize_annotations(predicted_annotations) == normalize_annotations(expected_annotations):
                    correct_annotations += 1
            elif not expected_annotations:
                correct_annotations += 1  # Credit for correctly predicting no annotations
    score = correct_annotations / total_annotations
    # Set a threshold for acceptable performance (e.g., 80% correct)
    print(f"Score: {score:.2f}, Threshold: {THRESHOLD}")
    assert score > THRESHOLD 

@pytest.mark.parametrize("test_dict", TEST_DICTS)
def test_candycrumbs_accuracy_top5(test_dict):
    result = CandyCrumbs(test_dict['glycan_string'], test_dict['masses'], 0.4, charge=test_dict['charge'],mass_tag=test_dict['label_mass'],simplify=False)
    total_annotations = len(test_dict['annotations'])
    correct_annotations = 0
    assert len(result) == total_annotations
    for mass, expected_annotations in zip(test_dict['masses'], test_dict['annotations']):
        if not expected_annotations:
            total_annotations=total_annotations-1
            continue
        if mass in result:
            if result[mass]:
                predicted_annotations = result[mass]['Domon-Costello nomenclatures']
                exp_norm = normalize_annotations(expected_annotations)
                if any(normalize_annotations(pred) == exp_norm for pred in predicted_annotations):
                    correct_annotations += 1
            elif not expected_annotations:
                correct_annotations += 1  # Credit for correctly predicting no annotations
    score = correct_annotations / total_annotations
    # Set a threshold for acceptable performance (e.g., 80% correct)
    print(f"Score: {score:.2f}, Threshold: {TOP5_THRESHOLD}")
    assert score > TOP5_THRESHOLD


DERIVATIZATION_GLYCANS = ['Neu5Ac(a2-3)Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc',
                          'Neu5Gc(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc', 'Kdn(a2-3)Gal(b1-4)Glc',
                          'Man6P(a1-2)Man(a1-2)Man',
                          'GalOS(b1-3)GalNAc', 'Neu5Ac9Ac(a2-3)Gal(b1-4)Glc', 'GlcNS6S(a1-4)IdoA2S(a1-4)GlcNS',
                          'Gal(b1-4)GlcN(a1-6)Glc', 'Xyl(b1-2)Man(b1-4)GlcNAc', 'Fuc(a1-2)Gal4S(b1-3)GalNAc']


@pytest.mark.parametrize("glycan", DERIVATIZATION_GLYCANS)
@pytest.mark.parametrize("sample_prep", ['underivatized', 'permethylated', 'peracetylated'])
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


def test_build_glycopeptide_input_modifications():
    # Met oxidation is a peptide residue, not a glycan; an unknown modification raises instead of becoming a massless glycan
    gp = build_glycopeptide_input('EMTSPGTPAR', 'M2(Oxidation);T3(HexNAc(1)Hex(1))')
    assert gp == {'peptide': 'EmTSPGTPAR', 'glycans': [{'HexNAc': 1, 'Hex': 1}], 'glycosites': [2]}
    y0 = round(mass.calculate_mass(sequence = 'EMTSPGTPAR') + calculate_adduct_mass('O') + PROTON_MASS, 4)
    assert CandyCrumbs(gp, [y0], charge = 2)[y0]['Domon-Costello nomenclatures'][0] == [['Peptide'], ['loss of glycan 1']]
    with pytest.raises(ValueError):
        build_glycopeptide_input('EMTSPGTPAR', 'S4(Phospho);T3(HexNAc(1)Hex(1))')


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


def test_glycopeptide_nsite_pieces():
    # The reducing end of an N-glycan is HexNAc, so a piece of a composition still on the peptide has to contain one: the peptide + 0,2X Hex
    # mass is no longer the peptide with '02X Hex', and no Y/Z/X piece without HexNAc is offered
    m = round(mass.calculate_mass(sequence = 'EEQYNSTYR') + calculate_adduct_mass('C2H2O') + PROTON_MASS, 4)
    result = CandyCrumbs({'peptide': 'EEQYNSTYR', 'glycans': ['Hex3HexNAc4dHex1'], 'glycosites': [4]}, [m], charge = 3, simplify = False)
    pieces = [y for label in result[m]['Domon-Costello nomenclatures'] for y in label[1] if y[0] in 'YZ' or y[:3][-1] == 'X']
    assert pieces and all('HexNAc' in y for y in pieces)


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