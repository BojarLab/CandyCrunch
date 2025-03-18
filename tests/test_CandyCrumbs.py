import pytest
import unittest
from CandyCrunch.analysis import CandyCrumbs

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
'charge':+2,
'label_mass':0,
'masses':[156.076, 204.086, 228.09, 243.108, 259.176, 274.092, 292.102, 355.148, 366.14, 425.178, 454.155, 468.232, 564.797, 657.234, 666.339, 747.365, 791.372, 828.393, 860.311, 892.912, 925.51, 973.939, 990.897, 1022.361, 1026.430, 1075.959, 1128.589, 1236.557, 1267.642, 1290.641, 1313.451, 1771.752, 1884.831, 1981.806],
'annotations':[[['y_1'],['loss of glycan 1']],[['No_Peptide'],['C_4_Alpha', 'Z_1_Beta', 'Z_1_Alpha']],[['z_2'],['loss of glycan 1']],[['y_2'],['loss of glycan 1']],[['c_3'],['loss of glycan 1']],[['No Peptide'], ['B_1_Beta', 'M_H2O']],[['No Peptide'], ['B_1_Beta']],[['z_3'], ['loss of glycan 1']],[['No Peptide'], ['Y_3_Alpha', 'B_3_Alpha']],[['w_4'], ['loss of glycan 1']],[['No Peptide'], ['B_2_Alpha']],[['z_4'], ['loss of glycan 1']],[['Peptide'], ['Y_1_Beta', 'Y_1_Alpha']],[['No Peptide'], ['B_3_Alpha']],[['Peptide'], ['Y_1_Beta', 'Y_2_Alpha']],[['Peptide'], ['Y_1_Beta', 'Y_3_Alpha']],[['Peptide'], ['Y_1_Alpha']],[['Peptide'], ['Y_2_Beta', 'Y_3_Alpha']],[['No Peptide'], ['Y_1_Beta']],[['Peptide'], ['Y_1_Beta']], [['Peptide'], ['Y_0_Alpha']],[['Peptide'], ['Y_3_Alpha']], [['z_6'], ['M']], [['No Peptide'], ['Y_3_Alpha']], [['z_7'], ['M']], [['z_8'], ['M']], [['Peptide'], ['Y_1_Beta', 'Y_1_Alpha']], [], [], [['Peptide'], ['Y_1_Alpha', 'Y_2_Beta']], [['No Peptide'], ['B_4_Alpha']], [['c_5'], ['M']], [['c_6'], ['M']], [['z_6'], ['M']]],
'ref':'https://doi.org/10.1007/s13361-018-1945-7'
}
]
THRESHOLD = 0.1
TOP5_THRESHOLD = 0.63
@pytest.mark.parametrize("test_dict", TEST_DICTS)
def test_candycrumbs_accuracy(test_dict):
    result = CandyCrumbs(test_dict['glycan_string'], test_dict['masses'], 0.4, charge=test_dict['charge'],mass_tag=test_dict['label_mass'],max_cleavages=test_dict.get('max_cleavages',3))
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
                if predicted_annotations == sorted(expected_annotations):
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
                if expected_annotations in predicted_annotations:
                    correct_annotations += 1
            elif not expected_annotations:
                correct_annotations += 1  # Credit for correctly predicting no annotations
    
    score = correct_annotations / total_annotations
    # Set a threshold for acceptable performance (e.g., 80% correct)
    print(f"Score: {score:.2f}, Threshold: {TOP5_THRESHOLD}")
    assert score > TOP5_THRESHOLD 