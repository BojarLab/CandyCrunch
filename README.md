<p align="center">
  <img src="https://raw.githubusercontent.com/BojarLab/CandyCrunch/main/images/candycrunch_logo_light_banner_pt.svg" style="height:100%;width:100%;">
</p>


# CandyCrunch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7940046.svg)](https://doi.org/10.5281/zenodo.7940046)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/bojarlab/candycrunch/blob/main/LICENSE)
[![Tests](https://github.com/BojarLab/CandyCrunch/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/BojarLab/CandyCrunch/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BojarLab/CandyCrunch/graph/badge.svg)](https://codecov.io/gh/BojarLab/CandyCrunch)

## What is CandyCrunch?
**CandyCrunch** is a package for predicting glycan structure from LC-MS/MS data. It contains the CandyCrunch model, along with the rest of the inference pipeline and downstream spectrum processing tools. These are further described in our manuscript [Urban et al. (2024)](https://www.nature.com/articles/s41592-024-02314-6) &ndash; ***Predicting glycan structure from tandem mass spectrometry via deep learning*** published in Nature Methods.

The desktop app runs CandyCrunch on one or several Thermo .raw, Bruker .d (timsTOF and ion trap), mzML, mzXML, mgf, .cc, or xlsx files without any code,
```
#### PyPI:
```bash
pip install candycrunch
```
## `CandyCrunch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/CandyCrunch/blob/main/CandyCrunch.ipynb)
If you are looking for a **convenient** and **easy-to-run** version of the code that does not require any local installations, we have also created a Google Colaboratory notebook.  
The notebook contains an example pipeline ready to run, which can be copied, executed, and customised in any way.  
The example file included in the notebook is the same as in `examples/` and is ready for use in the notebook workflow. 

## Using CandyCrunch &ndash; Desktop app
No Python or command line needed: download `CandyCrunch-<version>-Windows-Setup.exe` (Windows) or `CandyCrunch-<version>-macOS.dmg` (macOS) from the [latest release](https://github.com/BojarLab/CandyCrunch/releases/latest), install it, and start CandyCrunch from the Start menu or Applications. The Windows installer needs no administrator rights; afterwards, Thermo .raw, mzML, mzXML, and mgf files can be opened with CandyCrunch from their right-click menu, and saved results open with a double-click. The macOS app is not notarized, so on first start macOS asks you to allow it once under System Settings > Privacy & Security > Open Anyway.

Python users get the same app with
```bash
pip install "candycrunch[gui]"
candycrunch_gui
```
The desktop app runs CandyCrunch on one or several Thermo .raw, Bruker timsTOF .d, mzML, mzXML, mgf, or xlsx files without any code, with a progress bar that shows how far a run is and about how long it will still take. Results come as a table with SNFG drawings, and each glycan peak shows its annotated MS² spectrum (Save image… writes it at print size as PNG, SVG, or PDF, with the m/z of every labeled peak), fragment table, candidate structures and position on a retention time vs. m/z map. Several runs are harmonized into a feature table with per-run abundances. Predictions can be curated (assign another candidate, enter a structure, exclude a peak), exported as CSV or Excel (optionally with SNFG drawings), and saved to reopen later without rerunning. The CandyCrumbs tab annotates any peak list with any glycan, composition, or glycopeptide. To try the app without data of your own, File > Add example run adds a small bundled O-glycan run.

## Using CandyCrunch &ndash; Command line interface:
If you would like to run our main inference function from the command line, you can do so using the `candycrunch_predict` command included in this repository.

#### Requires at a minimum:
<pre>
--spectra_filepath, type=string(s): one or more filepaths to Thermo .raw files, Bruker timsTOF .d folders, mzML, mzXML, or mgf files or .cc/.xlsx files from extract_spectra, and/or folders of .raw, .d, mzML, mzXML, mgf, or .cc files (the .xlsx files in a folder are not used); several files are processed together via `wrap_inference_batch` <br />
--glycan_class, type=string(s): the glycan class measured ("N", "O", "lipid"/"free"), or several measured together in one run (e.g., `--glycan_class N O`) <br />
--output, type=string: an output filepath ending with `.csv` or `.xlsx`
</pre>

<details>
<summary>

#### Optional arguments:
</summary>
<pre>
--mode, type=string: mass spectrometry mode; options are 'negative' or 'positive'; default: 'negative' <br />
--max_charge, type=int: maximum precursor charge to consider; default:3 <br />
--sample_prep, type=string: options are 'underivatized', 'permethylated', and 'peracetylated'; default:'underivatized' <br />
--sialic_derivatization, type=string: linkage-specific sialic acid derivatization of underivatized samples, or of samples permethylated after it; options are 'EE' (ethyl esterification), 'EEA' (ethyl esterification and amidation), 'ME' (methyl esterification), 'DMA' (dimethylamidation), 'SALSA', 'GQ' (glycoqueuing: aniline on every sialic acid, which tells no linkage; with GirardP), and 'GQ-d5' (d5-aniline on a2-6); precursor masses then tell a2-6 from a2-3/8/9 sialic acids; default:None <br />
--reducing_end_variants, type=string(s): other reducing ends the glycans may carry: 'free' (glycans that escaped reduction or labelling, counted for their glycan) and/or 'ethyl_glycoside' (rows of their own); give none of them to turn this off; default: free, plus ethyl_glycoside for lipid <br />
--adducts, type=string(s): ions that carry the charge of the precursors in place of protons, e.g. Li+ for lithium salts in the mobile phase ([M+Li]+, [M+2Li]2+, ...); the usual adducts (Na+, K+, NH4+ in positive, acetate, formate, HCO3- in negative mode) are always considered; default:None <br />
--modification, type=string: reducing-end modification of glycans; options are 'reduced', 'reduced_D' (NaBD4), 'unreduced' (free reducing end), '2AA', '2AB', 'procainamide', 'PA', 'APTS', 'aminoxyTMT', 'GirardP', or 'custom'; default:'reduced'
| 
|--mass_tag, type=float: only if modification = "custom", mass of custom reducing end tag ; default:None <br />
--lc, type=string: type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC' <br />
--trap, type=string: type of mass detector used; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear' <br />
--rt_min, type=float: whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0 (runs longer than 20 min then skip their first 2 min) <br />
--rt_max, type=float: whether only spectra up to a maximum retention time (in minutes) should be considered; default:0 (runs longer than 40 min then skip their last 10%) <br />
--rt_diff, type=float: maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0 <br />
--spectra, type=bool: whether to also output the actual spectra used for prediction, as a peak_d column; default:False <br />
--get_missing, type=bool: whether to also organize spectra without a matching prediction but a valid composition; default:False <br />
--filter_out, type=string(s): space-separated monosaccharide or modification types; compositions containing any of them are not considered (e.g., Pen if you know there is none); the list replaces the default, and giving none of them filters nothing; Ac is always left out of permethylated and peracetylated samples; default: HexA <br />
--ppm_thresh, type=float: mass tolerance in ppm, used for grouping spectra, matching compositions, and filtering by ppm error; default:300 <br />
--pred_thresh, type=float: prediction confidence threshold used for filtering; default:0.01 <br />
--crumbs_thresh, type=float: CandyCrumbs annotation score a prediction has to exceed to be kept (2 lower for files with orbitrap or timsTOF survey scans); default:3 <br />
--extra_thresh, type=float: prediction confidence threshold at which to allow cross-class predictions; default:0.2 <br />
--frag_num, type=int: number of top fragments to show per spectrum; default:100 <br />
--supplement, type=bool: whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True <br />
--experimental, type=bool: whether to impute missing predictions via database searches etc.; default:True
|
|--taxonomy_level, type=string: taxonomic level used to restrict the glycan database; default:'Class' <br />
|--taxonomy_filter, type=string: taxon at taxonomy_level used to restrict the glycan database; default:'Mammalia' <br />
--intra_cat_thresh, type=float: required for several files, minutes the RT of a structure can differ from the mean of its group across files <br />
--top_n_isomers, type=int: several files only, number of isomer groups at each composition to retain in each file; default:5 <br />
--n_jobs, type=int: number of processes: files processed in parallel, and fragment scoring and supporting peaks within a file; default:1 <br />
--plot_glycans, type=bool: whether to write the output as an .xlsx file that contains SNFG images of all top1 predictions (a `.csv` output additionally gets an `.xlsx` of the same name); default:False
</pre>
</details>

#### Basic usage
> [!IMPORTANT]  
> Users must install CandyCrunch using pip before running the commands below
```console
/Users/xurbja $ candycrunch_predict --spectra_filepath path_to_my_files/file.mzML --glycan_class 'O' --output path_to_my_outputs/output_file.csv 
```

Several files, or a folder of Thermo .raw, Bruker timsTOF .d, mzML, mzXML, or mgf files, are harmonized across runs with `wrap_inference_batch`. The output file then holds a feature table with one row per isomer group (top1 prediction, consensus m/z and RT, number of files with MS2 evidence) and its relative abundance and evidence (MS2 or MS1-only) in every file, and each file's predictions are written next to it as `output_file_<file name>.csv`:
```console
/Users/xurbja $ candycrunch_predict --spectra_filepath path_to_my_files/ --glycan_class 'O' --intra_cat_thresh 1.75 --top_n_isomers 2 --output path_to_my_outputs/output_file.csv
```

## Using CandyCrunch &ndash; LC-MS/MS glycan annotation
### `wrap_inference` (in `candycrunch.prediction`) <br>
Wrapper function to predict glycan structures from raw LC-MS/MS spectra using `CandyCrunch`  
  
#### Requires at a minimum:  
<pre>
- spectra_filepath, type = string: a filepath to a Thermo .raw file, a Bruker .d folder, an mzML, mzXML, or mgf file, or a .cc or .xlsx file from extract_spectra, or a dataframe of spectra as load_spectra_filepath returns them <br />
- glycan_class,type = string or list: the glycan class measured ("N", "O", "lipid"/"free"), or a list of the classes measured together in one run (e.g., `["N", "O"]` for N- and O-glycans released from one sample)
</pre>
Thermo .raw, Bruker timsTOF .d, mzML, mzXML, and mgf files are internally processed into extracted spectra; .raw files are read directly, and timsTOF .d folders of DDA-PASEF runs (also those of early timsTOF software) as well, without any conversion or vendor software (survey frames are summed over their ion mobility scans, m/z come from the file's own TOF calibration, and the unfragmented precursor is removed from MS2 spectra; diaPASEF and timsTOF data without ion mobility (.tsf) are not supported). Bruker ion trap .d folders (esquire, HCT, amaZon, with an Analysis.yep) are read directly as well, from the centroided line spectra the instrument stores. With accurate (orbitrap or timsTOF) survey scans, a peak the model has no structure for gets a database structure of its composition (evidence 'composition') if the precursor m/z pins that composition and its fragments back the structure up. mzML files converted from timsTOF data work with or without combining ion mobility scans in msconvert. xlsx files need to be already extracted in the format as the example file in `examples/`. `extract_spectra(spectra_filepath)` (in `candycrunch.prediction`) writes such an xlsx from a .raw, .d, mzML, mzXML or mgf file, with precursor m/z already refined from MS1, so it gives the same predictions as the original file at a fraction of the size (only quantification falls back to precursor intensity, as MS1 scans are not kept). `extract_spectra(spectra_filepath, file_format = 'cc')` instead writes a compact binary .cc file that keeps everything CandyCrunch reads from the original file (MS2 and MS3 spectra exactly as loaded, the detected ion mode and analyzers, and the survey scans), so predictions and XIC quantification from it match the original (a 38 MB mzML becomes a 4.8 MB .cc, read in a tenth of a second); it is a zip of numpy arrays that `numpy.load` opens.
<details>
<summary>

#### Optional arguments:
</summary>
<pre>
model, type=Pytorch object or callable: the trained CandyCrunch model (default), or any structure predictor as a callable that takes a dataframe with one row per (pooled spectrum, candidate composition) and returns a list of (IUPAC structure, probability) tuples per row <br />
glycans, type=list: ordered list of glycans used to train CandyCrunch which can be predicted by the model <br />
bin_num, type=int: number of bins to separate the ms2 spectrum into; default:2048 <br />
frag_num, type=int: number of top fragments to show in df_out per spectrum; default:100 <br />
max_charge, type=int: maximum signed charge to consider; the sign sets the mass spectrometry mode (negative or positive); default:-3 <br />
modification, type=string: reducing-end modification of glycans; options are 'reduced', 'reduced_D', '2AA', '2AB', 'procainamide', 'PA', 'APTS', 'aminoxyTMT', 'GirardP', or 'custom' (without mass_tag for a free reducing end); default:'reduced'
| 
|--mass_tag, type=float: only if modification = "custom", mass of custom reducing end tag ; default:None <br />
lc, type=string: type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC' <br />
trap, type=string: type of mass detector used; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear' <br />
rt_min, type=float: whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0 (runs longer than 20 min then skip their first 2 min) <br />
rt_max, type=float: whether only spectra up to a maximum retention time (in minutes) should be considered; default:0 (runs longer than 40 min then skip their last 10%) <br />
rt_diff, type=float: maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0 <br />
pred_thresh, type=float: prediction confidence threshold used for filtering; default:0.01 <br />
temperature, type=float: the temperature factor used to calibrate logits; default:1.15 <br />
spectra, type=bool: whether to also output the actual spectra used for prediction; the result is then a tuple (dataframe, list of peak dictionaries); default:False <br />
get_missing, type=bool: whether to also organize spectra without a matching prediction but a valid composition; default:False <br />
filter_out, type=set: set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); Ac is always left out of permethylated and peracetylated samples; default:{'HexA'} <br />
ppm_thresh, type=float: mass tolerance in ppm, used for grouping spectra, matching compositions, and filtering by ppm error; default:300 <br />
crumbs_thresh, type=float: CandyCrumbs annotation score a prediction has to exceed to be kept (2 lower for files with orbitrap or timsTOF survey scans); default:3 <br />
sample_prep, type=string: options are 'underivatized', 'permethylated', and 'peracetylated'; default:'underivatized' <br />
extra_thresh, type=float: prediction confidence threshold at which to allow cross-class predictions (e.g., predicting N-glycans in O-glycan samples); default:0.2 <br />
supplement, type=bool: whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True <br />
experimental, type=bool: whether to impute missing predictions via database searches etc.; default:True
|
|--mass_dic, type=dict: only if experimental = "True", dictionary of form mass : list of glycans; will be generated internally 
|
|--taxonomy_level, type=string: taxonomic level used to restrict the glycan database; default:'Class'
|
|--taxonomy_filter, type=string: taxon at taxonomy_level used to restrict the glycan database; default:'Mammalia'
|
|--df_use, type=DataFrame: only if experimental = "True", sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan <br />
plot_glycans, type=bool: whether to save an .xlsx file with SNFG images of all top1 predictions next to spectra_filepath, named like it plus _output; default:False <br />
sialic_derivatization, type=string or tuple: linkage-specific sialic acid derivatization of underivatized samples, or of samples permethylated after it, 'EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5', or a pair of glycowork SIALIC_PRODUCTS for a2-6 and a2-3/8/9 sialic acids; precursor masses then tell a2-6 from a2-3/8/9 sialic acids; default:None <br />
reducing_end_variants, type=dict: other reducing ends the glycans may carry, as name: group on the free glycan's reducing end (formula or mass; 0 for a free reducing end), e.g. {'free': 0} or {'ethyl glycoside': 'C2H4'}; default:None (free reducing ends of reduced or labeled glycans, plus ethyl glycosides for 'lipid') <br />
adducts, type=string or list: ions of glycowork's adduct table that carry the charge of the precursors in place of protons, e.g. 'Li+' for lithium salts in the mobile phase; their compositions come before the protonated ones; the usual adducts (Na+, K+, NH4+ or acetate, formate, HCO3-) are always considered; default:None <br />
n_jobs, type=int: number of processes for fragment scoring and supporting peaks (a script using n_jobs > 1 needs if __name__ == '__main__' on Windows and macOS); default:1

</pre>
</details>

#### Basic usage
|     m/z | top1_pred                | predictions                            | composition                        |   num_spectra |   charge |    RT |   annotation_score |   rel_abundance | top_fragments                       | evidence   | adduct   |   ppm_error | notes   | GlyTouCan_ID   | supported_by                                                    |
|--------:|:-------------------------|:---------------------------------------|:-----------------------------------|--------------:|---------:|------:|-------------------:|----------------:|:------------------------------------|:-----------|:---------|------------:|:--------|:---------------|:----------------------------------------------------------------|
| 384.184 | Gal(b1-3)GalNAc          | [('Gal(b1-3)GalNAc', 0.8329)]          | {'Hex': 1, 'HexNAc': 1}            |             8 |       -1 |  6.76 |                  6 |            0.2  | [204.0339, 155.9961, 178.9275, ...] | strong     |          |       85.15 |         | G15559XW       |                                                                 |
| 530.211 | Fuc(a1-2)Gal(b1-3)GalNAc | [('Fuc(a1-2)Gal(b1-3)GalNAc', 0.9339)] | {'Hex': 1, 'HexNAc': 1, 'dHex': 1} |            13 |       -1 | 19.91 |                  6 |            0.81 | [205.0392, 204.032, 246.9845, ...]  | strong     |          |        3.81 |         | G05545HU       | 1/1 residues supported: Fuc(a1-2) on Gal (324.9 C2a, 307.1 B2a) |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

annotation_score is the CandyCrumbs score of the top1 structure's fragments, rel_abundance the peak's share (in %) of the summed abundance of all rows, evidence where the structure comes from ('strong': the model; 'medium': a biosynthetic intermediate of the run's other glycans; 'weak': a database glycan of that mass; 'composition': a database structure of a composition that accurate survey scans and fragments confirm), and supported_by the peaks that place each residue of the structure (see `supporting_ions`).
```

<details>
<summary>

#### This is what a truncated example of `annotated_spectra_df` would look like
</summary>

|         | predictions                                                                                                                             | composition             |   num_spectra |   charge |    RT | top_fragments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |   adduct | evidence   |
|--------:|:----------------------------------------------------------------------------------------------------------------------------------------|:------------------------|--------------:|---------:|------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------:|:-----------|
| 384.157 | [('Gal(b1-3)GalNAc', 0.9625)]                                                                                                           | {'Hex': 1, 'HexNAc': 1} |             8 |       -1 |  6.75 | [204.0202, 222.1731, 156.0888, 179.031, 160.7594, ...]                                                                                                                                                                                                                                                                                                                                                                                                                                                  |      nan | strong     |
| 425.036 | [('GalNAc(a1-3)GalNAc', 0.7947394540942927), ('GlcNAc(b1-3)GalNAc', 0.17965260545905706), ('HexNAc(?1-3)GalNAc', 0.025607940446650122)] | {'HexNAc': 2}           |             2 |       -1 | 15.88 | [381.005, 389.9802, 406.871, 326.8488, 212.01, ...]                         |      nan | strong     |
| ... | ...                                                                                                            | ... |            ... |       ... | ... | ... |      ... | ...     |                                                                                                                           | ...                     |           ... |      ... | ...   | ...                                                |      ... | ...     |

</details>

### `wrap_inference_batch` (in `candycrunch.prediction`) <br>
Wrapper function to predict glycan structures from multiple LC-MS/MS files using CandyCrunch. <br />
This function works similarly to `wrap_inference` except a list of filenames is provided and a tuple is returned: a feature table and a dictionary of output DataFrames, one for each input file, keyed by their filenames. <br />
The feature table has one row per isomer group with a structure (also those without one with get_missing; a glycan that files report at different charge states is one row) with its top1 prediction, consensus m/z, RT, charge, composition, GlyTouCan ID and the number of files with MS2 evidence, followed by its relative abundance in each file and an `evidence_<file>` column per file ('strong'/'weak' from MS2, 'composition' for a database structure of a composition that accurate survey scans and fragments confirm, 'ms1_only' if it was gap-filled from MS1). The first column holds the glycans, so the table can go straight into glycowork's `get_differential_expression` with the file columns as groups. <br />
Precursors are clustered by m/z across files within the mass tolerance, and glycan predictions are assigned to groups based on the prediction most often ranked first in the group across files (ties go to the highest probability summed over the group), which rows of the group's composition without a prediction also take. Useful for retention time correction but cannot correct LC runs in cases where noise exceeds signal. <br />

The algorithm operates under the assumption that the same structures should elute at a given RT ± intra_cat_thresh.
At each composition, an isomer group is kept if it is among the top_n_isomers groups of any file, preferring groups with a prediction, then groups seen in more files, then groups more abundant in that file. For files with MS1 scans (.raw, .d, mzML, mzXML), an isomer group without MS2 in a file is filled in from MS1 if the file shows a genuine chromatographic peak for it within intra_cat_thresh that is at least as intense as the weakest precursor that got MS2 in that file <br />

#### Requires at a minimum:  
<pre>
- spectra_filepath_list, type = list: list of filepaths to Thermo .raw files, Bruker .d folders, mzML, mzXML, or mgf files and/or .cc/.xlsx files <br />
- glycan_class, type = string or list: the glycan class measured ("N", "O", "lipid"/"free"), or a list of the classes measured together in one run <br />
- intra_cat_thresh, type = float: minutes the RT of a structure can differ from the mean of a group. <br />
</pre>

#### Optional arguments:
- top_n_isomers, type = int: number of isomer groups at each composition to retain in each file; default:5 <br />
- n_jobs, type = int: number of files to process in parallel (in separate processes), and of processes for the supporting peaks of each file's rows; default:1 <br />
- library_thresh, type = float: MS2 cosine at which a peak without a structure takes the structure of a peak at the same m/z and charge in another file; None turns this off; default:0.9 <br />
See `wrap_inference`  <br />
<br />  
```python
from candycrunch.prediction import wrap_inference_batch
spectra_filepath_list = ["C:/myfiles/my_spectra_exp1.mzML","C:/myfiles/my_spectra_exp2.mzML",
                          "C:/myfiles/my_spectra_exp3.mzML","C:/myfiles/my_spectra_exp4.mzML"]
combined_batch, results_dict = wrap_inference_batch(spectra_filepath_list, 'O', 1.75, 2)
```
<details>
<summary>
  
#### This is what `results_dict` would look like
</summary>
<pre>{'my_spectra_exp1': pd.DataFrame(...),
 'my_spectra_exp2': pd.DataFrame(...),
 'my_spectra_exp3': pd.DataFrame(...),
 'my_spectra_exp4': pd.DataFrame(...)}</pre>
</details>


## Using CandyCrumbs &ndash; MS2 fragment annotation
### `CandyCrumbs` (in `candycrunch.analysis`) <br>
Wrapper function to annotate MS2 fragments using `CandyCrumbs`  
  
#### Requires at a minimum:
<pre>
- input_string, type=string or dict: a glycan in IUPAC-condensed format, a composition, or a glycopeptide as 'PEPT*glycan*IDE' or a dict with 'peptide', 'glycans', and 'glycosites' (0-based) <br />
- fragment_masses, type=list: all observed masses which are to be annotated with possible fragment names
</pre>
```python
from candycrunch.analysis import CandyCrumbs
condensed_iupac_glycan = 'Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc'
ms2_fragment_masses = [425.07, 443.07, 546.19, 1216.32]
annotated_fragments_dict = CandyCrumbs(condensed_iupac_glycan, fragment_masses = ms2_fragment_masses, mass_threshold = 1)
```

<details>
<summary>

#### Optional arguments:
</summary>
<pre>
mass_threshold, type=float: the maximum tolerated mass difference between observed masses and possible fragments; default:None (0.5 Da, or 10 ppm for glycopeptides) <br />
mass_threshold_ppm, type=float: a ppm bound on each match on top of mass_threshold, for accurate MS2; default:None <br />
intensities, type=list: intensities of fragment_masses, which help choose between annotations; default:None <br />
max_cleavages, type=int: maximum number of allowed concurrent cleavages per possible fragment; default:3 <br />
simplify, type=bool: whether to select a single fragment for each mass based on mass difference, number of cleavages, and other fragments; default:True <br />
charge, type=int: the charge state of the precursor ion (singly-charged, doubly-charged, etc.); default:-1 <br />
mass_tag, type=float: mass of the glycan label or reducing end modification; default:None (alditol, 2 H) <br />
iupac, type=bool: whether to also return the fragment sequence in IUPAC-condensed nomenclature; default:False <br />
sample_prep, type=string: options are 'underivatized', 'permethylated', and 'peracetylated'; default:'underivatized' <br />
glycan_class, type=string: 'N', 'O', 'free' or 'lipid', which restricts cross-ring fragments to those seen in that class; default:None <br />
fragmentation_method, type=string: 'CID', 'HCD', 'ETD', 'ECD', 'EThcD', or 'ETciD', which sets the peptide ion types of glycopeptides and the hydrogen shifts of radical fragments; default:None <br />
sialic_derivatization, type=string or tuple: as in wrap_inference; every sialic acid then needs its linkage; default:None <br />
ms3_precursor, type=float: m/z of the MS2 fragment isolated for an MS3 spectrum, whose fragments are then annotated as fragments of it; default:None <br />
</pre>
</details>


<details>
<summary>
  
#### This is what `annotated_fragments_dict` would look like
</summary>
<pre>{425.07: {'Theoretical fragment masses': [425.13007],
  'Domon-Costello nomenclatures': [['02A_3_Alpha', 'M_H2O']],
  'Fragment charges': [-1]},
 443.07: {'Theoretical fragment masses': [443.14062],
  'Domon-Costello nomenclatures': [['02A_3_Alpha']],
  'Fragment charges': [-1]},
 546.19: {'Theoretical fragment masses': [546.20393],
  'Domon-Costello nomenclatures': [['Y_2_Alpha', 'Y_3_Beta']],
  'Fragment charges': [-1]},
 1216.32: {'Theoretical fragment masses': [1216.44723],
  'Domon-Costello nomenclatures': [['M_C2H4O2']],
  'Fragment charges': [-1]}}</pre>
</details>

It isn't always easy to quickly visualise the Domon-Costello nomenclature. Here is an example of how we can use GlycoDraw to visualise one of the outputs:
```python
from candycrunch.analysis import domon_costello_to_fragIUPAC
from glycowork.motif.draw import GlycoDraw
#This will calculate where on the glycan the fragments occurred and return a valid GlycoDraw input
fragment_iupac = domon_costello_to_fragIUPAC('Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc', ['Y_3_Beta', 'Y_2_Alpha'])

#Then we can simply draw the result with GlycoDraw
GlycoDraw(fragment_iupac)
```
<p align="center">
  <img width="460" height="300" src="https://raw.githubusercontent.com/BojarLab/CandyCrunch/main/images/frag_iupac_demo_white.svg">
</p>

### More tools (in `candycrunch.analysis`)
- `plot_annotated_spectrum(glycan, mzs, intensities)`: an MS2 spectrum with its CandyCrumbs annotations labeled and drawn as SNFG cartoons (also straight from the table and spectra of `wrap_inference(..., spectra = True)`)
- `supporting_ions(glycan, peaks, candidates = other_predictions)`: the intense peaks that tie each residue of a predicted structure to its position, and those that tell it apart from competing candidates
- `rank_glycopeptide_structures(peptide, modification_str, peaks)`, `rank_glycopeptide_compositions`, `rank_glycopeptide_sites`: the glycan structures, compositions, and sites of a glycoproteomics identification (e.g., `'N5(HexNAc(4)Hex(5)Fuc(1))'` from a search engine) ranked by their fragment evidence
```python
from candycrunch.prediction import load_spectra_filepath
from candycrunch.analysis import rank_glycopeptide_structures
spectra = load_spectra_filepath("C:/myfiles/my_glycopeptides.raw")
ranked = rank_glycopeptide_structures('EEQYNSTYR', 'N5(HexNAc(4)Hex(5)Fuc(1))', spectra['peak_d'][0], charge = 3, fragmentation_method = spectra['activation'][0])

## Modules
#### `prediction` <br>
- Includes all functions used in `wrap_inference`. 
- Contains `process_raw_stack`, `process_tdf_stack`, `process_mzML_stack` and `process_mzXML_stack` to extract spectra from Thermo .raw, Bruker timsTOF .d, .mzML, and .mzXML files <br>
#### `analysis` <br>
- Includes all functions used in `CandyCrumbs`.
- Contains functions to analyze and compare averaged spectra
- Contains other functions to manipulate glycan string representations, e.g., `domon_costello_to_fragIUPAC` <br>
#### `model` <br>
- Includes code for model definition, dataset handling, and data augmentation; only used in the back-end <br>
#### `examples` <br>
- Includes the extracted spectra of an example mzML file from Kouka et al. 2022

## Citation 
If you use `CandyCrunch` or any of our datasets in your work, please cite Urban et al., Nat Methods 2024.  
The data used to train `CandyCrunch` can be found at Zenodo, under [doi:10.5281/zenodo.7940047](https://zenodo.org/record/7940047)

## Code signing policy
Free code signing provided by [SignPath.io](https://about.signpath.io/), certificate by [SignPath Foundation](https://signpath.org/).

The Windows installer and the `CandyCrunch.exe` it installs are signed. Only files built from this repository by the [Desktop app workflow](https://github.com/BojarLab/CandyCrunch/blob/main/.github/workflows/app.yml) on GitHub Actions are signed, and every release is approved by hand before it is signed.

- Committers and reviewers: Daniel Bojar, James Urban, Vahid Atabaigi Elmi
- Approvers: Daniel Bojar

Privacy: the desktop app sends crash and bug reports only when you choose to send them, sends anonymous usage statistics unless you switch them off (Help > Send anonymous usage statistics), and asks GitHub whether a newer version exists. [packaging/privacy.txt](packaging/privacy.txt), which the installer shows before installing, lists exactly what each contains.
