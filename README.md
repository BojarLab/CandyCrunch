<p align="center">
  <img src="/images/candycrunch_logo_light_banner_pt.svg" style="height:100%;width:100%;">
</p>


# CandyCrunch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7940046.svg)](https://doi.org/10.5281/zenodo.7940046)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/bojarlab/candycrunch/blob/main/LICENSE)

## What is CandyCrunch?
**CandyCrunch** is a package for predicting glycan structure from LC-MS/MS data. It contains the CandyCrunch model, along with the rest of the inference pipeline and and downstream spectrum processing tools. These are further described in our manuscript [Urban et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.06.13.544793v1.full) &ndash; ***Predicting glycan structure from tandem mass spectrometry via deep learning*** on bioRxiv.

## Install CandyCrunch
#### Development version:
```bash
pip install git+https://github.com/BojarLab/CandyCrunch.git
``` 
#### Development version bundled with GlycoDraw:
> [!NOTE]  
> The Operating System specific installations for GlycoDraw are still required, read more in the [GlycoDraw installation guide](https://bojarlab.github.io/glycowork/examples.html#glycodraw-code-snippets)
```bash
pip install 'CandyCrunch[draw] @ git+https://github.com/Bojarlab/CandyCrunch'
``` 
#### PyPI:
```bash
pip install CandyCrunch
```
## `CandyCrunch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/CandyCrunch/blob/main/CandyCrunch.ipynb)
If you are looking for a **convenient** and **easy-to-run** version of the code that does not require any local installations, we have also created a Google Colaboratory notebook.  
The notebook contains an example pipeline ready to run, which can be copied, executed, and customised in any way.  
The example file included in the notebook is the same as in `examples/` and is ready for use in the notebook workflow. 

## Using CandyCrunch &ndash; Command line interface:
If you would like to run our main inference function from the command line, you can do so using the `candycrunch_predict` command included in this repository.

#### Requires at a minimum:
<pre>
--spectra_filepath,type=string: a filepath to an mzML/mzXML file or a .xlsx file <br />
--glycan_class, type=string: the glycan class measured ("N", "O", "lipid"/"free") <br />
--output, type=string: an output filepath ending with `.csv` or `.xlsx`
</pre>

<details>
<summary>

#### Optional arguments:
</summary>
<pre>
--mode, type=string: mass spectrometry mode; options are 'negative' or 'positive'; default: 'negative' <br />
--modification, type=string: chemical derivatization of glycans; options are “reduced”, “permethylated”, “2AA”, “2AB” or “custom”; default:”reduced”
| 
|--mass_tag, type=float: only if modification = "custom", mass of custom reducing end tag ; default:None <br />
--lc, type=string: type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC' <br />
--trap, type=string: type of mass detector used; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear' <br />
--rt_min, type=float: whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0 <br />
--rt_max, type=float: whether only spectra up to a maximum retention time (in minutes) should be considered; default:0 <br />
--rt_diff, type=float: maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0 <br />
--spectra, type=float: whether to also output the actual spectra used for prediction; default:False <br />
--get_missing, type=bool: whether to also organize spectra without a matching prediction but a valid composition; default:False
|
|--filter_out, type=set: only if get_missing = "True", set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:{'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'} <br />  
--mass_tolerance, type=float: permitted variation in Da, to still consider two masses to stem from the same molecule.; default:0.5 <br />
--supplement, type=bool: whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True <br />
--experimental, type=bool: whether to impute missing predictions via database searches etc.; default:True
|
|--taxonomy_class, type=string: only if experimental = "True", which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia' <br />
--plot_glycans, type=bool: whether you want to save an output.xlsx file that contains SNFG images of all top1 predictions, will be saved in the same folder as spectra_filepath; default:False
</pre>
</details>

#### Basic usage
> [!IMPORTANT]  
> Users must install CandyCrunch using pip before running the commands below
```console
/Users/xurbja $ candycrunch_predict --spectra_filepath path_to_my_files/file.mzML --glycan_class 'O' --output path_to_my_outputs/output_file.csv 
```

## Using CandyCrunch &ndash; LC-MS/MS glycan annotation
### `wrap_inference` (in `CandyCrunch.prediction`) <br>
Wrapper function to predict glycan structures from raw LC-MS/MS spectra using `CandyCrunch`  
  
#### Requires at a minimum:  
<pre>
- spectra_filepath, type = string: a filepath to an mzML/mzXML file or a .xlsx file <br />
- glycan_class,type = string: the glycan class measured ("N", "O", "lipid"/"free")
</pre>
mzML/mzXML files are internally processed into extracted spectra. xlsx files need to be already extracted in the format as the example file in `examples/`.
<details>
<summary>

#### Optional arguments:
</summary>
<pre>
model, type=Pytorch object: loaded from a checkpoint of a trained CandyCrunch model  <br />
glycans, type=list: ordered list of glycans used to train CandyCrunch which can be predicted by the model <br />
libr, type=list: library of monosaccharides, used as a mapping index to ensure unique graph construction <br />
bin_num, type=list: number of bins to separate the ms2 spectrum into <br />
frag_num, type=list: number of top fragments to show in df_out per spectrum; default:100 <br />
mode, type=string: mass spectrometry mode; options are 'negative' or 'positive'; default: 'negative' <br />
modification, type=string: chemical derivatization of glycans; options are “reduced”, “permethylated”, “2AA”, “2AB” or “custom”; default:”reduced”
| 
|--mass_tag, type=float: only if modification = "custom", mass of custom reducing end tag ; default:None <br />
lc, type=string: type of liquid chromatography; options are 'PGC', 'C18', and 'other'; default:'PGC' <br />
trap, type=string: type of mass detector used; options are 'linear', 'orbitrap', 'amazon', and 'other'; default:'linear' <br />
rt_min, type=float: whether only spectra from a minimum retention time (in minutes) onward should be considered; default:0 <br />
rt_max, type=float: whether only spectra up to a maximum retention time (in minutes) should be considered; default:0 <br />
rt_diff, type=float: maximum retention time difference (in minutes) to peak apex that can be grouped with that peak; default:1.0 <br />
pred_thresh, type=float: prediction confidence threshold used for filtering; default:0.01 <br />
temperature, type=float: the temperature factor used to calibrate logits; default:1.15 <br />
spectra, type=float: whether to also output the actual spectra used for prediction; default:False <br />
get_missing, type=bool: whether to also organize spectra without a matching prediction but a valid composition; default:False
|
|--filter_out, type=set: only if get_missing = "True", set of monosaccharide or modification types that is used to filter out compositions (e.g., if you know there is no Pen); default:{'Kdn', 'P', 'HexA', 'Pen', 'HexN', 'Me', 'PCho', 'PEtN'} <br />
mass_tolerance, type=float: permitted variation in Da, to still consider two masses to stem from the same molecule.; default:0.5 <br />
extra_thresh, type=float: prediction confidence threshold at which to allow cross-class predictions (e.g., predicting N-glycans in O-glycan samples); default:0.2 <br />
supplement, type=bool: whether to impute observed biosynthetic intermediaries from biosynthetic networks; default:True <br />
experimental, type=bool: whether to impute missing predictions via database searches etc.; default:True
|
|--mass_dic, type=dict: only if experimental = "True", dictionary of form mass : list of glycans; will be generated internally 
|
|--taxonomy_class, type=string: only if experimental = "True", which taxonomy class to pull glycans for populating the mass_dic for experimental=True; default:'Mammalia' 
|
|--df_use, type=DataFrame: only if experimental = "True", sugarbase-like database of glycans with species associations etc.; default: use glycowork-stored df_glycan <br />
plot_glycans, type=bool: whether you want to save an output.xlsx file that contains SNFG images of all top1 predictions, will be saved in the same folder as spectra_filepath; default:False 

</pre>
</details>

#### Basic usage
```python
annotated_spectra_df = wrap_inference("C:/myfiles/my_spectra.mzML", glycan_class)
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

### `wrap_inference_batch` (in `CandyCrunch.prediction`) <br>
Wrapper function to predict glycan structures from multiple LC-MS/MS files using CandyCrunch. <br />
This function similarly to `wrap_inference` except a list of filenames is provided and a dictionary of output DataFrames is returned, one for each input file, keyed by their filenames.  <br />
Glycan predictions are assigned to groups based on the most common prediction in the group across files. Useful for retention time correction but cannot correct LC runs in cases where noise exceeds signal. <br />

The algorithm operates under the assumption that the same structures should elute at a given RT ± intra_cat_threshold.  
The largest group of spectra across files at each composition is selected.  If groups are assigned different structres then the largest groups of the first n isomers will be selected <br />

#### Requires at a minimum:  
<pre>
- spectra_filepath_list, type = list: list of filepaths to mzML/mzXML file and/or a .xlsx files <br />
- glycan_class, type = string: the glycan class measured ("N", "O", "lipid"/"free") <br />
- intra_cat_threshold, type = float: minutes the RT of a structure can differ from the mean of a group. <br />
- top_n_isomers, type = int: number of different isomer groups at each composition to retain. 
</pre>

#### Optional arguments:
See `wrap_inference`  <br />  
<br />  
```python
spectra_filepath_list = ["C:/myfiles/my_spectra_exp1.mzML","C:/myfiles/my_spectra_exp2.mzML",
                          "C:/myfiles/my_spectra_exp3.mzML","C:/myfiles/my_spectra_exp4.mzML"]
results_dict = wrap_inference_batch(spectra_filepath_list, 'O', 1.75, 2)
```
<details>
<summary>
  
#### This is what `results_dict` would look like
</summary>
<pre>{'my_spectra_exp1: pd.DataFrame(...),
 'my_spectra_exp2: pd.DataFrame(...),
 'my_spectra_exp3: pd.DataFrame(...),
 'my_spectra_exp4: pd.DataFrame(...)}
</details>


## Using CandyCrumbs &ndash; MS2 fragment annotation
### `CandyCrumbs` (in `CandyCrunch.analysis`) <br>
Wrapper function to annotate MS2 fragments using `CandyCrumbs`  
  
#### Requires at a minimum:
<pre>
- glycan_string, type=string: a glycan in IUPAC-condensed format <br />
- fragment_masses, type=list: all observed masses which are to be annotated with a possible fragment names <br />
- mass_threshold, type=float: the maximum tolerated mass difference betweem observed masses and possible fragments 
</pre>
```python
condensed_iupac_glycan = 'Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc'
ms2_fragment_masses = [425.07,443.07,546.19,1216.32]
annotated_fragments_dict = CandyCrumbs(condensed_iupac_glycan,fragment_masses=ms2_fragment_masses,mass_threshold=1)
```

<details>
<summary>

#### Optional arguments:
</summary>
<pre>
max_cleavages, type=int: maximum number of allowed concurrent cleavages per possible fragment; default:3 <br />
simplify, type=bool: whether to select a single fragment for each mass based on mass difference, number of cleavages, and other fragments; default:True <br />
charge, type=int: the charge state of the precursor ion (singly-charged, doubly-charged, etc.); default:-1 <br />
label_mass, type=float: mass of the glycan label or reducing end modification; default:2.0156 <br />
iupac, type=bool: whether to also return the fragment sequence in IUPAC-condensed nomenclature; default:False <br />
libr, type=list: library of monosaccharides, used as a mapping index to ensure unique graph construction 
</pre>
</details>


<details>
<summary>
  
#### This is what `annotated_fragments_dict` would look like
</summary>
<pre>{425.07: {'Theoretical fragment masses': [425.12955],
  'Domon-Costello nomenclatures': [['02A_3_Alpha', 'M_H2O']],
  'Fragment charges': [-1]},
 443.07: {'Theoretical fragment masses': [443.1401],
  'Domon-Costello nomenclatures': [['02A_3_Alpha']],
  'Fragment charges': [-1]},
 546.19: {'Theoretical fragment masses': [546.18775],
  'Domon-Costello nomenclatures': [['Y_3_Beta', 'Y_2_Alpha']],
  'Fragment charges': [-1]},
 1216.32: {'Theoretical fragment masses': [1216.43105],
  'Domon-Costello nomenclatures': [['M_C2H4O2']],
  'Fragment charges': [-1]}}</pre>
</details>

It isn't always easy to quickly visualise the Domon-Costello nomenclature. Here is an example of how we can use GlycoDraw to visualise one of the outputs:
```python
#This will calculate where on the glycan the fragments occured and return a valid GlycoDraw input
fragment_iupac = domon_costello_to_fragIUPAC('Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc',['Y_3_Beta', 'Y_2_Alpha'])

#Then we can simply draw the result with GlycoDraw
GlycoDraw(fragment_iupac)
```
<p align="center">
  <img width="460" height="300" src="/images/frag_iupac_demo_white.svg">
</p>

## Modules
#### `prediction` <br>
- Includes all functions used in `wrap_inference`. 
- Contains `process_mzML_stack` and `process_mzXML_stack` to extract spectra from .mzML and .mzXML files <br>
#### `analysis` <br>
- Includes all functions used in `CandyCrumbs`.
- Contains functions to analyze and compare averaged spectra
- Contains other functions to manipulate glycan string representations, e.g., `domon_costello_to_fragIUPAC` <br>
#### `model` <br>
- Includes code for model definition, dataset handling, and data augmentation; only used in the back-end <br>
#### `examples` <br>
- Includes the extracted spectra of an example mzML file from Kouka et al. 2022

## Citation 
If you use `CandyCrunch` or any of our datasets in your work, please cite Urban et al., bioRxiv 2023.  
The data used to train `CandyCrunch` can be found at Zenodo, under [doi:10.5281/zenodo.7940047](https://zenodo.org/record/7940047)
