<p align="center">
  <img src="/images/candycrunch_logo.jpg" style="height:50%;width:50%;">
</p>

-----------------

# CandyCrunch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7940047.svg)](https://doi.org/10.5281/zenodo.7940047)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/bojarlab/candycrunch/blob/main/LICENSE)

## What is CandyCrunch?
**CandyCrunch** is a package for predicting glycan structure from LC-MS/MS data. It contains the CandyCrunch model, along with the rest of the inference pipeline and and downstream spectrum processing tools. These are further described in our upcoming manuscript (Urban et al., bioRxiv 2023).

## Install CandyCrunch
#### Development version:
```bash
pip install git+https://github.com/BojarLab/CandyCrunch.git
``` 
#### Development version bundled with GlycoDraw:
> **Note**
> The Operating System specific installations for GlycoDraw are still required, read more in the [GlycoDraw installation guide](https://bojarlab.github.io/glycowork/examples.html#glycodraw-code-snippets)
```bash
pip install 'CandyCrunch[draw] @ git+https://github.com/Bojarlab/CandyCrunch
``` 
#### PyPI:
> **Warning**
> This version is not yet published
```bash
pip install CandyCrunch
```
## `CandyCrunch.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BojarLab/CandyCrunch/blob/main/CandyCrunch.ipynb)
If you are looking for a **convenient** and **easy-to-run** version of the code that does not require any local installations, we have also created a Google Colaboratory notebook.  
The notebook contains and example pipeline ready to run, which can be copied, executed, and customised in any way. 

## Using CandyCrunch &ndash; LC-MS/MS glycan annotation
#### `wrap_inference` (in `CandyCrunch.prediction`) <br>
Wrapper function to predict glycan structures from raw LC-MS/MS spectra using `CandyCrunch`  
  
Requires at a minimum:
- a filepath to an mzML/mzXML file or a .xlsx file containting their extracted spectra
- the glycan class measured ("N", "O", "lipid", "free", or "other")
```python
annotated_spectra_df = wrap_inference(C:/myfiles/my_spectra.mzML, glycan_class)
```

## Using CandyCrumbs &ndash; MS2 fragment annotation
#### `CandyCrumbs` (in `CandyCrunch.analysis`) <br>
Wrapper function to annotate MS2 fragments using `CandyCrumbs`  
  
Requires at a minimum:
- a hypothesized glycan structure
- a list of peak m/z values
- a mass threshold
```python
condensed_iupac_glycan = 'Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc'
ms2_fragment_masses = [425.07,443.07,546.19,1216.32]
annotated_fragments_dict = CandyCrumbs(condensed_iupac_glycan,fragment_masses=ms2_fragment_masses,mass_threshold=1)
```
<details>
<summary>This is what `annotated_fragments_dict` would look like</summary>
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
#This will calculate the where on the glycans the fragments occured and return a valid GlycoDraw input
fragment_iupac = domon_costello_to_fragIUPAC('Gal(a1-3)Gal(b1-4)GlcNAc(b1-6)[GalNAc(b1-4)GlcNAc(b1-3)]Gal(b1-4)Glc',['Y_3_Beta', 'Y_2_Alpha'])

#Then we can simply draw the result with GlycoDraw
GlycoDraw(fragment_iupac)
```
<p align="center">
  <img width="460" height="300" src="/images/frag_iupac_demo.svg">
</p>

## Modules
#### `prediction` <br>
- Includes all functions used in `wrap_inference`. 
- Contains `process_mzML_stack` to extract spectra from .mzML files <br>
#### `analysis` <br>
- Includes all functions used in `CandyCrumbs`.
- Contains functions to analyze and compare averaged spectra
- Contains other functions to manipulate glycan string represntations e.g. `domon_costello_to_fragIUPAC` <br>
#### `model` <br>
- Includes code for model definition, dataset handling, and data augmentation; only used in the back-end <br>

## Citation 
If you use `CandyCrunch` or any of our datasets in your work, please cite Urban et al., bioRxiv 2023.  
The data used to train `CandyCrunch` can be found at Zenodo, under [doi:10.5281/zenodo.7940047](https://zenodo.org/record/7940047)
