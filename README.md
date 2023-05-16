# CandyCrunch
Predicting glycan structure from LC-MS/MS data, further described in our upcoming manuscript (Urban et al., bioRxiv 2023). If you use `CandyCrunch` or any of our datasets in your project, please cite Urban et al., bioRxiv 2023. The data used to train `CandyCrunch` can be found at Zenodo, under the doi:10.5281/zenodo.7940047

## Install
`pip install git+https://github.com/BojarLab/CandyCrunch.git` <br>
`import CandyCrunch`
via pip (WIP): <br> `pip install CandyCrunch` <br> `import CandyCrunch`

## Most important
`wrap_inference` (in `CandyCrunch.prediction`)
Wrapper function to predict glycan structures from raw LC-MS/MS spectra using `CandyCrunch`; requires at minimum a filepath/dataframe and the information which glycan class was measured ("N", "O", "lipid", "free", or "other").

`CandyCrumbs` (in `CandyCrunch.analysis`)
Wrapper function to annotate MSn fragments using `CandyCrumbs`; requires at minimum a hypothesized glycan structure, a list of peak m/z values, and a mass threshold.

## Modules
`prediction`
Contains all the code functionality used in `wrap_inference`. Also contains `process_mzML_stack` to extract spectra from .mzML files
`analysis`
Contains all the code functionality used in `CandyCrumbs`. Also contains functions to analyze and compare averaged spectra
`model`
Mostly contains code for model definition, dataset handling, and data augmentation; only used in the back-end
