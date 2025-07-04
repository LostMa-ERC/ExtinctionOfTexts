# The Extinction of Texts

Data and code for the Extinction of Texts simulations and data analysis.

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Reference

This repository contains code accompanying the following paper: 

**Jean-Baptiste Camps, Julien Randon-Furling, Ulysse Godreau, _On the transmission of texts: written cultures as complex systems_, arXiv:2505.19246 [physics.soc-ph],
[ 	
https://doi.org/10.48550/arXiv.2505.19246]( 	
https://doi.org/10.48550/arXiv.2505.19246).**

```bibtex
@misc{camps_transmission_2025,
	title = {On the transmission of texts: written cultures as complex systems},
	shorttitle = {On the transmission of texts},
	url = {http://arxiv.org/abs/2505.19246},
	doi = {10.48550/arXiv.2505.19246},
	publisher = {arXiv},
	author = {Camps, Jean-Baptiste and Randon-Furling, Julien and Godreau, Ulysse},
	year = {2025},
	note = {arXiv:2505.19246 [physics]}
}
```

## Content

This repository contains

* _simulation_analysis.ipynb_, Jupyter notebook gathering all computations and results
* _birth_death_utils.py_, python file containing utility functions needed for simulations and data analysis
* _corpus_stemmata_, directory containing the stemmata anaylized. Each subdirectory correspond to a work, and contains a ```.dot``` file describing the topology of the stemma as an acyclic directed graph, as well as a ```metadata.txt``` file containing various informations on corresponding text and witnesses.
*  _Old_French_witnesses.csv_, file containing the data used for frequency and datation analysis

## Installing

The code in this repository requires Python 3.12 or later. All dependencies can be installed by running the following command in the top directory of the repository

```bash
pip install requirements.txt
```

## Acknowledgements

<img src="https://erc.europa.eu/sites/default/files/2023-06/LOGO_ERC-FLAG_FP_NEGATIF.png"
     alt="ERC Logo"
     width="250"
     style="float: left; margin-right: 10px;" /> 
     
Funded by the European Union (ERC, LostMA, 101117408). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
