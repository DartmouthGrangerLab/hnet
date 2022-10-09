# hnet

## Terminology
- "edge"           - 2 bits per pixel pair (just a pair of pixel values, e.g. 1-1 = both pixels are white)
- "component"      - a single set of edge states (can be represented by one composite Hamiltonian)
- "component bank" - a set of components, stored together, sharing a graph
- "model"          - a hierarchy of component banks

types of components:
- "connected part" - a long string of edges, like a curve
- "group"          - a set of connected part components
- "meta"           - a second tier of components
- "meta group"     - a group of meta components

## Abbreviations
- comp/cmp - component (see terminology)
- idx      - index (into another array)
- h        - hamiltonian
- trn      - training data
- tst      - testing data

## Code Organization

- ```common/``` - lab common libraries
- ```figurecode/``` - functions that render results to file
- ```train/``` - code required exclusively for training
    - ```Train()```  - performs all training by calling other functions in the ```train/``` folder
- data structures
    - ```ComponentBank``` - stores one complete component bank
    - ```Dataset``` - loads whichever dataset you request, in a storage container for easy use
    - ```EDG``` - edge state enum
    - ```GRF``` - graph type enum
    - ```Model``` - stores a set of component banks, and the connections amongst them
- ```Encode()``` - matches components against new datapoints (often calls ```Energy()```)
- ```Energy()``` - computes the energy of each datapoint using each composite Hamiltonian
- ```Main()``` - code entry point; loads the data, trains, tests, prints output / figures

## Running the Code
Dependencies: matlab (designed for matlab version r2022a), python 3

Setup:
1) Pull (or download) hnet from https://github.com/DartmouthGrangerLab/hnet
2) Navigate matlab's working directory to ./hnet/matlab
3) Add the hnet/matlab/* subfolders to your matlab path.
4) to run: in matlab, execute one of the following:
	- ```Main('metacred',  'ucicreditgerman', 'tier1.memorize-->tier1.extractcorr.icacropsome.100.50.unsupsplit-->meta.extractcorr.kmeans.10.50.unsup');```
    - ```Main('groupedimg', 'mnistpy.128', 'connectedpart.memorize-->connectedpart.extractconnec.25-->connectedpart.transl.2');```
