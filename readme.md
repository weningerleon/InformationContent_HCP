# Structural Constraints on Information Content in Human Brain States
## Code accompanying the paper "The information content of brain states is explained by structural constraints on state energetic"

### Necessary files
- Diffusion connectivity matrices
- Preprocessed and parcellated fMRI timeseries

### Software requirements
- python 3.7 (not tested with other versions)
- packages: numpy, nibabel, matplotlib, seaborn, scipy, nilearn, statsmodels, sklearn, pandas, neurosynth
- MATLAB R2020b (not tested with other versions)
- MATLAB Engine API for Python

### Settings
In the file settings.py, the paths to find the necessary data and where to write the results need to be specified.
All other settings (e.g. time horizon, normalization factor used in the network control theory framework) can also be changed here.

### With the following files, information content, minimum control energy, and average controllability is calculated for all subjects/tasks/states 
- run_ic.py (for Information content)
- run_ce.py (for Minimum control energy)
- run_ctrbty.py (for Average controllability)

### After calculation the necessary results, the Figures in the manuscript can be reproduced using the following files
- fig1.py
- analyzeSTATEdist.py (Fig. 2)
- visualize_I_parcel.py (Fig. 2)
- analysis_energetics.py (Fig. 4)

### Results without figures were obtained with the following scripts
- ctrbty_pred_err.py (Result section D)
- relation_MCE_IC_SED.py (Appendix)
