# Structural Constraints on Information Content in Human Brain States
## Code accompanying the publication in xxxxx

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
- neurosynth_prep.py (for Neurosynth comparison)

### After calculation the necessary results, the Figures in the manuscript can be reproduced using the following files
- fig1.py
- analyzeSTATEdist.py (Fig. 2)
- GLM_compute.py, neurosynth_vis.py, analyzeREGIONS.py (Fig. 3)
- analysis_energetics.py (Fig. 4)

### Results without figures can be seen with the following scripts
- ctrbty_pred_err.py (Result section C)
- relation_MCE_IC_SED.py (Appendix)
