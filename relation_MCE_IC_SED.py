################################################################################
#  Copyright (C) 2021 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Code used for Appendix, to assess difference between
# Euclidean distance, control energy, and information content
# Figures are not used in manuscript
################################################################################

import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tqdm import tqdm
from os.path import join as opj
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_rel
from scipy.stats.stats import pearsonr

import dataloader
import settings

subjects = settings.get_subject_list()

################################################################################
#%% Load data

mces = []
mces_rand = []
state_ics = []
dists = [] # squared euclidean distance

for task in settings.datasets[2:]:
        print(task)
        for i, subject in tqdm(enumerate(subjects)):

                filename_ic = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_state_ic.p")
                nst_mce = subject + "_" + task + "_" + settings.pe + "_t" + str(settings.time_horizon) + "_c" + str(settings.norm_fac)
                filename_mce = opj(settings.work_dir_ce, nst_mce + "_mce.p")
                filename_mce_rand = opj(settings.work_dir_ce, nst_mce + "_mce_rand.p")

                activations = dataloader.get_fMRI_ts_clipped(subject, settings.scale, task, settings.pe)
                dist = np.sum(np.square(activations), axis=0)

                state_ic = pickle.load(open(filename_ic, "rb"))
                mce = pickle.load(open(filename_mce, "rb"))
                mce_rand = pickle.load(open(filename_mce_rand, "rb"))

                state_ics.extend(state_ic)
                mces.extend(mce)
                mces_rand.extend(mce_rand)
                dists.extend(dist)

state_ics = np.asarray(state_ics)
mces = np.asarray(mces)
mces_rand = np.asarray(mces_rand)
dists = np.asarray(dists)

print("SED - IC: " + str(pearsonr(dists, state_ics)))
print("SED - MCE: " + str(pearsonr(dists, mces)))
print("MCE - IC: " + str(pearsonr(mces, state_ics)))
####################################################################################
# => Further figures are not used in manuscript

mces  = mces .reshape(-1,1)
mces_rand = mces_rand.reshape(-1,1)
state_ics = state_ics.reshape(-1,1)
dists = dists.reshape(-1,1)

#%% Direct relationships between Minimum Control Energy (MCE), squared Euclidean distance (SED) and information content (IC)
clf = LinearRegression()

#%% SED -> MCE
clf.fit(dists, mces)
pred_d = clf.predict(dists)
score_ed = r2_score(mces, pred_d)

plt.plot(dists[::100], mces[::100], 'b.', dists[::100], pred_d[::100], 'r.', linewidth=1, label="REST2, T=" + str(settings.time_horizon))
plt.ylabel("Min Ctrl Energy")
plt.xlabel("SED")
plt.show()

#%% SED -> IC
clf.fit(dists, state_ics)
dists_ics = clf.predict(dists)
score_dists_ics = r2_score(dists_ics, state_ics)

#%% Plotting SED->IC
plt.plot(dists[::100], state_ics[::100], 'b.', dists[::100], dists_ics[::100], 'r.', linewidth=1, label="REST2, T=" + str(settings.time_horizon))
plt.ylabel("Information content")
plt.xlabel("SED")
plt.show()

#%% IC -> MCE
clf.fit(state_ics, mces)
pred_ics = clf.predict(state_ics)
score_ic = r2_score(mces, pred_ics)

plt.plot(state_ics[::100], mces[::100], 'b.', state_ics[::100], pred_ics[::100], 'r.', linewidth=1)
plt.ylabel("Min Ctrl Energy")
plt.xlabel("Information content")
plt.show()

#%% SED -> MCE null model
clf.fit(dists, mces_rand)
pred_rand_d = clf.predict(dists)
score_rand_ed = r2_score(mces_rand, pred_rand_d)

plt.plot(dists[::100], mces_rand[::100], 'b.', dists[::100], pred_rand_d[::100], 'r.', linewidth=1, label="REST2, T=" + str(settings.time_horizon))
plt.ylabel("Min Ctrl Energy (randomized CM)")
plt.xlabel("SED")
plt.show()

#%% IC -> MCE null model
clf.fit(state_ics, mces_rand)
pred_rand_ics = clf.predict(state_ics)
score_rand_ic = r2_score(mces_rand, pred_rand_ics)

plt.plot(state_ics[::100], mces_rand[::100], 'b.', state_ics[::100], pred_rand_ics[::100], 'r.', linewidth=1, label="REST2, T=" + str(settings.time_horizon))
plt.ylabel("Min Ctrl Energy (randomized CM)")
plt.xlabel("Information content")
plt.show()

print("###################### Direct relations, r2: #####################")
print("SED->MCE: score={}".format(score_ed))
print("IC->MCE: score={}".format(score_ic))
print("SED->MCE (null model): score={}".format(score_rand_ed))
print("IC->MCE (null model): score={}".format(score_rand_ic))
print("##################################################################")

####################################################################################
#%% Relationship MCE-IC with- and without SED

# joint prediction of MCE and IC -> SED
x = np.concatenate((dists, state_ics), axis=1)
clf.fit(x, mces)
pred_x = clf.predict(x)
score = r2_score(mces, pred_x)
print("SED&IC->MCE: score={}".format(score))

# Measure error with mean absolute error
mae_joint = mean_absolute_error(mces.T, pred_x.T, multioutput="raw_values")
mae_ed = mean_absolute_error(mces.T, pred_d.T, multioutput="raw_values")
t, p = ttest_rel(mae_joint, mae_ed)
print("MAE difference between Joint (={}) and SED (={}) prediction is significant with p={}".format(np.mean(mae_joint), np.mean(mae_ed), p))
print("MAE of only IC: ")

####################################################################################
# Different possibility to check whether IC is better in predicting MCE than SED (not in publication)
# First, SED is regressed out of MCE.
# Then, IC predicts MCE with SED regressed out
#%% Relationship MCE - IC, after regressing out SED
diff = mces-pred_d # MCE with SED regressed out
Y = diff.squeeze()

# Prediction using information content
X = sm.add_constant(state_ics.squeeze())
olsmod = sm.OLS(Y, X)
olsres = olsmod.fit()
print(olsres.summary())

ypred = olsres.predict(X)

plt.plot(state_ics[::100], diff[::100], '*', state_ics[::100], ypred[::100], "r*")
plt.ylabel("Diff Ctrl Energy")
plt.xlabel("Information content")
plt.show()
# After regressing out SED, IC is still a good predictor of MCE.
# Thus, the relationship between IC-MCE is not only due to SED