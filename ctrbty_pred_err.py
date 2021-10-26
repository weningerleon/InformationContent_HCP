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
# Code to reproduce parts of
# Results D: Covariates of minimum control energy and information content
################################################################################

import pickle
import numpy as np
import matplotlib.pyplot as plt

from os.path import join as opj
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats.stats import pearsonr

import settings

subjects = settings.get_subject_list()

################################################################################
# %% Load data

mces = []
state_ics = []
avg_cs = []

for subject in tqdm(subjects):
    for task in settings.datasets[2:]:

        filename_ic = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_state_ic.p")
        nst_mce = subject + "_" + task + "_" + settings.pe + "_t" + str(settings.time_horizon) + "_c" + str(
            settings.norm_fac)
        filename_mce = opj(settings.work_dir_ce, nst_mce + "_mce.p")

        state_ic = pickle.load(open(filename_ic, "rb"))
        mce = pickle.load(open(filename_mce, "rb"))

        state_ics.extend(state_ic)
        mces.extend(mce)

state_ics = np.asarray(state_ics)
mces = np.asarray(mces)

for subject in subjects:
    file_avg_c = opj(settings.work_dir_ctrbty, subject + "_avg_c.p")
    avg_c = pickle.load(open(file_avg_c, "rb"))
    avg_cs.append(avg_c)

avg_cs = np.asarray(avg_cs)

###################################################################################
# Predict Minimum Control Energy from information content with a linear model
clf = LinearRegression()
clf.fit(state_ics.reshape(-1,1), mces.reshape(-1,1))

predicted_mces = clf.predict(state_ics.reshape(-1,1))
prediction_error = predicted_mces - mces.reshape(-1,1) # prediction_error>0: prediction higher than actual value

diff_ps = np.mean(prediction_error.reshape((len(subjects), -1), order='C'), axis=1)  # Compute average prediction error per subject

plt.plot(avg_cs, diff_ps, '*')
plt.show()

print("Correlation and p-value of prediction error and control energy: ")
print(pearsonr(avg_cs, diff_ps))