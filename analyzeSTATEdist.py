#  Copyright (C) 2021 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Code to reproduce Fig 2 in the manuscript
################################################################################

import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join as opj
from scipy import stats
from statsmodels.stats.multitest import multipletests

import settings
import dataloader

sl = settings.get_subject_list()
tasks = settings.datasets[2:]
tasknames = ['Emotion', 'Gambling', 'Language', 'Motor', 'Relational', 'Social', 'Working\nmemory']

fig = plt.figure(figsize=(3.31, 9))
################################################################################
#%% Upper part of the figure
ax = fig.add_subplot(211)

########################################
# Load data
med_ics = {}
for t in tasks:
    med_ics[t] = []

for subject in sl:
    for t in tasks:
        filename = opj(settings.work_dir_ic, subject + "_" + t + "_" + settings.pe + "_state_ic.p")
        state_ic = pickle.load(open(filename, "rb"))
        if not t.startswith("REST"): # remove fixation blocks, just look at actual task-data
            onlytask, fixblock = dataloader.remove_fixblocks(state_ic, t)
            med_ics[t].append(np.mean(onlytask))
        else:
            med_ics[t].append(np.mean(state_ic))

lbls, data = [*zip(*med_ics.items())]  # 'transpose' items to parallel key, value lists, lbls==tasks
data = np.asarray(data)

########################################
#%% Plotting
sns.boxplot(data=data.T, showfliers=False, linewidth=1)

ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.set_axisbelow(True)
sns.despine(left=True, bottom=True, trim=True, ax=ax)
ax.tick_params(axis="y", width=0, length=0, direction="in")
ax.tick_params(axis="x", direction="in")
ax.set_ylabel("Whole brain information content")
ax.set_title("(a) Mean information content across tasks", fontweight="bold", size=8, loc="left")

ax.set_xticklabels(tasknames)
if settings.scale == "yeo_100":
    ax.yaxis.set_ticks(np.arange(400,520,20))

plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", rotation_mode="anchor")

########################################
# Test significance of distribution against rsfMRI
print("Significantly different than social, bonferroni-corrected:")

pvals = []
for i, t in enumerate(tasks):
    _, p = stats.ttest_ind(data[5,:], data[i,:])
    pvals.append(p)

p_adjusted = multipletests(pvals, alpha=0.05, method='bonferroni')
for i, t in enumerate(tasks):
    print(t + ": p=" + str(p_adjusted[1][i]))
print("################################################################################")
################################################################################
#%% Lower part of the figure
ax = fig.add_subplot(212)

# Loading
pvals_skew = []
skews = []
pvals_norm = []
for t_i, task in enumerate(tasks):
    m = np.zeros((len(sl), int(settings.scale[-3:])))

    for i, subject in enumerate(sl):

        filename = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_region_ic.p")
        region_ic = pickle.load(open(filename, "rb"))

        m[i,:] = region_ic

    region_means = np.mean(m, axis=0)
    regions = np.argsort(region_means)[::-1] # sort regions by information content

    skews.append(stats.skew(region_means))
    pvals_skew.append(stats.skewtest(region_means)[1])
    pvals_norm.append(stats.shapiro(region_means)[1])
    print("{}: p(normality) = {:.4f}, p(skew) = {:.4f}".format(task, stats.shapiro(region_means)[1],
                                                                                 stats.skewtest(region_means)[1]))

    ax.plot(region_means[regions], label=tasknames[t_i], linewidth=1)

# Plotting
ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.tick_params(axis="y", width=0, length=0, direction="in")
ax.tick_params(axis="x", direction="in")
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
if settings.scale == "yeo_100":
    ax.yaxis.set_ticks(np.arange(3.8, 5.5, 0.4))
ax.set_axisbelow(True)
ax.legend(framealpha=1, prop={'size': 6})
sns.despine(left=True, bottom=True, trim=True, ax=ax)
ax.set_ylabel("Mean region information content")
ax.set_xlabel("Region (sorted by information content)")
ax.set_title("(b) Regional skew of information content", fontweight="bold", loc="left")

fig.set_size_inches(3, 5)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.35)
plt.savefig(opj(settings.figures_dir, "fig2_" + settings.scale + ".pdf"))
plt.show()