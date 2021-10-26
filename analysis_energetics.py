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
# Code to reproduce Fig 4 in the manuscript
################################################################################

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import seaborn as sns
import scipy.stats as stats

from os.path import join as opj
from numpy.random import RandomState

import settings


tasknames = ['Emotion', 'Gambling', 'Language', 'Motor', 'Relational', 'Social', 'Working\nmemory']

subjects = settings.get_subject_list()
subjects.sort()
fig = plt.figure()

################################################################################
#%% Upper part of Figure
ax = fig.add_subplot(121)

# load data
for num_task, task in enumerate(settings.datasets[2:]):
    print(task)
    mces = []
    state_ics = []
    for i, subject in enumerate(subjects):
        filename_ic = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_state_ic.p")
        nst_mce = subject + "_" + task + "_" + settings.pe + "_t" + str(settings.time_horizon) + "_c" + str(
            settings.norm_fac)
        filename_mce = opj(settings.work_dir_ce, nst_mce + "_mce.p")

        state_ic = pickle.load(open(filename_ic, "rb"))
        mce = pickle.load(open(filename_mce, "rb"))

        state_ics.append(np.mean(state_ic))
        mces.append(np.mean(mce))

    idxs = RandomState(2).randint(0, len(mces), 50)
    state_ics = np.asarray(state_ics)
    mces = np.asarray(mces)
    sns.scatterplot(x=state_ics[idxs], y=mces[idxs], linewidth=0, alpha=0.5, label=tasknames[num_task])
# plotting
if settings.scale == "yeo_100":
    ax.set_xlim([405,525])
    ax.set_ylim([1,278])
ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.set_axisbelow(True)
ax.tick_params(axis="y", width=0, length=0, direction="in", pad=-20)
ax.tick_params(axis="x", direction="in")
sns.despine(left=True, trim=True, ax=ax)
ax.set_ylabel("Mean Subject Min Ctrl Energy")
ax.set_xlabel("Mean subject state information content")
ax.set_title("(a) Control energy versus information content", fontweight="bold")
offset = matplotlib.transforms.ScaledTranslation(0, 1/12, fig.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

ax.legend(loc="upper left", bbox_to_anchor=(0.1,1.0),framealpha=1)
################################################################################
#%% Right part of Figure
# load data
mces = []
state_ics = []
mce_rands = []
gains_low = []
gains_mid = []
gains_high = []

all_gains = []
for i, subject in enumerate(subjects):
    gains = []
    state_ics = []
    for task in settings.datasets[2:]:

        filename_ic = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_state_ic.p")
        filename_mce = opj(settings.work_dir_ce, subject + "_" + task + "_" + settings.pe +
                           "_t" + str(settings.time_horizon) + "_c" + str(settings.norm_fac) + "_mce.p")
        filename_mce_rand = opj(settings.work_dir_ce, subject + "_" + task + "_" + settings.pe +
                                "_t" + str(settings.time_horizon) + "_c" + str(settings.norm_fac) + "_mce_rand.p")

        state_ic = pickle.load(open(filename_ic, "rb"))
        mce = pickle.load(open(filename_mce, "rb"))
        mce_rand = pickle.load(open(filename_mce_rand, "rb"))

        gain = (np.asarray(mce_rand) - np.asarray(mce))/np.asarray(mce_rand)*100 # gain in percentage
        gains.extend(gain)
        state_ics.extend(state_ic)

    # for each subject, divide states into low, mid, high IC states
    all_gains.extend(gains)
    gains = np.asarray(gains)
    state_ics = np.asarray(state_ics)

    args = np.argsort(state_ics)
    state_ics = state_ics[args]
    gains = gains[args]

    third = int(len(gains)/3)
    gains_low.append(np.mean(gains[:third]))  # collaps across states and tasks
    gains_mid.append(np.mean(gains[third:third*2]))
    gains_high.append(np.mean(gains[third*2:]))

# plotting
ax = fig.add_subplot(122)
sns.boxplot(data=(gains_low, gains_mid, gains_high), width=0.7, showfliers=False, linewidth=1, palette="Blues")
sns.despine(left=True, bottom=True, trim=True, ax=ax)
ax.set_xticklabels(['Low $\mathregular{I_{state}}$', 'Mid $\mathregular{I_{state}}$', 'High $\mathregular{I_{state}}$'])
ax.set_ylabel("Rel. improvement over null model")
ax.set_title("(b) Efficiency of actual brain wiring", fontweight="bold")
ax.tick_params(axis="x", width=1, length=0, direction="in", pad=0)
ax.tick_params(axis="y", width=0, length=0, direction="in", pad=-8)
ax.yaxis.set_ticks([10, 15, 20])
ax.yaxis.set_ticklabels(["10%","15%","20%"])
offset = matplotlib.transforms.ScaledTranslation(0, 1/12, fig.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

ax.yaxis.grid(True)  # Hide the horizontal gridlines
ax.set_axisbelow(True)

fig.set_size_inches(6.85, 2.5)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
plt.savefig(opj(settings.figures_dir, 'fig4_' + settings.scale + '.pdf'))
plt.show()

f_statistic, pval = stats.f_oneway(gains_low, gains_mid, gains_high)
print("ANOVA: the difference between groups has pval={}".format(pval))
