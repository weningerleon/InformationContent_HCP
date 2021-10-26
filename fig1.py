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
# Code to reproduce Fig 1 in the manuscript                                    #
################################################################################
import matplotlib.pyplot as plt
import numpy as np

import pickle
from os.path import join as opj

import dataloader
import settings
import seaborn as sns
# Plot single subject
subject = settings.get_subject_list()[0]
print("Using subject :" + subject + " as an example...")

#############################################################
fMRIdict = dataloader.get_fMRI_dict(subject, settings.scale, settings.pe)
activations_wm = fMRIdict["WM"]
activations_rest = fMRIdict["REST1"]

ic_wm = pickle.load(open(opj(settings.work_dir_ic, subject + "_WM_" + settings.pe + "_ic.p"), "rb"))
ic_rest1 = pickle.load(open(opj(settings.work_dir_ic, subject + "_REST1_" + settings.pe + "_ic.p"), "rb"))

# Extract only the first run
activations_wm = activations_wm[:,:405]
ic_wm = ic_wm[:,:405]
t = np.arange(activations_wm.shape[1]) * 0.72 # time from TR to seconds

# sort regions to find region with median IC
region_ic = np.sum(ic_rest1, axis=1)
region_sorted = np.argsort(region_ic)
i = region_sorted[np.floor_divide(region_sorted.shape[0],2)]

print("Region {} has median IC and is visualized".format(i))
fig = plt.figure()

####################################
# Raw activation
ax = plt.subplot(311)
r_act_wm = activations_wm[i, :]
ax.plot(t, r_act_wm, color="silver",linewidth=3.0)
ax.set_title("Median IC region - region {}".format(i), fontsize=18)
sns.despine()
ax.yaxis.set_ticks(np.arange(-50,51,50))
ax.set_xlabel("Seconds", loc="right", fontsize=16)
ax.set_ylabel("Activation\n intensity", fontsize=16)
ax.set_xlim([-5,295])

####################################
# Region IC
ax = plt.subplot(312)
r_ic_wm = ic_wm[i, :] # region IC WM task
ax.plot(t, r_ic_wm, color="silver", linewidth=3.0)
sns.despine()
ax.yaxis.set_ticks(np.arange(4,7,1))
ax.set_xlabel("Seconds", loc="right", fontsize=16)
ax.set_ylabel("$\mathregular{I_{parcel}}$", fontsize=16)
ax.set_xlim([-5,295])

####################################
# Brain IC
ax = plt.subplot(313)
r_ic_wm = np.sum(ic_wm, axis=0)
ax.plot(t, r_ic_wm, color="silver",linewidth=3.0)
ax.set_title("Whole brain",fontsize=18)
sns.despine()
ax.set_xlabel("Seconds", loc="right", fontsize=16)
ax.set_ylabel("$\mathregular{I_{state}}$", fontsize=16)
ax.set_xlim([-5,295])
plt.tight_layout()
plt.savefig(opj(settings.figures_dir, 'fig1b.pdf'), format='pdf')
plt.show()

####################################
fig = plt.figure()
ax = plt.subplot()
r_act_rest = activations_rest[i, :] # region activation REST
r_ic_rest = ic_rest1[i, :] # region IC WM task

# sort them by magnitude so that there's a nice IC line
idxs = np.argsort(r_act_rest)
r_act_rest = r_act_rest[idxs]
r_ic_rest = r_ic_rest[idxs]

ax.hist(r_act_rest, bins=50, density=True, color="silver")
ax.set_ylabel("Histogram/Density est. of activations", color="k", fontsize=16)
ax.set_xlabel("Activation magnitude",fontsize=16)
ax2=ax.twinx()
plt.plot(r_act_rest, r_ic_rest, color="darkblue",linewidth=3.0)
ax2.set_ylabel("Information content", color="darkblue",fontsize=16)
#ax.tick_params(axis="y", width=1, length=5, direction="in")
ax.yaxis.set_ticks(np.arange(0,0.025,0.01))
sns.despine(right=False)

plt.tight_layout()
plt.savefig(opj(settings.figures_dir, 'fig1a.pdf'), format='pdf')
plt.show()
