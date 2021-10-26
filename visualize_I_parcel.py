################################################################################
#  Copyright (C) 2020 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Code for Fig 3 in the manuscript
################################################################################

import pickle
import dataloader
import numpy as np
import nibabel as nib
from statsmodels.stats.multitest import multipletests
from os.path import join as opj
from scipy.stats import ttest_rel

# nilearn plotting confuses matplotlib, making this charade necessary
import matplotlib
import matplotlib.pyplot as plt
bknd = matplotlib.get_backend()
from nilearn import plotting
matplotlib.use(bknd)

import settings
from scipy.stats import zscore



################################################################################
# Visualize regional distribution with glass brain
def vis_regions(regional_data, atlas_name, title=None):
    region_means = np.mean(regional_data, axis=0)

    atlas, aff = dataloader.get_atlas(atlas_name)
    brain = np.zeros(atlas.shape)
    for i, val in enumerate(region_means):
        brain[atlas==(i+1)] = val
    brain[brain>0] = zscore(brain[brain>0])
    img = nib.Nifti1Image(brain, aff)
    nib.save(img, "tmp.nii.gz")
    plotting.plot_glass_brain(img, colorbar=True, plot_abs=False, title=title, cmap="bwr", symmetric_cbar=False)
    plt.savefig(opj("/work/scratch/weninger/share", title + "_ic_from_onlytask.pdf"))
    plt.show()

# Visualize difference in regional distribution with glass brain
def vis_regions_diff(region_vals_task, region_vals_rest, atlas_name, task, vmax=None):
    # Under these filenames, the nifti files can also be retrieved for plotting with other tools
    filename_nifti = opj(settings.work_dir_base, "visualize_region_IC", settings.scale + "_" + task + ".nii.gz")
    filename_txt = opj(settings.work_dir_base, "visualize_region_IC", settings.scale + "_" + task + ".txt")

    atlas, aff = dataloader.get_atlas(atlas_name)
    brain = np.zeros(atlas.shape)

    # Zero-out all non-significant regions
    t_all, p_all = ttest_rel(region_vals_task, region_vals_rest, axis=0)
    reject, p_adjusted, _, _ = multipletests(p_all, alpha=0.01, method='bonferroni')

    region_vals = np.mean(region_vals_task,axis=0) - np.mean(region_vals_rest, axis=0)
    for i, val in enumerate(region_vals):
        if reject[i]:
            brain[atlas==(i+1)] = val
        else:
            region_vals[i]=0

    # Saving and plotting
    img = nib.Nifti1Image(brain, aff)
    nib.save(img, filename_nifti)
    np.savetxt(filename_txt, region_vals)

    plotting.plot_glass_brain(img, colorbar=True, plot_abs=False, vmax=vmax, title=task)
    plt.savefig(opj("/work/scratch/weninger/share", task + "_ic_taskspecific.pdf"))
    plt.show()


def get_mean_region_ic(task):
    sl = settings.get_subject_list()

    m = np.zeros((len(sl), int(settings.scale[-3:])))

    for i, subject in enumerate(sl):

        filename = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_ic.p")
        ic = pickle.load(open(filename, "rb"))

        if not task.startswith("REST"):
            onlytask, onlyfix = dataloader.remove_fixblocks(ic, task)
            region_ic = np.mean(onlytask,axis=1)
        else:
            region_ic = np.mean(ic, axis=1)

        m[i,:] = region_ic

    return m


################################################################################
def main():
    tasks = settings.datasets[2:]

    regional_data_rest1 = get_mean_region_ic("REST2")
    for task in tasks:
        regional_data = get_mean_region_ic(task)
        vis_regions_diff(regional_data, regional_data_rest1, atlas_name=settings.scale, task=task, vmax=1)


if __name__ == "__main__":
    main()
