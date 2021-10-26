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
################################################################################

import settings
from os.path import join as opj
import numpy as np
import pandas as pd
import nibabel as nib



#%% Diffusion Connectivity Matrices
def get_dCM(subject, parcellation, zero_diagonal=True, symmetric=True):
    dcm_path = opj(settings.image_dir_sc, subject + "_" + parcellation + "_matrix.npy")
    dcm = np.load(dcm_path)
    if symmetric and zero_diagonal:
        dcm = np.triu(dcm, k=1)
        dcm = dcm + dcm.transpose()
    elif zero_diagonal:
        # set diagonal elements to zero
        s = dcm.shape
        dcm = dcm.flatten()
        dcm[0::s[0]+1] = 0
        dcm = dcm.reshape(s)
    else:
        raise Exception("Not implemented!!!")

    dcm = dcm / np.sum(np.abs(dcm))
    return dcm

# return complete fMRI timeseries data
# LR/RL gets concatenated for phase_encoding="both"
def get_fMRI_ts_unclipped(subject, parcellation, task, phase_encoding="LR"):

    if task == "REST":
        ts1 = get_fMRI_ts_unclipped(subject, parcellation, "REST1", phase_encoding)
        ts2 = get_fMRI_ts_unclipped(subject, parcellation, "REST2", phase_encoding)
        return np.concatenate((ts1, ts2), axis=1)
    if phase_encoding == "both":
        ts1 = get_fMRI_ts_unclipped(subject, parcellation, task, phase_encoding="LR")
        ts2 = get_fMRI_ts_unclipped(subject, parcellation, task, phase_encoding="RL")
        return np.concatenate((ts1, ts2), axis=1)

    assert parcellation in ("yeo_100", "yeo_400"), "wrong parcellation"
    assert task in settings.datasets, "wrong task"
    assert phase_encoding in ("LR", "RL"), "wrong phase encoding"

    file = parcellation + "_" + subject + "_" + task + "_" + phase_encoding + "_ts.npy"
    ts_path = opj(settings.image_dir_ts, file)
    ts = np.load(ts_path)

    # demean ts, sometimes there was a slight offset
    ts = ts - np.mean(ts, axis=1, keepdims=True)
    return ts

# return complete fMRI timeseries data, clipped according to REST1 distribution
def get_fMRI_dict(subject, parcellation, phase_encoding, clipped=True):

    data = {}
    for t in settings.datasets:
        ts = get_fMRI_ts_unclipped(subject, parcellation, t, phase_encoding=phase_encoding)
        data[t] = ts

    if clipped:
        for i in range(data["REST1"].shape[0]):
            thr_p = min(data["REST1"][i, :].max(), data["REST1"][i, :].std() * 4)
            thr_n = max(data["REST1"][i, :].min(), -data["REST1"][i, :].std() * 4)
            for t in settings.datasets:
                data[t][i, :] = np.clip(data[t][i, :], thr_n, thr_p)
    return data

# Get all fMRI timeseries, stored in a dictionary
def get_fMRI_ts_clipped(subject, parcellation, task, phase_encoding="LR"):

    ts = get_fMRI_ts_unclipped(subject, parcellation, task, phase_encoding=phase_encoding)
    rest1_data = get_fMRI_ts_unclipped(subject, parcellation, task, phase_encoding=phase_encoding)

    for i in range(rest1_data.shape[0]):
        thr_p = min(rest1_data[i, :].max(), rest1_data[i, :].std() * 4)
        thr_n = max(rest1_data[i, :].min(), -rest1_data[i, :].std() * 4)
        for t in settings.datasets:
            ts[i, :] = np.clip(ts[i, :], thr_n, thr_p)
    return ts

def get_atlas(parcellation):

    dir = opj(settings.image_dir_base, "Schaefer2018_LocalGlobal", "Parcellations", "MNI")
    if parcellation == "yeo_100":
        path = opj(dir, "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.nii.gz")
    elif parcellation == "yeo_400":
        path = opj(dir, "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.nii.gz")
    else:
        raise Exception("sorry, not implemented")

    atlas = nib.load(path)
    return atlas.get_fdata(), atlas.affine

def get_mni152():
    path = opj(settings.image_dir_base, "mni_icbm152_nlin_sym_09a", "mni_icbm152_t1_tal_nlin_sym_09a.nii")

    atlas = nib.load(path)
    return atlas.get_fdata()

# returns a random matrix with the same degree, weight, and strength distribution
def randomMatrix(w, iter=100):
    n = w.shape[0]
    w0 = w.copy()
    for i in range(iter):
        p = np.random.permutation(n)
        if np.random.uniform()<0.5:
            w0 = w0[:,p]
            w0 = w0[p,:]
        else:
            w0 = w0[p,:]
            w0 = w0[:,p]
    return w0

# Splits the fixation / initation blocks from the t-fMRI time series
def remove_fixblocks(data, task_name, tr=0.72):
    switcher = {
        "WM": [(0, 8), (64, 79), (135, 150), (206, 221), (277, 291)],
        "GAMBLING": [(0, 8), (36, 51), (79, 94), (122, 137), (165, 180)],
        "MOTOR": [(0, 8)],
        "LANGUAGE": [(0, 8)],
        "SOCIAL": [(0, 8), (31, 46), (69, 84), (107, 122), (145, 160), (183, 198)],
        "RELATIONAL": [(0, 8), (44, 60), (98, 114), (150, 166)],
        "EMOTION": [(0, 8)],
    }
    fix_blocks = switcher[task_name]

    istask = np.ones(data.shape[-1]).astype(np.bool)
    for block in fix_blocks:
        start = np.int(np.rint(block[0] / tr))
        end = np.int(np.rint(block[1] / tr))
        istask[start:end] = False
    if settings.pe == "both":
        for block in fix_blocks:
            start = np.int(np.rint(block[0] / tr) + data.shape[-1] / 2)
            end = np.int(np.rint(block[1] / tr) + data.shape[-1] / 2)
            istask[start:end] = False

    if len(data.shape)==1:
        data = np.expand_dims(data, axis=0)
    onlytask = data[:,istask]
    onlyfix = data[:,~istask]
    return onlytask, onlyfix