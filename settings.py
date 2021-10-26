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

import pickle
import numpy as np
from sys import platform
from os.path import join as opj
import os
import random
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager

#%% Actual Settings --> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasets = ("REST1", "REST2", "EMOTION", "GAMBLING", "LANGUAGE", "MOTOR", "RELATIONAL", "SOCIAL", "WM")
pe = "both"  # phase_encoding: LR, RL, or both
scale = "yeo_100"  # parcellation scheme, yeo_100 or yeo_400
time_horizon = 10  # time horizon used for min control energy
norm_fac = 0.001  # normalization factor in A_norm = A/(max(abs(max(eigv(A)))*(1+c))-I
#%% <-- Actual Settings-> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Plotting and output settings
np.set_printoptions(formatter={'float_kind':"{:.5f}".format})
matplotlib.rcParams.update({'font.size': 7})
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 300


#%% Local paths > %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_dir_base = '/images/Diffusion_Imaging/brainInformationContent'
work_dir_base = '/work/scratch/weninger/brainIC_results/'

image_dir_ts = opj(image_dir_base, "fmri_timeseries", scale)
image_dir_sc = opj(image_dir_base, "diffusion_connectivity_matrices", scale)
work_dir_ic = opj(work_dir_base, "informationContent", scale)
work_dir_ce = opj(work_dir_base, "controlEnergy", scale)
work_dir_ctrbty = opj(work_dir_base, "controllability", scale)
work_dir_neurosynth = opj(work_dir_base, "neurosynth")

matlab_dir = "/home/staff/weninger/tmp/hcp_fmri"

figures_dir = opj(work_dir_base, "figures")

#%% Subject list --> %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def determine_subject_list(parcellation, task, phase_encoding):
    # recursive definition, that looks through all possible data

    if type(parcellation) == list:
        sets = []
        for p in parcellation:
            sets.append(determine_subject_list(p, task, phase_encoding))
        intersection = sets[0].intersection(*sets[1:])
        print("{} subjects have all data".format(intersection.__len__()))
        return intersection
    if task == "all":
        sets = []
        for task in datasets:
            sets.append(determine_subject_list(parcellation, task, phase_encoding))
        intersection = sets[0].intersection(*sets[1:])
        print("{} subjects have all {} data".format(intersection.__len__(), parcellation))
        return intersection
    if task == "REST":
        setA = determine_subject_list(parcellation, "REST1", phase_encoding)
        setB = determine_subject_list(parcellation, "REST2", phase_encoding)
        intersection = setA.intersection(setB)
        diff = setA.symmetric_difference(setB)
        print("{} subject with either only REST1 or only REST2 were found".format(diff.__len__()))
        print("Returning {} total matching subjects".format(intersection.__len__()))
        return intersection
    if phase_encoding == "both":
        setA = determine_subject_list(parcellation, task, phase_encoding="LR")
        setB = determine_subject_list(parcellation, task, phase_encoding="RL")
        intersection = setA.intersection(setB)
        diff = setA.symmetric_difference(setB)
        print("{} subject with only one phase encoding were found".format(diff.__len__()))
        print("Returning {} total matching subjects".format(intersection.__len__()))
        return intersection

    assert parcellation in ("yeo_100", "yeo_400"), "wrong parcellation"
    assert task in datasets, "wrong task"
    assert phase_encoding in ("LR", "RL"), "wrong phase encoding"

    #%% fMRI
    subjects_fMRI = []
    files = os.listdir(opj(image_dir_base, "fmri_timeseries", parcellation))
    for f in files:
        if not f.endswith("ts.npy"):
            continue
        [p1,p2,s,t,e, _] = f.split(".")[0].split("_")
        p = p1 + "_" + p2
        if (p == parcellation) and (t==task) and (e==phase_encoding):
            subjects_fMRI.append((s))

    #%% dCM
    subjects_dCM = []
    files = os.listdir(opj(image_dir_base, "diffusion_connectivity_matrices", parcellation))
    for f in files:
        if f.endswith("Lausanne_matrix.npy") or f.endswith("Desikan_matrix.npy"):
            continue
        [s,p1,p2,_] = f.split("_")
        p = p1 + "_" + p2
        if (p == parcellation):
            subjects_dCM.append(s)

    setA = set(subjects_dCM)
    setB = set(subjects_fMRI)
    intersection = setA.intersection(setB)
    diff = setA.symmetric_difference(setB)
    print(task + ": {} subject with either only connectivity matrix or only fMRI were found".format(diff.__len__()))
    print(task + ": Returning {} total matching subjects".format(intersection.__len__()))
    return intersection


#%% Creating or loading a list of complete subjects
def get_subject_list():
    subject_list_file = opj(work_dir_base,"subject_list.p")
    if os.path.isfile(subject_list_file):
        sl = pickle.load(open(subject_list_file, "rb"))
    else:
        sl = determine_subject_list(["yeo_100", "yeo_400"], "all", "both")
        sl = list(sl)
        pickle.dump(sl, open(subject_list_file, "wb"))
    random.shuffle(sl)
    return sl
