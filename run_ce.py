#  Copyright (C) 2021 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Script for calculating Control Energy for all states/tasks/subjects
# Uses mutex-files, to make parallelization across machines possible
################################################################################

import os
import pickle
import dataloader
import numpy as np
from tqdm import tqdm
from matlab_interface import calc_min_eng_cont
import settings as s
from os.path import join as opj

sl = s.get_subject_list()

for subject in sl:
    print("########## " + subject)

    file_ce_wm1 = opj(s.work_dir_ce, subject + "_WM_" + s.pe + "_t" + str(s.time_horizon) + "_c" + str(s.norm_fac) + "_mce.p")
    if os.path.exists(file_ce_wm1):
        print("Already processed")
        continue

    fMRIdict = dataloader.get_fMRI_dict(subject, s.scale, s.pe)
    dcm = dataloader.get_dCM(subject, s.scale, zero_diagonal=True)
    dcm_rand = dataloader.randomMatrix(dcm, iter=1000)

    for task in s.datasets:
        print(task)
        filestem = subject + "_" + task + "_" + s.pe + "_t" + str(s.time_horizon) + "_c" + str(s.norm_fac)
        file = opj(s.work_dir_ce, filestem + "_mce.p")
        file_rand = opj(s.work_dir_ce, filestem + "_mce_rand.p")
        mutex_task = opj(s.work_dir_ce, filestem + "_mce_rand.mutex")

        if os.path.exists(file):
            print("This task has already been processed")
            continue
        if os.path.exists(mutex_task):
            print("This task is currently beeing processed")
            continue
        os.mknod(mutex_task)
        print("Processing...")
        activations = fMRIdict[task]
        energies = np.zeros((activations.shape[1]))
        energies_rand = np.zeros((activations.shape[1]))
        for i in tqdm(range(activations.shape[1])):
            state = activations[:,i]
            energy = calc_min_eng_cont(dcm, state, x0=0)
            energy_rand = calc_min_eng_cont(dcm_rand, state, x0=0)
            energies[i] = energy
            energies_rand[i] = energy_rand

        pickle.dump(energies, open(file, "wb"))
        pickle.dump(energies_rand, open(file_rand, "wb"))

        os.remove(mutex_task)

