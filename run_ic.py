#  Copyright (C) 2021 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Script for calculating Information content for all states/tasks/subjects
# Uses mutex-files, to make parallelization across machines possible
################################################################################

import os
import pickle
import dataloader
import numpy as np
from os.path import join as opj
from informationContent import fit_ic, predict_ic
import settings

sl = settings.get_subject_list()

for subject in sl:
    print("########## " + subject)

    # Check if already processed or currently beeing processed
    file_ic_rs1 = opj(settings.work_dir_ic, subject + "_REST1_" + settings.pe + "_ic.p")
    mutex = opj(settings.work_dir_ic, subject + "_" + settings.pe + ".mutex")
    if os.path.exists(file_ic_rs1) or os.path.exists(mutex):
        continue
    os.mknod(mutex)

    fMRIdict = dataloader.get_fMRI_dict(subject, settings.scale, settings.pe)

    state_ics = {}
    state_ic_vars = {}
    for task in settings.datasets:
        state_ics[task] = []
        state_ic_vars[task] = []

    models = fit_ic(fMRIdict["REST1"])

    for task in settings.datasets:
        print(task)

        ic = predict_ic(fMRIdict[task], models)
        state_ic = np.sum(ic, axis=0)
        region_ic = np.mean(ic, axis=1)

        filename_state = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_state_ic.p")
        filename_region = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_region_ic.p")
        filename_ic = opj(settings.work_dir_ic, subject + "_" + task + "_" + settings.pe + "_ic.p")
        pickle.dump(state_ic, open(filename_state, "wb"))
        pickle.dump(region_ic, open(filename_region, "wb"))
        pickle.dump(ic, open(filename_ic, "wb"))

    os.remove(mutex)
