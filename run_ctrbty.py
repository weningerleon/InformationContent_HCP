#  Copyright (C) 2021 by RWTH Aachen University                                #
#  License:                                                                    #
#                                                                              #
#   This software is dual-licensed under:                                      #
#   • Commercial license (please contact: lfb@lfb.rwth-aachen.de)              #
#   • AGPL (GNU Affero General Public License) open source license             #
################################################################################
#   Author: Leon Weninger                                                      #
################################################################################
# Script for calculating average controllability for all subjects
# Quickly processed for all subjects
################################################################################

import os
import pickle
import dataloader
from os.path import join as opj

import settings


from matlab_interface import avg_nrj_and_ctrbty

#%% Load data
subjects = settings.get_subject_list()
subject = subjects[-1]
dcm = dataloader.get_dCM(subject, settings.scale, zero_diagonal=True)

ics = []
avg_cs = []
for i, subject in enumerate(subjects):
    print("Processing " + subject + " ...")
    file_avg_c = opj(settings.work_dir_ctrbty, subject + "_avg_c.p")
    file_avg_input_nrj = opj(settings.work_dir_ctrbty, subject + "_avg_input_nrj.p")
    if os.path.exists(file_avg_input_nrj):
        continue

    dcm = dataloader.get_dCM(subject, settings.scale, zero_diagonal=True)
    dcm_rand = dataloader.randomMatrix(dcm, iter=1000)

    avg_inp_nrj, avg_controllability = avg_nrj_and_ctrbty(dcm)
    pickle.dump(avg_controllability, open(file_avg_c, "wb"))
    pickle.dump(avg_inp_nrj, open(file_avg_input_nrj, "wb"))
