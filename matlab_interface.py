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

import numpy as np
import matlab.engine
import settings

eng = matlab.engine.start_matlab("-singleCompThread")
eng.addpath(eng.genpath(settings.matlab_dir))


def calc_min_eng_cont(a, xf, x0=0):
    # A: connectivity matrix
    # xf: final state
    # X0: starting state, set to zero
    # B: control nodes, set to identity matrix

    A = matlab.double(a.tolist())
    Xf = matlab.double(xf.tolist())
    Xf = eng.transpose(Xf)
    B = eng.eye(a.shape[0])
    if np.array_equal(x0,0):
        X0 = eng.zeros(a.shape[0], 1)
    else:
        X0 = matlab.double(x0.tolist())
        X0 = eng.transpose(X0)
    T = eng.double(settings.time_horizon)
    C = eng.double(settings.norm_fac)

    # Calls the min control energy
    X, U, N_err = eng.min_cont_nrj(A, T, B, X0, Xf, C, nargout=3)
    u = np.asarray(list(U))
    min_energy = np.sum(np.square(u)) / np.cumprod(u.shape)[-1]

    return min_energy


def avg_nrj_and_ctrbty(a):
    A = matlab.double(a.tolist())
    B = eng.eye(a.shape[0])
    T = eng.double(settings.time_horizon)
    C = eng.double(settings.norm_fac)

    # Calls the ctrb gramian function
    G = eng.ctrb_gramian(A, B, T, C, nargout=1)
    g = np.asarray(list(G))
    g_inv = np.linalg.inv(g)
    avg_controllability = np.trace(g)
    avg_inp_nrj = np.trace(g_inv)

    return avg_inp_nrj, avg_controllability
