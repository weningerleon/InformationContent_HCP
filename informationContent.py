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
from sklearn.neighbors import KernelDensity


def fit_ic(activations):

    models = []

    for i in range(activations.shape[0]):
        ts = activations[i, :]
        #%% calc bandwidth: Silverman, B.W.(1986).Density Estimation for Statistics and
        # Data Analysis. London: Chapman & Hall/CRC. p4. ISBN 978-0-412-24620-3.
        n = ts.shape[0]
        std = ts.std()
        h = 1.06*std*np.power(n,-1/5)

        model = KernelDensity(bandwidth=h, kernel='gaussian')

        model.fit(ts.reshape(-1, 1))
        models.append(model)

    return tuple(models)


def predict_ic(activations, modelz):

    ic = np.zeros(activations.shape)

    for i in range(activations.shape[0]):
        model = modelz[i]
        ts = activations[i, :]
        log_p = model.score_samples(ts.reshape(-1, 1))
        ic[i, :] = -log_p

    return ic