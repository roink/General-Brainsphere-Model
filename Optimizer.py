import BrainsphereModel
import numpy as np

import sys


class Optimizer:
    def __init__(self, bsm: BrainsphereModel.BrainsphereModel, type=None, ):
        self.bsm = bsm
        self.type = type

    def optimize(self):
        if self.type is None:
            self.simulated_annealing(max_try=15, minT=0.001)
            self.adam(learning_rate=1e-2, max_iter=100)
            self.adam(learning_rate=5e-3, max_iter=100)
            self.adam(learning_rate=1e-3)
        elif type == "annealing":
            self.simulated_annealing()
        elif type == "adam":
            self.adam()
        else:
            raise AttributeError('Usupported Optimizer Type ' + type)

    def adam(self, learning_rate=1e-3, max_iter=1000):
        beta1 = .9
        beta2 = .999
        iterCount = 1
        M = {}
        R = {}
        for key in self.bsm.params:
            M[key] = 0.0
            R[key] = 0.0

        while iterCount < max_iter:
            params_old = self.bsm.params.copy()
            grad = self.bsm.gradient4()
            print(self.bsm.params)
            print(self.bsm.lastloss)
            sys.stdout.flush()

            params_updated = {}
            for key, value in params_old.items():
                M[key] = beta1 * M[key] + (1. - beta1) * grad[key]
                R[key] = beta2 * R[key] + (1. - beta2) * grad[key] ** 2

                M_hat = M[key] / (1. - beta1 ** iterCount)
                R_hat = R[key] / (1. - beta2 ** iterCount)
                params_updated[key] = value - learning_rate * M_hat / (np.sqrt(R_hat) + 1e-8)
            self.bsm.params = params_updated
            iterCount += 1

    def simulated_annealing(self, Tinit=1, minT=1e-8, max_try=15, minF=0, max_consec_rejections=40, max_success=5):

        k = 1

        itry = 0
        success = 0
        consec = 0
        T = Tinit
        initenergy = self.bsm.run()
        oldenergy = initenergy
        total = 0

        parent = self.bsm.params

        while True:
            itry += 1
            current = parent

            if itry >= max_try or success >= max_success:
                if T < minT or consec >= max_consec_rejections:
                    total += itry
                    break
                else:
                    T *= .8
                    print('  T = %7.5f, loss = %10.5f\n' % (T, oldenergy))
                    total += itry
                    itry = 1
                    success = 0

            newparam = {}
            for key, value in current.items():
                newparam[key] = value * np.random.lognormal(sigma=0.5 * T)
            self.bsm.params = newparam
            print(newparam)
            newenergy = self.bsm.run()
            print(newenergy)
            sys.stdout.flush()

            if (newenergy < minF):
                parent = current
                self.bsm.params = parent

                break

            if (oldenergy - newenergy) > 1e-6:
                parent = newparam
                oldenergy = newenergy
                success += 1
                consec = 0
            else:
                if np.random.rand() < np.exp((oldenergy - newenergy) / (k * T)):
                    parent = newparam
                    self.bsm.params = parent
                    oldenergy = newenergy
                    success += 1
                else:
                    consec += 1
        self.bsm.params = parent
