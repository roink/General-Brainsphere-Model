import numpy as np


class Loss:
    def __init__(self, loss_type, patient_data):
        if loss_type == "mse":
            def loss_func(a, b):
                return np.square(a - b).sum()

            self.lossfunc = loss_func
        self.totalLoss = 0
        self.patient_data = patient_data

    def get(self, concentration_history):
        self.totalLoss = 0
        sums = np.sum(concentration_history,axis=1)
        #print(np.shape(sums))
        m, _ = np.shape(self.patient_data)

        try:
            for i in range(m):
                def patient_loss_func(x):
                    return self.lossfunc(x, self.patient_data[i, :])
                candidates = concentration_history[np.abs(sums-np.sum(self.patient_data[i,:]))<50,:]
                #print(np.shape(candidates))

                a = np.apply_along_axis(patient_loss_func, 1, candidates)
                self.totalLoss += np.amin(a)
        except :
            return 9999
        return self.totalLoss
