import numpy as np


class ProductionConcentrationInverseSigmoid:
    parameterNameA = 'ConcentrationInvSigmoidA'
    parameterNameB = 'ConcentrationInvSigmoidB'
    parameterNameC = 'ConcentrationInvSigmoidC'

    def __init__(self):
        self.params = {self.parameterNameA: 1.0,
                       self.parameterNameB: -1.0,
                       self.parameterNameC: 2.0}

    def get_params(self):
        return self.params

    def produce(self, params=None, concentration=None, connectivity=None):
        if params is not None:
            self.params = params
        a = self.params.get(self.parameterNameA)
        b = self.params.get(self.parameterNameB)
        c = self.params.get(self.parameterNameC)
        return a / (1 + np.exp(-b * (concentration - c)))
