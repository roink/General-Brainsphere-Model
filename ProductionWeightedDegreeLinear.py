import numpy as np


class ProductionWeightedDegreeLinear:
    parameterName = 'WD-LinearA'

    def __init__(self):
        self.params = {self.parameterName: 1.0}

    def get_params(self):
        return self.params

    def produce(self, params=None, concentration=None, connectivity=None):
        if params is not None:
            self.params = params
        a = self.params.get(self.parameterName)
        return a * np.sum(connectivity, axis=0)
