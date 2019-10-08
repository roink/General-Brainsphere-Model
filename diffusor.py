import numpy as np


class Diffusor:
    def __init__(self, diffusor_type: str, params: dict, **kwargs):
        self.params = params
        if diffusor_type == "euclidean":
            self.euclidean_adjacency = kwargs.get("EuclideanAdjacency")

            def diffuse(concentration):
                return np.sum((concentration.transpose() * np.ones((1, concentration.size)) -
                              concentration * np.ones((concentration.size, 1))) * self.euclidean_adjacency, axis=1)

            self.diffuse = diffuse

    def diffuse(self, concentration):
        return self.diffuse(concentration)
