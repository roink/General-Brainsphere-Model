import numpy as np


class Initializer:
    out: np.ndarray

    def __init__(self, init_type, num_nodes, params):
        if init_type == "braak1":
            if ("lowerInit" in params) and ("upperInit" in params):
                self.out = np.ones((1, num_nodes)) * params.get("lowerInit")
                self.out[0, 549:551] = params.get("upperInit")
                self.out[0, 565:567] = params.get("upperInit")
            else:
                raise Exception("Parameters not defined")

    def get(self):
        return self.out
