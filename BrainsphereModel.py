import numpy as np
from initializer import Initializer
from producer import Producer
from loss import Loss
from diffusor import Diffusor
from joblib import Parallel, delayed


class BrainsphereModel:
    def __init__(self, functional_connectivity, patient_data, **kwargs):
        self.types = ['ConcentrationLinear', 'Constant', 'ConcentrationSigmoid', 'WeightedDegreeLinear',
                      'WeightedDegreeSigmoid']
        self.producer = Producer(self.types)
        self.params = self.producer.params

        for key, value in kwargs.items():
            if key == "nodeCoordinates":
                self.nodeCoordinates = value
            elif key == "optimizer":
                self.optimizer = value
            elif key == "loss":
                self.loss = value
            elif key == "euclideanAdjacency":
                self.euclideanAdjacency = value
            elif key == "producer":
                self.producer = value
            elif key == "diffuser":
                self.diffuser = value
            elif key == "params":
                self.params.update(value)
            else:
                raise TypeError("Illegal Keyword '" + str(key) + "'")

        self.functionalConnectivity = functional_connectivity
        self.patientData = patient_data
        self.numNodes, _ = np.shape(functional_connectivity)
        self.loss = Loss("mse", self.patientData)
        self.lastloss = 0

        self.reset()

    def reset(self):
        self.initializer = Initializer("braak1", self.numNodes, self.params)
        self.concentration = self.initializer.get()
        self.concentrationHistory = np.copy(self.concentration)

        self.producer = Producer(self.types)
        self.diffusor = Diffusor("euclidean", self.params, EuclideanAdjacency=self.euclideanAdjacency)

    def run(self):
        stop_concentration = 1100
        timesteps = 2500

        self.reset()
        deltaT = 0.0001
        self.concentration += deltaT * (
                self.producer.produce(params=self.params, concentration=self.concentration,
                                      connectivity=self.functionalConnectivity) + self.diffusor.diffuse(
            self.concentration))
        self.concentrationHistory = np.append(self.concentrationHistory, self.concentration, axis=0)
        deltaConc = np.sum(self.concentrationHistory[1, :]) - np.sum(self.concentrationHistory[0, :])
        if deltaConc <= 0.0:
            return 9999999
        else:
            deltaT *= stop_concentration / timesteps / deltaConc

        while (np.sum(self.concentration) < stop_concentration) and (np.sum(deltaConc) > 0):
            deltaConc = deltaT * (
                    self.producer.produce(params=self.params, concentration=self.concentration,
                                          connectivity=self.functionalConnectivity) + self.diffusor.diffuse(
                self.concentration))
            self.concentration += deltaConc
            self.concentrationHistory = np.append(self.concentrationHistory, self.concentration, axis=0)
            # print(self.loss.get(self.concentrationHistory))

        self.lastloss = self.loss.get(self.concentrationHistory)

        return self.lastloss

    def gradient(self, loss = None):
        if loss is None:
            loss = self.run()


        params_new = {}
        params_old = self.params.copy()
        deltaX = {}
        for key, value in params_old.items():
            deltaX[key] = np.sign(np.random.randn()) * 0.01
            params_new[key] = value + deltaX[key]

        self.params = params_new
        new_loss = self.run()
        grad = {}
        self.params = params_old

        for key, value in params_old.items():
            grad[key] = (new_loss - loss) / (deltaX[key])

        return grad

    def gradient4(self):
        loss = self.run()
        gradients = Parallel(n_jobs=4)(delayed(self.gradient)(loss) for i in range(4))

        grad = {}

        for key in self.params:
            gradsum = 0
            count = 0
            for g in gradients:
                gradsum += g.get(key)
                count += 1.0
            grad[key] = gradsum / count
        return grad
