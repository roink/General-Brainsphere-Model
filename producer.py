from joblib import Parallel, delayed

class Producer:
    def __init__(self, producer_type):
        self.functions = set()
        self.params = {}
        self.add(producer_type)

    def add(self, producer_type):
        if isinstance(producer_type,str):
            if producer_type == 'ConcentrationLinear':
                from ProductionConcentrationLinear import ProductionConcentrationLinear as productionFunction

            elif producer_type == 'Constant':
                from ProductionConstant import ProductionConstant as productionFunction

            elif producer_type == 'ConcentrationSigmoid':
                from ProductionConcentrationSigmoid import ProductionConcentrationSigmoid as productionFunction

            elif producer_type == 'ConcentrationInverseSigmoid':
                from ProductionConcentrationInverseSigmoid import ProductionConcentrationInverseSigmoid as productionFunction

            elif producer_type == 'WeightedDegreeLinear':
                from ProductionWeightedDegreeLinear import ProductionWeightedDegreeLinear as productionFunction

            elif producer_type == 'WeightedDegreeSigmoid':
                from ProductionWeightedDegreeSigmoid import ProductionWeightedDegreeSigmoid as productionFunction

            elif producer_type == 'WeightedDegreeInverseSigmoid':
                from ProductionWeightedDegreeInverseSigmoid import ProductionWeightedDegreeInverseSigmoid as productionFunction
            else:
                raise AttributeError('Unknow Producer Type ' + producer_type)

            p = productionFunction()
            self.functions.add(p)
            self.params.update(p.get_params())
        elif isinstance(producer_type,list):
            for s in producer_type:
                self.add(s)
        else:
            raise AttributeError("Producer_Type must be a string or list of Strings")

    def produce(self, params=None, concentration=None, connectivity=None):
        out = concentration * 0
        for f in self.functions:
            out += f.produce(params, concentration, connectivity)

        return out
