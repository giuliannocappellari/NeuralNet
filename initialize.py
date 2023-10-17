import numpy as np
from typing import Callable
from abc import ABC


class WeightsInitialize(ABC):

    def __init__(self, neurons:int, input_size:int, seed:int=1234) -> None:
        self.input_size = input_size
        self.neurons = neurons
        self.seed = np.random.seed(seed)

    def initialize_weights(self):
        ...


class HeNormal(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        stddev = np.sqrt(2. / self.neurons)
        return np.random.normal(0, stddev, (self.neurons, self.input_size))
    

class HeUniform(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        limit = np.sqrt(6. / self.neurons)
        return np.random.uniform(-limit, limit, (self.neurons, self.input_size))


class XavierNormal(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        stddev = np.sqrt(2. / (self.neurons + self.input_size))
        return np.random.normal(0, stddev, (self.neurons, self.input_size))


class XavierUniform(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        limit = np.sqrt(6. / (self.neurons + self.input_size))
        return np.random.uniform(-limit, limit, (self.neurons, self.input_size))
    

class Normal(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        return np.random.normal(0, 1, (self.neurons, self.input_size))


class Uniform(WeightsInitialize):

    def __init__(self, neurons: int, input_size: int, seed: int = 1234) -> None:
        super().__init__(neurons, input_size, seed)

    def initialize_weights(self) -> np.array:
        np.random.seed(self.seed)
        return np.random.uniform(-1, 1, (self.neurons, self.input_size))
    
    
class InitializeBias():

    def __init__(self, neurons:int) -> None:
        self.neurons = neurons

    def initialize_bias(self) -> np.array:
        return np.zeros((self.neurons,1))