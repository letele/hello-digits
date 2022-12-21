import numpy as np

class Sigmoid:
    """
    A class that represents the sigmoid activation function
    """
    def func(self, x):
        return 1/(1+np.exp(-x))
    
    def der(self, x):
        return self.func(x)*(1 - self.func(x))