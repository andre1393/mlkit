from abc import ABC, abstractmethod

import numpy as np
import math


class PriorKnowledge(ABC):
    @abstractmethod
    def get_prior_probability(self, value):
        pass


class CategoricalPriorKnowledge(PriorKnowledge):
    def __init__(self, column, unique_values):
        self._unique_values = unique_values
        self._counts = column.value_counts()
        self._probabilities = ((self._counts + 1)/(sum(self._counts) + self._unique_values)).to_dict()

    def get_prior_probability(self, value):
        return self._probabilities.get(value, 1/(sum(self._counts) + self._unique_values))


class ContinuousPriorKnowledge(PriorKnowledge):
    def __init__(self, column):
        self.mean = sum(column)/len(column)
        self.sd = math.sqrt(sum((np.array(column) - self.mean) ** 2)/(len(column) - 1))

    def get_prior_probability(self, value):
        return math.exp(-((value - self.mean)**2)/(2*(self.sd**2)))/(math.sqrt(2*math.pi)*self.sd)
