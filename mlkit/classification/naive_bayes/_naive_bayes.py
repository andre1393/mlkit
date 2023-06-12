""" Naive Bayes Classifier simple implementation

This implementation is not meant to be faster, more robust or even better than any other available library like
Scikit-Learn but is meant to be as simple as possible to help understand the concepts instead.

https://www.ic.unicamp.br/~rocha/teaching/2011s2/mc906/aulas/naive-bayes-classifier.pdf
https://www.atoti.io/how-to-solve-the-zero-frequency-problem-in-naive-bayes/
https://towardsdatascience.com/continuous-data-and-zero-frequency-problem-in-naive-bayes-classifier-7784f4066b51
"""
from typing import List

import numpy as np
from pandas import Series, DataFrame
from pandas.api.types import is_numeric_dtype

from mlkit.classification.base_classifier import BaseClassifier
from mlkit.classification.naive_bayes._prior_knowledge import (
    ContinuousPriorKnowledge,
    CategoricalPriorKnowledge,
    PriorKnowledge
)
from mlkit.exceptions import NotFittedException


class NaiveBayes(BaseClassifier):

    def __init__(self, continuous_variables=None):
        super().__init__()
        self._continuous_variables = continuous_variables or {}
        self.p_x_given_y = {}
        self.classes_probability = None
        self.classes_count = None
        self.classes_ = None

    def fit(self, x: DataFrame, y: Series) -> None:
        """Given a X and y, fit a Naive Bayes classifier

        The fit outputs are:
        p_x_given_y: (dictionary)
            {
                'Yes': {
                    'Outlook': CategoricalPriorKnowledge(),
                    'Temperature': CategoricalPriorKnowledge(),
                    'Humidity': CategoricalPriorKnowledge(),
                    'Wind': CategoricalPriorKnowledge()
                    },
                'No': {
                    'Outlook': CategoricalPriorKnowledge(),
                    'Temperature': CategoricalPriorKnowledge(),
                    'Humidity': CategoricalPriorKnowledge(),
                    'Wind': CategoricalPriorKnowledge()
                }
            }

        And classes_probabilities: Series

            Yes     0.64
            No      0.36

        :param x: X (features)
        :param y: y (target)
        :return: None
        """
        self.classes_count = y.value_counts()
        self.classes_probability = self.classes_count / sum(self.classes_count)
        self.classes_ = list(self.classes_probability.index)

        for p_class in self.classes_probability.index:
            x_given_y = x[y == p_class]
            self.p_x_given_y[p_class] = {
                i: self._calculate_feature_prior(c, len(x[i].unique())) for i, c in x_given_y.items()
            }

    def _calculate_feature_prior(self, column: Series, unique_values: int) -> PriorKnowledge:
        """
        Check feature type and choose appropriate class to process probability

        :param column: feature column subset
        :param unique_values: quantity of unique values for feature column
        :return: instance of class that processes the feature
        """
        return (
            ContinuousPriorKnowledge(column)
            if (is_numeric_dtype(column) and column.name not in self._continuous_variables) or
               (self._continuous_variables.get(column.name, False))
            else CategoricalPriorKnowledge(column, unique_values)
        )

    def _likelihood(self, features: Series, c: str) -> float:
        """
        Compute likelihood for given class

        :param features: features from X dataframe
        :param c: yi class name
        :return: probability of x given yi
        """
        return np.prod([self.p_x_given_y[c][idx].get_prior_probability(value) for idx, value in features.items()])

    def _posteriori(self, features: Series):
        """
        Compute posteriori for given feature

        :param features: features series from X dataframe
        :return: posteriori
        """
        return [
            (c, class_prior * self._likelihood(features, c))
            for c, class_prior in self.classes_probability.items()
        ]

    def _maximum_a_posteriori(self, features: Series):
        """
        Compute posteriori for each class and return class with maximum value

        :param features: features series from X dataframe
        :return: class name
        """
        return max(self._posteriori(features), key=lambda x: x[1])[0]

    def _compute_proba(self, features: Series) -> List[float]:
        """
        Compute probability of each each class given x
        :param features: instance from X
        :return: list containing probability of each class
        """
        probabilities = self._posteriori(features)
        sum_probabilities = sum([p[1] for p in probabilities])
        return [(p_class[1]/sum_probabilities) for p_class in probabilities]

    def _check_fitted(self):
        """ Check if the fit() function has been called yet

        :return: None
        """
        if (self.p_x_given_y is None) or (self.classes_probability is None):
            raise NotFittedException()

    def predict(self, x: DataFrame) -> Series:
        """
        Apply prediction on a dataframe

        :param x: X (features)
        :return: Series with the prediction class for each instance in X
        """
        self._check_fitted()

        return x.apply(self._maximum_a_posteriori, axis=1)

    def predict_proba(self, x):
        """
        Apply predict_proba on a dataframe

        :param x: X (features)
        :return: Series with probability for each class
        """
        self._check_fitted()
        return x.apply(self._compute_proba, axis=1)
