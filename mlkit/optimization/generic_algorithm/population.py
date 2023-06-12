from typing import Union, List
from numpy import array


class Population:
    def __init__(self, individuals: Union[array, List]):
        self.individuals = array(individuals) if isinstance(individuals, List) else individuals

    def n_best_individuals(self, n: int) -> array:
        return sorted([i for i in self.individuals], reverse=True)[:n]

    def get_chromosomes(self) -> List:
        return [i.chromosome for i in self.individuals]

    def get_fitness_scores(self):
        return [i.fitness_score for i in self.individuals]

    def size(self):
        return len(self.individuals)

    def __getitem__(self, item):
        return self.individuals[item]
