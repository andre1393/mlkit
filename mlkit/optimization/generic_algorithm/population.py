from numpy import array


class Population:
    def __init__(self, individuals: array):
        self.individuals = individuals

    def n_best_individuals(self, n):
        idx = sorted([i for i in self.individuals], reverse=True)[:n]
        return self.individuals[idx]

    def get_chromosomes(self):
        return [i.chromosome for i in self.individuals]

    def get_fitness_scores(self):
        return [i.fitness_score for i in self.individuals]
