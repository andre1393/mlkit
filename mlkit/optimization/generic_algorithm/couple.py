from mlkit.optimization.generic_algorithm.individual import Individual


class Couple:
    def __init__(self, parent1: Individual, parent2: Individual):
        self.parent1 = parent1
        self.parent2 = parent2
