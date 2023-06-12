class Individual:
    def __init__(self, chromosome, fitness_score: float = None):
        self.chromosome = chromosome
        self.fitness_score = fitness_score

    def __lt__(self, other) -> bool:
        return self.fitness_score < other.fitness_score

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self._equal(other)

    def __gt__(self, other) -> bool:
        return self.fitness_score > other.fitness_score

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or self._equal(other)

    def _equal(self, other) -> bool:
        return self.fitness_score == other.fitness_score
