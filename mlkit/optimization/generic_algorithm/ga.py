"""
run genetic algorithm
"""
from enum import Enum
from typing import Union
import logging

import numpy as np

from individual import Individual
from mlkit.optimization.generic_algorithm.couple import Couple
from mlkit.optimization.generic_algorithm.population import Population

logger = logging.getLogger(__name__)


class ChrType(Enum):
    """
    Chromosome representation

    **binary**: chromosome genes are represented as binary. Example: [1 0 1 1]

    **permutation**: chromosome genes represents an order, each value appears only once. Example: [1, 3, 2, 4]

    **value**: chromosome genes represents any value. Example: [9 7 3 2]
    """
    BINARY = 'binary'
    PERMUTATION = 'permutation'
    VALUE = 'value'


class ParentSelection(Enum):
    """
    Parent selection method

    **fitness_proportionate**: Fitness proportionate selection method

    **tournament**: Tournament selection method
    """
    FITNESS_PROPORTIONATE = 'fitness_proportionate'
    TOURNAMENT = 'tournament'

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            'Invalid option for parent_selection. Valid options are: %s' % (', '.join([repr(m.value) for m in cls]))
        )


class GA:
    def __init__(
            self,
            chr_type: Union[str, ChrType] = ChrType.VALUE,
            gene_values: range = None,
            max_gen=100, population_size=10, n_chr=4, replace=False, p_crossover=0.5, p_mutation=0.005, n_elitism=1,
            parent_selection=ParentSelection.FITNESS_PROPORTIONATE
    ):
        self.chr_type = chr_type if isinstance(chr_type, ChrType) else ChrType(chr_type)
        self.gene_values = gene_values
        self.max_gen = max_gen
        self.population_size = population_size
        self.n_chr = n_chr
        self.replace = replace
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.n_elitism = n_elitism

        self.parent_selection = self._fitness_proportionate_selection
        self.crossover = self._ordered_crossover if self.chr_type == ChrType.PERMUTATION \
            else self._single_point_crossover

        self.best_individuals = []
        self.population = np.zeros(population_size)

        if gene_values is None:
            if self.chr_type == ChrType.BINARY:
                self.gene_values = [0, 1]
            elif self.chr_type == ChrType.PERMUTATION:
                self.gene_values = range(n_chr)
            else:
                raise ValueError('For chr_type, you must define gene_values. This is the range of possible values')

    def _initialize(self, evaluate):

        population = np.array([
            np.random.choice(
                self.gene_values,
                replace=(self.chr_type != ChrType.PERMUTATION),
                size=self.n_chr
            ) for _ in range(self.population_size)
        ])
        logger.debug(f'initial population: {population}')
        fitness_score = self._fitness_score_population(evaluate, population)
        return [Individual(individual, score) for individual, score in zip(population, fitness_score)]

    @staticmethod
    def _fitness_proportionate_selection(population):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm

        :return:
        """
        fitness_score = population.get_fitness_scores()
        parents_idx = np.random.choice(
            range(len(population)), size=len(population) + 1, p=fitness_score/sum(fitness_score)
        )
        parents_selected = population[parents_idx]
        return [Couple(parent1, parent2) for parent1, parent2 in zip(parents_selected[::2], parents_selected[1::2])]

    @staticmethod
    def _ordered_crossover(parent1, parent2):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

        :param parent1: parent 1
        :param parent2: parent 2
        :return: child 1 and child 2
        """
        start_end_gene = np.random.choice(range(len(parent1)), size=2, replace=False)
        start_gene, end_gene = min(start_end_gene), max(start_end_gene)

        def get_child(p1, p2):
            child_p1 = p1[start_gene:end_gene]
            child_p2 = [item for item in p2 if item not in child_p1]
            return np.concatenate([child_p2[:start_gene], child_p1, child_p2[start_gene:]])

        return get_child(parent1, parent2), get_child(parent2, parent1)

    def _crossover_population(self, parent_couples):
        offspring = []
        for couple in parent_couples:
            offspring.extend(self.crossover(*couple))

        return offspring

    @staticmethod
    def _single_point_crossover(parent1, parent2):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm

        :param parent1: parent 1
        :param parent2: parent 2
        :return: child 1 and child 2
        """
        split_point = np.random.choice(range(len(parent1)))

        child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
        child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))

        return child1, child2

#    def crossover(self):
#        new_generation = []
#        half = np.random.choice(range(self.n_chr)
#        for i in range(0, len(self.population), 2):
#            if np.random.uniform() < self.p_crossover:  # crossover
#                self.population[i][0:half]
#                self.population[i + 1][half:]
#                n1 = np.concatenate((self.population[i][0:half], self.population[i + 1][half:]), axis = 0)
#                n2 = np.concatenate((self.population[i][half:], self.population[i + 1][0:half]), axis = 0)
#                new_generation.append(n1)
#                new_generation.append(n2)
#            else:
#                new_generation.append(self.population[i])
#                new_generation.append(self.population[i+1])
#
#        self.population = new_generation

    def _mutate_population(self, population):
        return Population([
            self._mutate_gen(individual) if np.random.uniform() < self.p_mutation
            else individual for individual in population
        ])

    def _mutate_gen(self, individual):
        individual.chromosome[np.random.randint(len(individual.chromosome))] = np.random.choice(self.gene_values)
        return individual

#    def select(self, evaluate, elitism=True, best=[]):
#        self.population, best = self.gen_choice(
#            evaluate, self.population, p=evaluate(self.population)/sum(evaluate(self.population)), size=len(self.population), elitism=elitism, best_i=best
#        )
#        return best
#
#    def gen_choice(self, evaluate, population, p, size, elitism, best_i):
#        idx = np.random.choice(range(len(population)), p=p, size=size)
#        selected_population = []
#        for i in idx:
#            if evaluate([self.population[i]]) >= evaluate([best_i]):
#                best_i = self.population[i]
#            selected_population.append(self.population[i])
#
#        selected_population[0] = best_i
        
#        self.bests.append(best_i)
#        return selected_population, best_i

    @staticmethod
    def _fitness_score_population(evaluate, population):
        try:
            return np.array([evaluate(subject) for subject in population])
        except:
            pass

    def _population_ranked(self, fitness, population):
        return population[self._population_ranked_idx(fitness, population)]

    @staticmethod
    def _population_ranked_idx(fitness_score, population):
        return [idx for _, idx in sorted(zip(fitness_score, range(len(population))), reverse=True)]

    def run(self, evaluate):
        population = Population(self._initialize(evaluate))
        self.best_individuals = population.n_best_individuals(1)

        for i in range(self.max_gen):
            parent_couples = self.parent_selection(population)
            offspring = self._crossover_population(parent_couples)
            mutated_population = self._mutate_population(offspring)

            # elitism
            mutated_population.individuals[:self.n_elitism] = self.population.n_best_individuals(self.n_elitism)
            self.population = mutated_population

            fitness_score = self._fitness_score_population(evaluate, self.population)
            population_ranked_idx = self._population_ranked_idx(fitness_score, self.population)

            self.best_subjects.append(self.population[population_ranked_idx[0]])
            self.best_fitness.append(fitness_score[population_ranked_idx[0]])
        return self.best_fitness, self.best_subjects
