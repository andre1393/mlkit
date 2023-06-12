"""
run genetic algorithm
"""
from enum import Enum
from typing import Union
import logging

import numpy as np

from .individual import Individual
from .couple import Couple
from .population import Population

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
            max_gen=100,
            population_size=10,
            n_chr=None,
            p_mutation=0.005,
            n_elitism=1,
            parent_selection=ParentSelection.FITNESS_PROPORTIONATE,
            initial_population=None
    ):
        parent_selection_methods = {
            ParentSelection.FITNESS_PROPORTIONATE: self._fitness_proportionate_selection
        }

        self.chr_type = chr_type if isinstance(chr_type, ChrType) else ChrType(chr_type)
        self.gene_values = gene_values
        self.max_gen = max_gen
        self.population_size = population_size
        self.n_chr = n_chr
        self.p_mutation = p_mutation
        self.n_elitism = n_elitism
        self.best_fitness_score = None
        self.solution = None
        self.parent_selection = parent_selection_methods.get(parent_selection)
        self.crossover = self._ordered_crossover if self.chr_type == ChrType.PERMUTATION \
            else self._single_point_crossover
        self.initial_population = initial_population
        self.best_individuals = []
        self.population = np.zeros(population_size)

        if gene_values is None:
            if self.chr_type == ChrType.BINARY:
                self.gene_values = [0, 1]
            elif self.chr_type == ChrType.PERMUTATION:
                self.gene_values = range(n_chr)
            else:
                raise ValueError('For chr_type, you must define gene_values. This is the range of possible values')

        if (self.initial_population is None) and (not self.n_chr):
            raise ValueError('Either initial_population or n_chr must be provided')

    def _initialize(self, evaluate):
        """
        Initialize the population using the provided params and fitness score evaluate function

        :param evaluate: fitness score evaluate function
        :return: the initial population with fitness score set for every individual
        """
        population = Population(
            np.array([
                Individual(
                    np.random.choice(
                        self.gene_values,
                        replace=(self.chr_type != ChrType.PERMUTATION),
                        size=self.n_chr
                    )
                ) for _ in range(self.population_size)
            ]))
        logger.debug(f'initial population: {population}')
        return self._fitness_score_population(evaluate, population)

    @staticmethod
    def _fitness_proportionate_selection(population):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        **Fitness Proportionate Selection**

        :return: list of parents selected
        """
        fitness_score = population.get_fitness_scores()
        parents_selected = np.random.choice(
            population.individuals, size=population.size() + 1, p=fitness_score/sum(fitness_score)
        )
        return [Couple(parent1, parent2) for parent1, parent2 in zip(parents_selected[::2], parents_selected[1::2])]

    @staticmethod
    def _ordered_crossover(parent1, parent2):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        **Davis Order Crossover**

        :param parent1: parent 1
        :param parent2: parent 2
        :return: child 1 and child 2
        """
        start_end_gene = np.random.choice(range(len(parent1.chromosome)), size=2, replace=False)
        start_gene, end_gene = min(start_end_gene), max(start_end_gene)

        def get_child(p1, p2):
            chr1, chr2 = p1.chromosome, p2.chromosome
            child_p1 = chr1[start_gene:end_gene]
            child_p2 = [item for item in chr2 if item not in child_p1]
            return Individual(np.concatenate([child_p2[:start_gene], child_p1, child_p2[start_gene:]]))

        return get_child(parent1, parent2), get_child(parent2, parent1)

    def _crossover_population(self, parent_couples):
        """
        Apply crossover function for each individual pair selected from population
        
        :param parent_couples: list of a pair of individuals to generate next generation offspring
        :return: individuals after crossover
        """
        offspring = []
        for couple in parent_couples:
            offspring.extend(self.crossover(*couple.get_parents()))

        return offspring

    @staticmethod
    def _single_point_crossover(parent1, parent2):
        """
        https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm
        **Single Point Crossover**
        In this one-point crossover, a random crossover point is selected and the tails of its two parents
        are swapped to get new off-springs.

        :param parent1: parent 1
        :param parent2: parent 2
        :return: child 1 and child 2
        """
        split_point = np.random.choice(range(len(parent1)))

        child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
        child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))

        return child1, child2

    def _mutate_population(self, population):
        """
        Apply mutation on a random part of the population

        :param population: population
        :return: population after mutation function applied
        """
        return Population([
            self._mutate_gen(individual) if np.random.uniform() < self.p_mutation
            else individual for individual in population
        ])

    def _mutate_gen(self, individual):
        """
        Apply gene mutation for a specific individual.

        :param individual: individual to apply mutation in
        :return: individual mutated
        """
        individual.chromosome[np.random.randint(len(individual.chromosome))] = np.random.choice(self.gene_values)
        return individual

    @staticmethod
    def _fitness_score_population(evaluate, population):
        """
        Apply fitness score function for each individual in population.

        :param evaluate: fitness score evaluate function
        :param population: population
        :return: population with the fitness score set for every individual
        """
        for idx, individual in enumerate(population.individuals):
            population[idx].fitness_score = evaluate(individual.chromosome)
        return population

    def run(self, evaluate):
        """
        Run the GA based on the params provided

        :param evaluate: Fitness Score evaluate function
        :return: the best solution and the respective fitness score
        """
        self.population = self.initial_population or self._initialize(evaluate)
        self.best_individuals = self.population.n_best_individuals(1)
        population = self.population
        for i in range(self.max_gen):
            parent_couples = self.parent_selection(population)
            offspring = self._crossover_population(parent_couples)
            mutated_population = self._mutate_population(offspring)

            # elitism
            mutated_population.individuals[:self.n_elitism] = self.population.n_best_individuals(self.n_elitism)
            population = mutated_population

            population = self._fitness_score_population(evaluate, population)
            self.population = population
            
            self.best_individuals.append(self.population.n_best_individuals(1)[0])
            
        self.best_fitness_score = [i.fitness_score for i in self.best_individuals]
        self.solution = max(self.best_individuals)
        return self.solution.chromosome, self.solution.fitness_score
