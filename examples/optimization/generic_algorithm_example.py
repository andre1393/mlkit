from mlkit.optimization.generic_algorithm.ga import GA
import matplotlib.pyplot as plt
import numpy as np


def evaluate_salesman_travel():
    """
    Matrix to represent the traveling salesman problem.
    https://core.ac.uk/download/4836412.pdf
    Each cell represents the distance from (row) one stop to another (column)
    """
    distances = np.array([
        [0, 4.8, 2.0, 1.6, 2.8, 3.3, 4.9, 2.3, 0.8],
        [4.8, 0, 3.6, 5.6, 6.8, 1.9, 9.6, 2.8, 5.6],
        [2.0, 3.6, 0, 2.1, 4.3, 2.0, 6.3, 2.1, 2.8],
        [1.6, 5.6, 2.1, 0, 4.0, 4.1, 4.3, 3.5, 1.9],
        [2.8, 6.8, 4.3, 4.0, 0, 6.2, 4.3, 4.1, 2.2],
        [3.3, 1.0, 2.0, 4.1, 6.2, 0, 8.3, 2.3, 4.5],
        [4.9, 9.6, 6.3, 4.3, 4.3, 8.3, 0, 7.1, 4.2],
        [2.3, 2.8, 2.1, 3.5, 4.1, 2.3, 7.1, 0, 2.9],
        [0.8, 5.6, 2.8, 1.9, 2.2, 4.5, 4.2, 2.9, 0]
    ])

    def evaluate(subject):
        """
        Calculates a fitness score for a subject. The command `np.roll(subject, -1)`
        shifts each element from the array one index to left. \n
        Example `subject = [1 5 2 3 0 8 4 7]`, `np.roll = [5 2 3 0 8 4 7 1]`\n
        The fitness score is calculated by summing the distance value for each pair subject/`np.roll`.\n
        Since GA is an optimization algorithm, we need to multiply the result by -1 to actually minimize the distance
        :param subject: subject (chromosome) represented by a numpy array.\n
        Where the indexes represent each stop, in order
        :return: fitness score for the subject
        """
        return sum([distances[int(_from), int(to)] for _from, to in zip(subject, np.roll(subject, -1))]) * -1

    return evaluate


def main():
    model = GA(max_gen=2000, chr_type='permutation', n_chr=9, population_size=50)
    best_solution, fitness_score = model.run(evaluate_salesman_travel())
    plt.plot(range(0, len(model.best_fitness_score)), [v * -1 for v in model.best_fitness_score])
    plt.grid()
    plt.xlabel('generation')
    plt.ylabel('min distance')
    plt.title('Generation vs Min Distance')
    plt.show()

    print(f'solution: {best_solution} | fitness_score: {fitness_score}')


if __name__ == '__main__':
    main()
