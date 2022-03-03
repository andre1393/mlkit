from mlkit.optimization.generic_algorithm.ga import GA
import matplotlib.pyplot as plt
import numpy as np


def evaluate_salesman_travel():
    """
    https://core.ac.uk/download/4836412.pdf
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
        return sum([distances[int(_from), int(to)] for _from, to in zip(subject, np.roll(subject, -1))])

    return evaluate


def main():
    model = GA(max_gen=200, chr_type='permutation', n_chr=9, replace=True, population_size=50)
    model.run(evaluate_salesman_travel())
    plt.plot(range(0, len(model.best_fitness)), model.best_fitness)
    plt.show()
    print("solution: ")
    print(model.best_subjects)


if __name__ == '__main__':
    main()
