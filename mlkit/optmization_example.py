import numpy as np
import pandas as pd
from optimization.generic_algorithm.GA import GA
import matplotlib.pyplot as plt

def evaluate(population):
    ev = []
    for i in population:
        sum = 0
        for k in range(len(i)):
            sum += k * i[k]
        ev.append(sum)

    return ev
	
model = GA(max_gen = 200, replace = True, n_individuals=50)
model.fit(evaluate, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.plot(range(0, len(model.bests)), model.bests)
plt.show()
print("solution: ")
print(model.bests)