import numpy as np
import pandas as pd
from ga import GA
import matplotlib.pyplot as plt

def evaluate(population):
    ev = []
    for i in population:
        ev.append(sum(i))
    
    return ev
	
model = GA(max_gen = 100, replace = True)
model.fit(evaluate, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.plot(range(0, len(model.bests)), model.bests)
print(model.bests)