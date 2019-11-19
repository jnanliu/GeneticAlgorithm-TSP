# -*- coding: UTF-8 -*-

import os
import random
import itertools
import time
import numpy as np
from tqdm import tqdm
from eprogress import LineProgress
import matplotlib.pyplot as plt

from genetic import Genetic

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_NAME = 'att48.txt'
TOTAL_EPOCHS = 1500
TOTAL_POINTS = 48
INITAIL_POPULATION_SIZE = 150
POINTS_CX = np.empty([INITAIL_POPULATION_SIZE, TOTAL_POINTS])
POINTS_CY = np.empty([INITAIL_POPULATION_SIZE, TOTAL_POINTS])

def fitnessFunction(gene:np.ndarray) -> np.ndarray :

    x, y = gene.copy(), gene.copy()
    xx, yy = np.empty([INITAIL_POPULATION_SIZE, TOTAL_POINTS]), np.empty([INITAIL_POPULATION_SIZE, TOTAL_POINTS])
    y[:, :-1], y[:, -1] = x[:, 1:].copy(), x[:, 0].copy()
    
    for i in range(INITAIL_POPULATION_SIZE) :
        xx[i], yy[i] = POINTS_CX[i, x[i]].copy() - POINTS_CX[i, y[i]].copy(), \
            POINTS_CY[i, x[i]].copy() - POINTS_CY[i, y[i]].copy()
    ret = np.sqrt(np.square(xx) + np.square(yy))
    ret = np.sum(ret, axis=-1)

    return 1.0 / ret;

if __name__ == '__main__' :

    if not os.path.exists(os.path.join(BASE_DIR, 'points.txt')) :

        with open(os.path.join(BASE_DIR, 'points.txt'), 'w') as f :

            for i in range(TOTAL_POINTS) :

                x, y = random.randint(0, 200), random.randint(0, 200)
                f.write(str((x, y)) + '\n')

    with open(os.path.join(BASE_DIR, DATA_NAME), 'r') as f :

        for i in range(TOTAL_POINTS) :
            
            s = f.readline().split(' ')
            POINTS_CX[:, i], POINTS_CY[:, i] = float(s[1]), float(s[2])
    
    genetic = Genetic(TOTAL_POINTS, fitnessFunction, initialGeneNum = INITAIL_POPULATION_SIZE)
    genetic.initialPopulation()
    pbar = LineProgress(total=TOTAL_EPOCHS, symbol='>')
    x, y = [], []
    startTime = time.time()
    for i in range(TOTAL_EPOCHS) :

        genetic.evaluate()
        genetic.crossover()
        genetic.mutate()
        pbar.update(i)

        x.append(i)
        y.append(1.0 / genetic.getBest()[0])

    totalTime = time.time() - startTime
    print('')
    print("total time: {:.0f}min{:.0f}s".format(totalTime / 60, totalTime % 60))
    plt.plot(x, y, label="distance")

    plt.legend()
    #plt.savefig(os.path.join(BASE_DIR, '{}_{}epochs.jpg'.format(DATA_NAME.split('.')[0], TOTAL_EPOCHS)))
    plt.show()

    print('')
    print(1.0 / genetic.getBest()[0], genetic.getBest()[1])

    