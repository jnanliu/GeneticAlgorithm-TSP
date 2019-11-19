# -*- coding: UTF-8 -*-

import os
import math
import numpy as np
import random
from copy import deepcopy

class Genetic() :

    def __init__(self, totalPoints, fitnessFunction, initialGeneNum = 500, EliteRepRatio=0.25, crossoverProb=0.8, mutateProb=0.1) :
        '''
        遗传算法

        Args:

        totalPoints: 总点数

        initialGeneNum：初始种群大小

        EliteRepRatio: 精英复制比例

        fitnessFunction: 适应度函数

        crossoverProb: 交叉概率

        mutateProb: 变异概率
        '''
        
        self.totalPoints = totalPoints
        self.codeLen = math.ceil(math.log(totalPoints, 2))
        self.geneLen = self.totalPoints * self.codeLen
        self.fitnessFunction = fitnessFunction
        self.initialPopulationSize = initialGeneNum
        self.EliteRepRatio = EliteRepRatio
        self.crossoverPopulationSize = int(self.initialPopulationSize * (1. - self.EliteRepRatio))
        self.crossoverProb = crossoverProb
        self.mutateProb = mutateProb
        self.population = np.empty([self.initialPopulationSize, self.totalPoints], dtype=int)
        self.crossoverPopulation = np.empty([self.crossoverPopulationSize, self.totalPoints], dtype=int)

    def encode(self) :
        '''
        染色体编码
        '''

        return None

    def decode(self) :
        '''
        染色体解码
        '''

        return None

    def initialPopulation(self) :
        '''
        初始化族群

        Args:

        initialPopulationSize: 族群中个体的数目
        '''

        initialList = np.arange(self.totalPoints)
        for i in range(self.initialPopulationSize) :

            np.random.shuffle(initialList)
            self.population[i,:] = initialList.copy()

    def evaluate(self) :
        '''
        选择算子
        '''

        currentPopulation = self.population.copy()
        #计算适应度
        prob = self.fitnessFunction(self.population)
        prob = prob * 1. / prob.sum()
        ptrs = np.arange(len(currentPopulation))
        #依据适应度选择M个个体
        indices = np.random.choice(ptrs, size=self.crossoverPopulationSize, p=prob)
        for idx, i in enumerate(indices) :

            self.crossoverPopulation[idx] = currentPopulation[i].copy()

    def crossover(self) :
        '''
        交叉算子
        '''

        #保留当前族群适应度最高的个体
        currentPopulation = self.population.copy()
        fitnesses = self.fitnessFunction(self.population)
        indices = np.argsort(fitnesses)[:: -1]
        indices = indices[: self.initialPopulationSize - self.crossoverPopulationSize]

        #交叉，采用基于位置的交叉
        pairs = np.arange(self.crossoverPopulationSize)
        np.random.shuffle(pairs)
        ptrs = np.arange(self.totalPoints)
        for i in range(int(len(pairs) / 2)) :

            # 交叉概率，不知道这样写对不对。。
            p = np.random.uniform()
            if p > self.crossoverProb :
                continue
            crossoverPointNum = np.random.randint(low = 1, high = self.totalPoints)
            crossoverFrag = np.random.choice(ptrs, crossoverPointNum, replace=False)
            marks = np.ones(self.totalPoints)
            for j in crossoverFrag :

                marks[j] = 0. 
            findices = self.crossoverPopulation[pairs[i * 2], crossoverFrag].copy()
            mindices = self.crossoverPopulation[pairs[i * 2 + 1], crossoverFrag].copy()
            bpfgene, bpmgene = self.crossoverPopulation[pairs[i * 2]].copy(), self.crossoverPopulation[pairs[i * 2 + 1]].copy()
            self.crossoverPopulation[pairs[i * 2]], self.crossoverPopulation[pairs[i * 2 + 1]] = \
                (self.crossoverPopulation[pairs[i * 2]] * marks).copy(), (self.crossoverPopulation[pairs[i * 2 + 1]] * marks).copy()
            fptr, mptr = 0, 0
            for point in bpmgene :

                if (findices == point).sum() == 1 :
                    while fptr < self.totalPoints and marks[fptr] != 0 :
                        fptr = fptr + 1
                    if fptr < self.totalPoints and marks[fptr] == 0 :
                        self.crossoverPopulation[pairs[i * 2], fptr] = deepcopy(point)
                        fptr = fptr + 1
            for point in bpfgene :

                if (mindices == point).sum() == 1 :
                    while mptr < self.totalPoints and marks[mptr] != 0 :
                        mptr = mptr + 1
                    if mptr < self.totalPoints and marks[mptr] == 0 :
                        self.crossoverPopulation[pairs[i * 2 + 1], mptr] = deepcopy(point)
                        mptr = mptr + 1

        #精英复制
        self.population[: self.initialPopulationSize - self.crossoverPopulationSize] = self.population[indices].copy()
        self.population[self.initialPopulationSize - self.crossoverPopulationSize: ] = self.crossoverPopulation.copy()

    def mutate(self) :
        '''
        变异算子
        '''

        currentGene = self.population.copy()
        oldFitness = self.fitnessFunction(self.population)
        for i in range(self.initialPopulationSize) :

            #基于位置的变异
            tag = np.random.uniform()
            seq = np.arange(self.totalPoints)
            if tag <= self.mutateProb :

                mutateGeneFrag = np.random.choice(seq, 2, replace=False)
                mutateGeneFrag = np.sort(mutateGeneFrag)
                self.population[i, mutateGeneFrag[0] : mutateGeneFrag[1]] = \
                    self.population[i, mutateGeneFrag[1] - 1 : mutateGeneFrag[0] - 1 : -1] if mutateGeneFrag[0] - 1 >=0 \
                        else self.population[i, mutateGeneFrag[1] - 1 : : -1]

        newFitness = self.fitnessFunction(self.population)
        indices = np.tile((oldFitness >= newFitness).reshape(self.initialPopulationSize, 1), self.totalPoints)
        self.population = indices * currentGene + self.population * (~indices)

    def getBest(self) :
        '''
        获得当前族群中最优的个体
        '''

        standards = self.fitnessFunction(self.population)
    
        indices = np.argmax(standards)

        return standards[indices], self.population[indices]