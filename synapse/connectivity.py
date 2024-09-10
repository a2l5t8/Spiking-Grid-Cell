import numpy as np
import pymoNNtorch
import CoNeX
import math
import random
import torch

"""
Connectivity Scheme of Lateral Inhibtion of Grid Cells on themselves 

Behaviors: 
    class GridWeightInitializer()
    class GridWeightInitializer2()
"""

class GridWeightInitializer(Behavior) :

    def initialize(self, synapse) :
        """
        Initialize the fixed synaptic weights in a grid cell plane based on a radius of connectivity.

        Args:
            R (int): Radius of connectivity and neighboring neurons in grid cell plane.
            w (float) : the constant synaptic weight in grid cell plane to each other.
            l (float) : Orientation prefence shift scaler.
        """

        self.R = self.parameter("R", required = True)
        self.w = self.parameter("w", required = True)
        self.l = self.parameter("l", required = True)

        synapse.W = synapse.matrix(mode = "zeros")

        """
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        What in the actual fuck !!!!!! [DOWN]
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        """

        a = []
        for i in range(synapse.W.shape[0]) : 
            a.append([])
            for j in range(synapse.W.shape[1]) : 
                a[i].append(0)

        X = []
        Y = []
        Dir = []
        for i in synapse.src.x : 
            X.append(i.item())
                    
        for i in synapse.src.y : 
            Y.append(i.item())
        
        for i in synapse.src.dir : 
            Dir.append([])
            for j in range(2) : 
                Dir[-1].append(i[j].item())

        MOD = (max(synapse.src.x).item() * 2 + 1)

        """
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        What in the actual fuck !!!!!! [UP]
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        """

        for S in tqdm.tqdm(range(len(a))) :
            for D in range(len(a)) :
                distance = math.sqrt((X[S] - X[D] - Dir[S][0])**2 + (Y[S] - Y[D] - Dir[S][1])**2)

                if(distance <= self.R or MOD - distance <= self.R) :
                    a[S][D] = self.w

        synapse.W = torch.Tensor(a)
        


    def dist(self, synapse, S, D) :
        """
        computes the euclidian distance between two 'S' and 'D' neurons in the grid cell plane.

        Args :
            synapse (SynapseGroup) : synapse group of the grid cell to itself.
            S (int) : index of the source neuron in grid cell neuron group.
            D (int) : index of the destination neuron in grid cell neuron group.
        """

        return (synapse.src.x[S] - synapse.dst.x[D] - synapse.src.dir[S][0])**2 + (synapse.src.y[S] - synapse.dst.y[D] - synapse.src.dir[S][1])**2

class GridWeightInitializer2(Behavior) :

    def initialize(self, synapse) :
        """
        Initialize the fixed synaptic weights in a grid cell plane based on a radius of connectivity.

        Args:
            R (int): Radius of connectivity and neighboring neurons in grid cell plane.
            w (float) : the constant synaptic weight in grid cell plane to each other.
            l (float) : Orientation prefence shift scaler.
        """

        self.R = self.parameter("R", required = True)
        self.l = self.parameter("l", required = True)
        self.a = self.parameter("a", 1)
        self.beta = self.parameter("beta", 0.2)
        self.gamma = self.parameter("gamma", 1.05)

        synapse.W = synapse.matrix(mode = "zeros")

        """
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        What in the actual fuck !!!!!! [DOWN]
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        """

        a = []
        for i in range(synapse.W.shape[0]) : 
            a.append([])
            for j in range(synapse.W.shape[1]) : 
                a[i].append(0)

        X = []
        Y = []
        Dir = []
        for i in synapse.src.x : 
            X.append(i.item())
                    
        for i in synapse.src.y : 
            Y.append(i.item())
        
        for i in synapse.src.dir : 
            Dir.append([])
            for j in range(2) : 
                Dir[-1].append(i[j].item())


        """
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        What in the actual fuck !!!!!! [UP]
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        """

        for S in tqdm.tqdm(range(len(a))) :
            for D in range(len(a)) :
                x = X[S] - X[D] - self.l * Dir[S][0]
                y = Y[S] - Y[D] - self.l * Dir[S][1]

                sz_squared = x*x + y*y
                metric = 3/(13*13)

                a[S][D] = self.a * np.exp(-self.gamma * metric * sz_squared) - np.exp(-self.beta * metric * sz_squared)


        synapse.W = torch.Tensor(a) * self.R

        
    def W0(self, x, y) :
        metric = 3/(13*13)
        
        return self.a * np.exp(-self.gamma * metric * sz_squared) - np.exp(-self.beta * metric * sz_squared)


    def dist(self, synapse, S, D) :
        """
        computes the euclidian distance between two 'S' and 'D' neurons in the grid cell plane.

        Args :
            synapse (SynapseGroup) : synapse group of the grid cell to itself.
            S (int) : index of the source neuron in grid cell neuron group.
            D (int) : index of the destination neuron in grid cell neuron group.
        """

        return (synapse.src.x[S] - synapse.dst.x[D] - synapse.src.dir[S][0])**2 + (synapse.src.y[S] - synapse.dst.y[D] - synapse.src.dir[S][1])**2