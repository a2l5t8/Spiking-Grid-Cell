import numpy as np
import pymoNNtorch
import CoNeX
import math
import random
import torch

"""
Neuron Model of GridCell based on LIF Model as Behaviors.

Behaviors: 
    class GridCell()
    class SynInp()
"""

class GridCell(Behavior) :

    def initialize(self, neurons) :
        """
        Set neuron attributes. and adds Fire function as attribute to population.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        self.add_tag(self.__class__.__name__)

        neurons.R = self.parameter("R", None, required=True)
        neurons.tau = self.parameter("tau", None, required=True)
        neurons.threshold = self.parameter("threshold", None, required=True)
        neurons.v_reset = self.parameter("v_reset", None, required=True)
        neurons.v_rest = self.parameter("v_rest", None, required=True)

        neurons.v = self.parameter("init_v", neurons.vector())
        neurons.spikes = neurons.vector()

        neurons.spiking_neuron = self


        """
        Grid cell unique charactaristics.

        Arsgs :
            posX: x cordinate of each neuron in the 2D grid cell plane. [Implemented while using NeuronDimension]
            posY: y cordiante of each neuron in the 2D grid cell plane. [Implemented while using NeuronDimension]

            dir: unit vector of the desired direction of each neuron.
                [0, 1] -> N,
                [0, -1] -> S,
                [1, 0] -> E,
                [-1, 0] -> W
        """

        neurons.dir = torch.Tensor(neurons.size, 2)

        for i in range(neurons.NeuronDimension.height) :
            for j in range(neurons.NeuronDimension.width) :
                idx = i * neurons.NeuronDimension.height + j
                if(i % 2 == 0 and j % 2 == 0) :
                    neurons.dir[idx] = torch.Tensor([0, 1])
                if(i % 2 == 0 and j % 2 == 1) :
                    neurons.dir[idx] = torch.Tensor([0, -1])
                if(i % 2 == 1 and j % 2 == 0) :
                    neurons.dir[idx] = torch.Tensor([1, 0])
                if(i % 2 == 1 and j % 2 == 1) :
                    neurons.dir[idx] = torch.Tensor([-1, 0])

    def _RIu(self, neurons):
        """
        Part of neuron dynamic for voltage-dependent input resistance and internal currents.
        """
        return neurons.R * neurons.I

    def _Fu(self, neurons):
        """
        Leakage dynamic
        """
        return neurons.v_rest - neurons.v

    def Fire(self, neurons):
        """
        Basic firing behavior of spiking neurons:

        if v >= threshold then v = v_reset.
        """
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.v_reset

    def forward(self, neurons):
        """
        Single step of dynamics.

        Args:
            neurons (NeuronGroup): the neural population.
        """
        neurons.v += (
            (self._Fu(neurons) + self._RIu(neurons)) * neurons.network.dt / neurons.tau
        )

        self.Fire(neurons)



class SynInp(Behavior) : 
    
    def forward(self, ng) : 
        for syn in ng.afferent_synapses["All"] : 
            ng.I += syn.I