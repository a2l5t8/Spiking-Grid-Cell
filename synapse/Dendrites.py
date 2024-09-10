import numpy as np
import pymoNNtorch
import CoNeX
import math
import random
import torch

"""
Dendrite Input Encoding and Computation.

Behaviors: 
    class GridDendriteInput()
    class GridDendriteInput2()
    class SynFun()
"""


class GridDendriteInput(Behavior) :

    def initialize(self, neurons) :
        """
        convert the given random walk into speed vector and input current of grid cells.

        Args:
            I_ext (float) : External Input Current which is a constant feedforward throughout the simulation.
            I_vel (float) : Velocity current scaler.
            a (float) : envelope function coefficient.
            walk (tuple) : Tuple of (x, y) of positions in the random walk of length the size of simulation.
            orientation (float) : degree of orientation of the grid cell plane.
            offset_x (float) : offset of the origin of the grid cell plane X-axis.
            offset_y (float) : offset of the origin of the grid cell plane Y-axis.
        """

        self.I_ext = self.parameter("I_ext", 2.4)
        self.I_vel = self.parameter("I_vel", 0.175)
        self.a = self.parameter("a", 4)
        self.walk = self.parameter("walk", None, required = True)

        self.orientation = self.parameter("orientation", 0)
        self.offset_x = self.parameter("offset_x", 0)
        self.offset_y = self.parameter("offset_y", 0)

        neurons.I = neurons.vector(mode = self.I_ext)

    def forward(self, neurons) :
        itr = neurons.network.iteration

        prev = [0, 0]
        if(itr != 0) : prev = [self.walk[0][itr - 1], self.walk[1][itr - 1]]

        curr = [self.walk[0][itr], self.walk[1][itr]]

        speed_vector = torch.Tensor([curr[0] - prev[0], curr[1] - prev[1]])
        speed_vector = self.rotate(speed_vector)

        neurons.I = (self.I_ext + self.I_vel * (speed_vector[0] * (neurons.x + self.offset_x) + speed_vector[1] * (neurons.y + self.offset_y))) * self.f(neurons)

    def f(self, neurons) :
        return torch.exp(-self.a * torch.sqrt(neurons.x**2 + neurons.y**2)/neurons.NeuronDimension.height)

    def rotate(self, speed_vector) : 
        phi = torch.tensor(self.orientation * math.pi / 180)
        s = torch.sin(phi)
        c = torch.cos(phi)
        rot = torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])])

        return speed_vector @ rot.t()


class GridDendriteInput2(Behavior) :

    def initialize(self, neurons) :
        """
        convert the given random walk into speed vector and input current of grid cells.

        Args:
            I_ext (float) : External Input Current which is a constant feedforward throughout the simulation.
            I_vel (float) : Velocity current scaler.
            a (float) : envelope function coefficient.
            walk (tuple) : Tuple of (x, y) of positions in the random walk of length the size of simulation.
            orientation (float) : degree of orientation of the grid cell plane.
            offset_x (float) : offset of the origin of the grid cell plane X-axis.
            offset_y (float) : offset of the origin of the grid cell plane Y-axis.
        """

        self.I_ext = self.parameter("I_ext", 1)
        self.I_vel = self.parameter("I_vel", 0.175)
        self.a = self.parameter("a", 4)
        self.walk = self.parameter("walk", None, required = True)

        self.orientation = self.parameter("orientation", 0)
        self.offset_x = self.parameter("offset_x", 0)
        self.offset_y = self.parameter("offset_y", 0)

        neurons.I = neurons.vector(mode = self.I_ext)

    def forward(self, neurons) :
        itr = neurons.network.iteration

        prev = [0, 0]
        if(itr != 0) : prev = [self.walk[0][itr - 1], self.walk[1][itr - 1]]

        curr = [self.walk[0][itr], self.walk[1][itr]]

        speed_vector = torch.Tensor([curr[0] - prev[0], curr[1] - prev[1]])

        prev_vec = self.vector_resize(copy.deepcopy(speed_vector))
        speed_vector = self.rotate(speed_vector)
        speed_vector = self.vector_resize(speed_vector)
        
        neurons.I = (neurons.vector(f"normal({self.I_ext}, {0})") + self.I_vel * (speed_vector[0] * (neurons.dir[:, 0]) + speed_vector[1] * (neurons.dir[:, 1]))) * self.envelope(neurons)
        # if(itr < 20) : 
        #     print(prev_vec)
        #     print(speed_vector)
        #     print(speed_vector)
        #     print((speed_vector[0] * (neurons.dir[2080, 0]) + speed_vector[1] * (neurons.dir[2080, 1])))

    def envelope(self, neurons) :
        return torch.exp(-self.a * torch.sqrt(neurons.x**2 + neurons.y**2)/neurons.NeuronDimension.height)

    def relu(self, x) :
        return torch.maximum(x, torch.Tensor([0]))

    def rotate(self, speed_vector) : 
        phi = torch.tensor(self.orientation * math.pi / 180)
        s = torch.sin(phi)
        c = torch.cos(phi)
        rot = torch.stack([torch.stack([c, -s]),
                        torch.stack([s, c])])

        return speed_vector @ rot.t()

    def vector_resize(self, vel) : 
        sz = np.sqrt(vel[0]**2 + vel[1]**2)
        return vel/sz


class SynFun(Behavior) : 

    def initialize(self, sg) : 
        # sg.W = sg.matrix(mode="normal(0.5, 0.3)")
        sg.I = sg.dst.vector()

    def forward(self, sg) :
        sg.I = torch.sum(sg.W[sg.src.spikes], axis = 0)