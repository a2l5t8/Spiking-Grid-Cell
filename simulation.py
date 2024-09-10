import numpy as np
import pymoNNtorch
import CoNeX
import math
import random
import torch

from L6.neuron.neuron_model import *
from L6.synapse.connectivity import *
from L6.synapse.Dendrites import *

from helper.walk import *
from helper.visualize import Visualize

pos_x, pos_y = random_walk(1000, R = 10)

net = Network(behavior={
    1 : TimeResolution(dt = 1)
})

offset_x = torch.rand(6) * 0
offset_y = torch.rand(6) * 0

Rs = [120, 5,6, 6, 6, 3]
Bs = [1, 0.15, 0.3, 0.7 ,0.6, 0.3]
Gs = [1.05, 1.05, 1.05, 1.15, 1.05, 1.02]

ngs = []
for i in range(1) : 
    ng = NeuronGroup(size = NeuronDimension(width = 64, height = 64), net = net, behavior={
        2 : GridDendriteInput2(
            walk = (pos_x, pos_y),
            a = 2, 
            orientation = 0,
            offset_x = offset_x[i], 
            offset_y = offset_y[i],
            I_ext = 15,
            I_vel = 25
        ),
        3 : SynInp(),
        4 : GridCell(
            R = 10,
            tau = 5,
            threshold = -63,
            v_rest = -65,
            v_reset = -67,
            init_v = -65
        ),

        9 : Recorder(["I", "v"]),
        10 : EventRecorder(['spikes'])
    })

    ngs.append(ng)

    sg = SynapseGroup(net = net, src = ng, dst = ng, tag = 'GLUTAMATE', behavior={
        5 : GridWeightInitializer2(R = Rs[0], a = 1, beta = Bs[0], gamma = Gs[0], l = 2),
        6 : SynFun()
    })

net.initialize()
for i in range(1) : 
    net.simulate_iterations(1000)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = list(prop_cycle.by_key()['color'])

offset_x = [+2, +2.5, +2.25]
offset_y = [+2, -10.5, -10.25]
for i in range(150, 200) :
    cnt = 0
    for ng in ngs : 
        Visualize.iter_spike_multi(ng,
            itr = i, 
            step = 4, 
            color = colors[cnt], 
            save = False, 
            lib = "noise", 
            label = "grid" + str(cnt + 1), 
            offset_x = offset_x[cnt], 
            offset_y = offset_y[cnt],
            base_offset_x = 0,
            base_offset_y = 0
            )
        cnt += 1
        
    plt.show()