import pymoNNtorch
import numpy as np
import torch
from matplotlib import pyplot as plt

class Visualize() : 

    def __init__(self) : 
        pass


    @staticmethod
    def spike_range(ng, L, R) :
        """
        Returns the index of neurons which has spikes in interval of [L, R] as the number of simulation.

        Note:
            In order to use these helper methods you should record 'spikes' of the specific neuron group.

        Args: 
            ng (NeuronGroup) : The neuron group which we have recorded its spikes.
            L (int) : The left boundary of intreval of spikes.
            R (int) : The right boundary of interval of spikes.
        """
        return ng['spikes', 0][torch.logical_and(ng['spikes', 0][:, 0] >= L, ng['spikes', 0][:, 0] < R)][:,1]

    @staticmethod
    def plot_range(ng, L, R) :
        """
        Plots the results of of the `spike_range` method in a 2D Grid, same size as the grid-cells grid module.

        for Notes and Args see `spike_range` method.
        """
        plt.plot(pos_x[L:R], pos_y[L:R], '-')
        plt.plot(ng.x[spike_range(ng, L, R)] * 25/25, ng.y[spike_range(ng, L, R)] * 25/25, 'o', color = 'red')
        plt.show()

    @staticmethod
    def iter_spike(ng, itr, save = False, step = False, color = 'red', scale = 50, lib = "figs") :
        """
        Plots the neurons spike in each specific direction in a 2D grid, same size as the grid module in a specific iteration.

        Note:
            In order to use these helper methods you should record 'spikes' of the specific neuron group.

        Args: 
            ng (NeuronGroup) : The neuron group which we have recorded its spikes.
            itr (int): The iteration of spikes capturing.
            save (bool): If set to True, the plot would be saved in the specified directory `lib`. default is False.
            step (int): indicates the step size which we move through iterations. default is 1. (shows each iteration).
            color (color code) : Inidicates the color of the spikes in the plot.
            scale (int) : The scale size of each neuron compared to size of the grid module.
            lib (address) : The directory address in which you want the plot to be saved in.
        """

        plt.xlim(-25, +25)
        plt.ylim(-25, +25)

        prev = max(0, itr*step - 30)

        ng_idx = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step][:,1]

        
        E = []
        W = []
        N = []
        S = []
        for idx in ng_idx :
            i = idx//64
            j = idx%64

            if(i % 2 == 0 and j % 2 == 0) :
                N.append(idx.item())
            if(i % 2 == 0 and j % 2 == 1) :
                S.append(idx.item())
            if(i % 2 == 1 and j % 2 == 0) :
                E.append(idx.item())
            if(i % 2 == 1 and j % 2 == 1) :
                W.append(idx.item())

        pos_X = torch.Tensor(pos_x)
        pos_Y = torch.Tensor(pos_y)

        plt.plot(pos_X[prev:itr*step] * -1, pos_Y[prev:itr*step] * -1, color = 'black')
        
        plt.plot(ng.x[E] *(50 / (max(ng.x)*2 + 1)), ng.y[E] * (50 / (max(ng.x)*2 + 1)),'o',color = 'blue', markersize = (4 * scale/(ng.shape.width)) + 2)
        plt.plot(ng.x[W] *(50 / (max(ng.x)*2 + 1)), ng.y[W] * (50 / (max(ng.x)*2 + 1)),'o',color = 'red', markersize = (4 * scale/(ng.shape.width)) + 2)
        plt.plot(ng.x[N] *(50 / (max(ng.x)*2 + 1)), ng.y[N] * (50 / (max(ng.x)*2 + 1)),'o',color = 'green', markersize = (4 * scale/(ng.shape.width)) + 2)
        plt.plot(ng.x[S] *(50 / (max(ng.x)*2 + 1)), ng.y[S] * (50 / (max(ng.x)*2 + 1)),'o',color = 'yellow', markersize = (4 * scale/(ng.shape.width)) + 2)

        plt.legend(["path", "W", "E", "S", "N", "loc"], loc = 'upper right')


        plt.plot(pos_x[itr*step] * -1, pos_y[itr*step]* -1, '.', color = 'red')

        if(save) :
            plt.savefig("{}/fig{}.png".format(lib, itr))

        
    @staticmethod
    def iter_spike_multi_subplot(ng, itr, save = False, step = False, color = 'red', scale = 50, lib = "figs", label = "grid", offset_x = 0, offset_y = 0, base_offset_x = 0, base_offset_y = 0) :
        """
        Plots the neurons spike in a 2D grid, same size as the grid module in a specific iteration, suitable for multiple grid modules at once in different subplots.

        Note:
            In order to use these helper methods you should record 'spikes' of the specific neuron group.

        Args: 
            ng (NeuronGroup) : The neuron group which we have recorded its spikes.
            itr (int): The iteration of spikes capturing.
            save (bool): If set to True, the plot would be saved in the specified directory `lib`. default is False.
            step (int): indicates the step size which we move through iterations. default is 1. (shows each iteration).
            color (color code) : Inidicates the color of the spikes in the plot.
            scale (int) : The scale size of each neuron compared to size of the grid module.
            lib (address) : The directory address in which you want the plot to be saved in.
        """

        prev = max(0, itr*step - 30)

        ng_idx = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step][:,1]


        pos_X = torch.Tensor(pos_x)
        pos_Y = torch.Tensor(pos_y)

        plt.figure(figsize = (10, 3))

        plt.subplot(1, 2, 1)
        plt.title("Grid Module Spikes")
        plt.xlim(-25, +25)
        plt.ylim(-25, +25)
        plt.plot((ng.x[ng_idx] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, label = label)
        plt.legend(loc = 'upper right')

        
        plt.subplot(1, 2, 2)
        plt.title("Rat Movement")
        plt.xlim(-25, +25)
        plt.ylim(-25, +25)
        plt.plot((pos_X[:itr*step] + base_offset_x) * -1, (pos_Y[:itr*step] + base_offset_y) * -1, color = 'gray', alpha = 0.5)
        plt.plot((pos_X[prev:itr*step] + base_offset_x) * -1, (pos_Y[prev:itr*step] + base_offset_y) * -1, color = 'black')
        plt.plot((pos_x[itr*step] + base_offset_x) * -1, (pos_y[itr*step] + base_offset_y)* -1, '^', color = 'red', markersize = 10)
        
        plt.suptitle(f"iteration = {itr * step}", y = -0.01)

        if(save) :
            plt.savefig("{}/fig{}.png".format(lib, itr))

    
    @staticmethod
    def iter_spike_multi(ng, itr, save = False, step = False, color = 'red', scale = 50, lib = "figs", label = "grid", offset_x = 0, offset_y = 0, base_offset_x = 0, base_offset_y = 0) :
        """
        Plots the neurons spike in a 2D grid, same size as the grid module in a specific iteration, suitable for multiple grid modules at once in a single plot.

        Note:
            In order to use these helper methods you should record 'spikes' of the specific neuron group.

        Args: 
            ng (NeuronGroup) : The neuron group which we have recorded its spikes.
            itr (int): The iteration of spikes capturing.
            save (bool): If set to True, the plot would be saved in the specified directory `lib`. default is False.
            step (int): indicates the step size which we move through iterations. default is 1. (shows each iteration).
            color (color code) : Inidicates the color of the spikes in the plot.
            scale (int) : The scale size of each neuron compared to size of the grid module.
            lib (address) : The directory address in which you want the plot to be saved in.
        """

        plt.xlim(-25, +25)
        plt.ylim(-25, +25)

        prev = max(0, itr*step - 30)

        ng_idx = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step][:,1]
        ng_idx_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 4][:,1]
        ng_idx_prev_prev = ng["spikes", 0][ng["spikes", 0][:, 0] == itr * step - 8][:,1]

        pos_X = torch.Tensor(pos_x)
        pos_Y = torch.Tensor(pos_y)

        plt.plot((pos_X[prev:itr*step] + base_offset_x) * -1, (pos_Y[prev:itr*step] + base_offset_y) * -1, color = 'gray')
        
        plt.plot((ng.x[ng_idx_prev_prev] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx_prev_prev] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, alpha = 0.2)
        plt.plot((ng.x[ng_idx_prev] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx_prev] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, alpha = 0.35)
        plt.plot((ng.x[ng_idx] + offset_x) *(50 / (max(ng.x)*2 + 1)), (ng.y[ng_idx] + offset_y) * (50 / (max(ng.x)*2 + 1)),'o',color = color, markersize = (4 * scale/(ng.shape.width)) + 2, label = label)
        plt.legend(loc = 'upper right')

        plt.plot((pos_x[itr*step] + base_offset_x) * -1, (pos_y[itr*step] + base_offset_y)* -1, '^', color = 'red', markersize = 10)

        plt.title(f"iteration = {itr * step}", y = -0.2)
        plt.suptitle("Grid Module Spikes")

        if(save) :
            plt.savefig("{}/fig{}.png".format(lib, itr))

    
    @staticmethod
    def most_spike_plot(ng, threshold = 10, color = 'red', scale = 50) : 
        """
        Plots the neurons with spikes more than a certain threshold throughout the whole simulation.

        Note:
            In order to use these helper methods you should record 'spikes' of the specific neuron group.

        Args: 
            ng (NeuronGroup) : The neuron group which we have recorded its spikes.
            threshold (int) : The threshold for the number of spikes.
            color (color code) : Inidicates the color of the spikes in the plot.
            scale (int) : The scale size of each neuron compared to size of the grid module.
        """
        cnt = ng.vector('zeros')

        for spike in ng['spikes', 0] :
            cnt[spike[1]] += 1
        
        plt.plot(ng.x[cnt > threshold] *(50 / (max(ng.x)*2 + 1)), ng.y[cnt > threshold]*(50 / (max(ng.x)*2 + 1)),'o', color = color, markersize = (4 * scale/(ng.shape.width)) + 3)
        plt.xlim(-25, +25)
        plt.ylim(-25, +25)