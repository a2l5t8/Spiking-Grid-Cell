import math
import numpy as np
import random

"""
Generating a random walk for the simulation, starting at (0,0) in the plane.

Args : 
    WINDOW_WIDTH (int): The width of the plane of simulation. If it is set to `n` then the x-coordinates would be (-n/2, +n/2)
    WINDOW_HEIGHT (int): The height of the plane of simulation. If it is set to `n` then the y-coordinates would be (-n/2, +n/2)
    R (float): The radius of movement centering (0,0).add()
    length (int) : Number of iterations in the simulation.

Returns : 
    a tuple of two lists.

    pos_x (float) : a list of length `length` of x-coordinates at each iteration of simulation.
    pos_y (float) : a list of length `length` of y-coordinates at each iteration of simulation.
"""

WINDOW_WIDTH = 50
WINDOW_HEIGHT = 50

window_x = [-WINDOW_WIDTH/2, +WINDOW_WIDTH/2]
window_y = [-WINDOW_HEIGHT/2, +WINDOW_HEIGHT/2]

def conv(angel) : 
    x = np.cos(np.radians(angel))
    y = np.sin(np.radians(angel))

    return x, y

def random_walk(length, R = 20,  initialize = True) : 

    if(initialize) : 
        pos_x = [0]
        pos_y = [0]

    theta = 90
    cnt = 0
    length_cnt = 0

    for _ in range(length) : 

        dist = np.sqrt(pos_x[-1]**2 + pos_y[-1]**2)
        # print(dist)
        if(dist > R) : 
            ang = np.angle(complex(pos_x[-1], pos_y[-1]), deg = True)
            theta = ang + np.random.randint(90, 180) % 360

        pos_x.append(pos_x[-1] + (conv(theta)[0] + 6/5 * np.random.uniform(-0.5,0.5)) * 1/10)
        pos_y.append(pos_y[-1] + (conv(theta)[1] + 6/5 * np.random.uniform(-0.5,0.5)) * 1/10)
            
    return pos_x, pos_y
    