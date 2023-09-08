import numpy as np


def normal_dist(x:float, sd, mean:float=0):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density