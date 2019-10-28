import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
import pint

# MTFlow data
Re_lref = np.array([20,30,40,50,60]) * 1000
C_T = np.array([[-0.82586, -0.69618, -0.62810, -0.58561, -0.55667],[-0.95486,-0.73617, -0.63740, -0.57832, -0.53903]]).transpose()

# Process
Re = Re_lref * 10
Cd = C_T * -1 * 1 ** 2 / (np.pi * 2.5**2) # First column Ncrit = 4, second column Ncrit = 9.

if __name__=="__main__":
    plt.figure()
    plt.plot(Re,Cd,'-o')
    plt.show()