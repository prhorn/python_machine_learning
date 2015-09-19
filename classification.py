#! /usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from machine_learning import *
from toy_data import *

X,Y = classification_gaussians(5000,2,5,False,True)
X,Y = classification_gaussians(5000,2,5,True,True)

#mean = np.array([40,60])
#cov = np.array([100,30,30,140]).reshape(2,2)
#x,y = np.random.multivariate_normal(mean,cov,5000).T
#plt.plot(x,y,'x'); plt.axis('equal'); plt.show()




