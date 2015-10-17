#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from arima_utility import *

time_steps = 2000

print 'randome walk: ARIMA(0,1,0)'
plot_example_arima(0,1,0,time_steps)
