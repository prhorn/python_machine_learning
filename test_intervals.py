#! /usr/bin/env python

from machine_learning import *
import numpy as np

n = 6

x1 = np.random.randn(n)
x2 = np.random.randn(n)
x3 = np.random.randn(n)

y = 2.0*x1 + 5.5*x2 + np.random.randn(n)
y = y + 3.0

print 'x1, x2, x3, y'
print x1
print x2
print x3
print y

X = np.column_stack((x1,x2,x3))
B = linear_regression(X,y,True)
lin_reg_statistics(X,y,B)
