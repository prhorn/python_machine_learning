#! /usr/bin/env python

import numpy as np
import os

rootdir=os.getcwd()

DatapointEnergies=numpy.loadtxt(open(rootdir+'/SR_LSDAPW92VV10_0p3_QZVPPD_250974_Energies.csv'))
DatapointJacobian=numpy.loadtxt(open(rootdir+'/SR_LSDAPW92VV10_0p3_QZVPPD_250974_Jacobian.csv'),delimiter=",")
RefValues=numpy.loadtxt(open(rootdir+'/Reference_New.csv'))
WTrainDiagonal=numpy.loadtxt(open(rootdir+'/WTrainDiagonal_Attempt26.csv'))
WTotalDiagonal=numpy.loadtxt(open(rootdir+'/WTotalDiagonal_Attempt26.csv'))

