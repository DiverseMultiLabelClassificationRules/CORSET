#o!/usr/bin/python

import numpy
import PlotCreator

import sys
sys.path.insert(1,'includes/pcv/')
import PCV



inputfile = 'dialect.csv'
outputfile = 'plots/dialect.pdf'

# load the matrix
A = numpy.loadtxt(inputfile)

# compute the clusters of the matrix
k = 5
[leftClusters,rightClusters] = PCV.PCV(A,k,0.5,minClusterSize=1,useHeuristic=True)

# visualize the matrix
PlotCreator.plotClusteredMatrix(A,
				leftClusters,
				rightClusters, 
				plotUnorderedMatrix=True, 
				plotOrderedMatrix=True, 
				plotClusterMatrix=True, 
				plotInRows=True, 
				transpose=False, 
				outputfile=outputfile, 
				showPlot=False)

