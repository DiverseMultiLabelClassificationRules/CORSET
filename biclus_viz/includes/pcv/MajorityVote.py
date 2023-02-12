import numpy

import Heuristic

def BinaryMajorityVote(B, transactionClusters, threshold=0.5, minClusterSize=1, useHeuristic=True):
	clusters = list()
	for transactionCluster in transactionClusters:
		if len(transactionCluster) < minClusterSize:
			continue

		Bsubmatrix = B[list(transactionCluster), :]
		if useHeuristic:
			ps = numpy.arange(0.3,1,0.1)
			qs = numpy.arange(0.02,0.11,0.02)
			_,_,cluster = Heuristic.estimatePQ(ps, qs, Bsubmatrix)
		else:
			cluster = BinaryMajorityVoteSingleCluster(Bsubmatrix, threshold)
		clusters.append(cluster)

	return clusters

def BinaryMajorityVoteSingleCluster(B, threshold):
	'''
	Returns a set of all column indices where the fractions of 1s in the column
	is larger than threshold.
	'''
	columnIndices = set()

	rows,cols = B.shape

	nonzeroCountPerCol = numpy.count_nonzero(B, axis=0)
	absThreshold = rows * threshold

	for col in range(cols):
		if nonzeroCountPerCol[col] > absThreshold:
			columnIndices.add(col)

	return list(columnIndices)

