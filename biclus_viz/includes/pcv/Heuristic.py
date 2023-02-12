import math
import numpy

import MajorityVote

def estimatePQ(ps, qs, B):
	bestP = 1
	bestQ = 0
	bestCluster = set()

	minDiff = numpy.Inf

	for p in ps:
		for q in qs:
			if q >= p:
				continue

			estP, estQ, cluster = estimatePQSingleCluster(p,q,B)

			diff = numpy.abs(p - estP) + numpy.abs(q - estQ)

			if diff < minDiff:
				minDiff = diff
				bestP = p
				bestQ = q
				bestCluster = cluster

#	print('bestP: {}, bestQ: {}, cluster: {}'.format(bestP, bestQ, cluster))
	return bestP, bestQ, bestCluster

def estimatePQSingleCluster(p,q,B):
	threshold = roundingThreshold(p,q)
	cluster = MajorityVote.BinaryMajorityVoteSingleCluster(B, threshold)

	totalOnes = B.sum()
	onesInCluster = B[:, cluster].sum()
	onesOutsideCluster = totalOnes - onesInCluster

	m, n = B.shape
	k = len(cluster)

	areaCluster = m*k
	areaOutsideCluster = m*(n-k)

	estP = 1
	estQ = 0
	if areaCluster > 0:
		estP = onesInCluster / areaCluster
	if areaOutsideCluster > 0:
		estQ = onesOutsideCluster / areaOutsideCluster

# print('estP: {}, estQ: {} we used p: {}, q: {}'.format(estP, estQ, p, q))

	return estP, estQ, cluster

def roundingThreshold(p,q):
	return math.log( (1-q)/(1-p), 2 ) / math.log(p*(1-q) / q / (1-p), 2);

