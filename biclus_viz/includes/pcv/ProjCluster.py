import numpy
import random

from sklearn.cluster import KMeans

from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import svds

''' 
	The algorithm clusters the COLUMNS of the given matrix G.

	Types of clusterings:
		- kmeans -- uses k-means algorithm
'''
def ProjCluster(G, k, tau, projection='simple', gamma=0.5, useRandomProjection=False, clustering = 'kmeans'):
	rows, columns = G.shape

	H = numpy.zeros((1,1))
	P = Proj(G, k, proj=projection, gamma=gamma)
	H = numpy.dot(P,G)

	clusters = list()
	if clustering == 'kmeans':
		clusters = kmeans(H,tau)
	else:
		print('unknown clustering method')

	return clusters

def Proj(A, k, proj='simple',gamma=0.5):
	if proj == 'simple':
		return SimpleProj(A, k)
	else:
		print('Unsupported projection method: {0}'.format(proj))
		return

def SimpleProj(A, k):
	m,n = A.shape
	svdK = min(k,n,m)
	
	Aop = aslinearoperator(A)
	U,_,_ = svds(Aop,svdK,return_singular_vectors='u')
	return numpy.dot( U, U.T )

def kmeans(H,k):
	est = KMeans(n_clusters=int(k))
	est.fit(H.T)
	labels = est.labels_

	clusterDict = {}
	for i in range(len(labels)):
		cluster = labels[i]
		if cluster not in clusterDict:
			clusterDict[cluster] = [i]
		else:
			clusterDict[cluster].append(i)

	clusters = list(clusterDict.values())

	return clusters

