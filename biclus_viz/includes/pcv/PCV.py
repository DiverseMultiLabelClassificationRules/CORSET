import MajorityVote
import ProjCluster

'''
	Returns leftClusters and rightClusters which are lists of lists.
'''
def PCV(B, k, threshold, tau=None, projection="simple", gamma=0.5, clustering="kmeans", minClusterSize=10, useHeuristic=True):
	if tau == None:
		tau = k

	leftClusters = ProjCluster.ProjCluster(B.T, k, tau, projection=projection, gamma=gamma, clustering=clustering)
	rightClusters = MajorityVote.BinaryMajorityVote(B,leftClusters,threshold=threshold,minClusterSize=0,useHeuristic=useHeuristic)

	return leftClusters, rightClusters

