import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib import colors

BW_CMAP = colors.ListedColormap(['#FFFFFF', '#666'])
FRAME_COLOR = '#aaa'
EXTENT_PARAM = 100
"""
This is an implementation by Thibault Marette and Stefan Neumann of the
ADVISER algorithm from the following paper:

Alessandro Colantonio, Roberto Di Pietro, Alberto Ocello, Nino Vincenzo Verde:
Visual Role Mining: A Picture Is Worth a Thousand Roles. IEEE Trans. Knowl. Data Eng. 24(6): 1120-1133 (2012)

Please do not share this code without our expressed permission.
"""


"""
	
	Parameters:
		A:
			the original matrix as numpy matrix
		leftClusters:
			the left/row clusters
		rightClusters:
			the right/column clusters
		plotUnorderedMatrix:
			if True, plots the original matrix A (without reordering)
		plotOrderedMatrix:
			if True, plots the original matrix A after reordering via the
			ADVISER algorithm
		plotClusterMatrix:
			if True, plots the low-rank cluster matrix after reordering via the
			ADVISER algorithm
		plotInRows:
			if True, plots the matrices in rows (i.e., below) each other. by
			default, the matrices are plotted in columns (side by side)
		transpose:
			if True, plots the transpose of the matrices
		outputfile:
			if not None, writes the plot to the file given by the string in
			outputfile (e.g., outputfile='~/Desktop/myGreatPlot.pdf')
		showPlot:
			if True, shows the plot using Python's GUI
"""


def plotClusteredMatrix(
        A,
        leftClusters,
        rightClusters,
        plotUnorderedMatrix=False,
        plotOrderedMatrix=True,
        plotClusterMatrix=False,
        plotInRows=False,
        transpose=False,
        outputfile=None,
        showPlot=False,
        aspect=1.0
):
    if transpose:
        plotBlock(
            A.transpose(),
            rightClusters,
            leftClusters,
            plotUnorderedMatrix=plotUnorderedMatrix,
            plotOrderedMatrix=plotOrderedMatrix,
            plotClusterMatrix=plotClusterMatrix,
            plotInRows=plotInRows,
            transpose=False,
            outputfile=outputfile,
            showPlot=showPlot,
        )
        return

    blockBlockMatrix = BlockBlockMatrix(leftClusters, rightClusters, A)

    numPlots = (
        int(plotUnorderedMatrix) + int(plotOrderedMatrix) + int(plotClusterMatrix)
    )

    if numPlots == 1:
        fig, ax = plt.subplots()
        axs = [ax]
    else:  # numPlots > 1
        if plotInRows:
            fig = plt.figure(figsize=(12, 12 * numPlots))
            gs = fig.add_gridspec(nrows=numPlots, ncols=1)
            axs = []
            for i in range(numPlots):
                ax = fig.add_subplot(gs[i, 0])
                axs.append(ax)
        else:
            base_size = 24
            fig = plt.figure(figsize=(base_size * numPlots / 10, base_size ))
            # fig = plt.figure(figsize=(12, 12 ))
            gs = fig.add_gridspec(nrows=1, ncols=numPlots)
            axs = []
            for i in range(numPlots):
                ax = fig.add_subplot(gs[0, i])
                axs.append(ax)

    if plotOrderedMatrix or plotClusterMatrix:
        blockBlockMatrix.order()

    if plotUnorderedMatrix:
        idx = 0
        cmap = BW_CMAP
        axs[idx].matshow(blockBlockMatrix.originalMatrix, extent=None, aspect=aspect, cmap=cmap)

    if plotOrderedMatrix:
        idx = int(plotUnorderedMatrix) + int(plotOrderedMatrix) - 1
        blockBlockMatrix.plotOriginalMatrix(axs=axs[idx], aspect=aspect)

    if plotClusterMatrix:
        idx = numPlots - 1
        blockBlockMatrix.plotMatrix(axs=axs[idx], aspect=aspect)

    fig.tight_layout()
    for ax in axs:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False,
        )
        plt.setp(ax.spines.values(), color=FRAME_COLOR)  # change frame color

    if outputfile != None:
        fig.savefig(outputfile, bbox_inches="tight")
    if showPlot:
        plt.show()
    plt.close("all")
    return fig


def createBlocks(clusters):
    indexToClusters = {}
    for i in range(len(clusters)):
        cluster = clusters[i]
        for index in cluster:
            if index not in indexToClusters:
                indexToClusters[index] = []
            indexToClusters[index].append(
                i
            )  # node index is in cluster i : {'node index' : ['cluster i']}

    blocks = {}
    allIndices = indexToClusters.keys()  # that are in a cluster

    for index in allIndices:
        blockName = str(indexToClusters[index])
        # print(blockName)
        if blockName not in blocks:
            blocks[blockName] = []
        blocks[blockName].append(
            index
        )  # {'['cluster i']' : ['node index']} : indexes that share the same clusters

    blocks = list(blocks.values())  # list of indexes that share the same cluster
    return blocks


class BlockBlockMatrix:
    def __init__(self, leftClusters, rightClusters, originalMatrix=[]):
        # original matrix
        self.originalMatrix = originalMatrix
        self.rows, self.cols = originalMatrix.shape
        # clusters
        self.leftClusters = []
        self.rightClusters = []
        for i in range(len(leftClusters)):
            if (leftClusters[i] != []) and (rightClusters[i] != []):
                self.leftClusters.append(leftClusters[i])
                self.rightClusters.append(rightClusters[i])
        # print(len(leftClusters), len(rightClusters))
        self.nbClusters = len(self.leftClusters)
        # blocks
        self.leftBlocks = createBlocks(self.leftClusters)
        self.rightBlocks = createBlocks(self.rightClusters)
        self.m = len(self.leftBlocks)
        self.n = len(self.rightBlocks)
        # block clusters
        self.leftClustersBlock = self.createBlockedClusters(
            self.leftClusters, self.leftBlocks
        )
        self.rightClustersBlock = self.createBlockedClusters(
            self.rightClusters, self.rightBlocks
        )
        # blocks permutation
        self.leftBlocksPermutation = [i for i in range(self.m)]
        self.rightBlocksPermutation = [i for i in range(self.n)]
        random.shuffle(self.leftBlocksPermutation)
        random.shuffle(self.rightBlocksPermutation)
        # block adjacency matrix
        self.C = (
            self.createBlockBlockAdj()
        )  # C[i,j] = 1 means that block i and j belong to the same cluster, and thus all indices in these block belong to the same cluster.

    # Creating the blockBlockAdjacency matrix.
    def createBlockBlockAdj(self):
        C = np.zeros([self.m, self.n])
        for i in range(self.m):
            for j in range(self.nbClusters):
                leftBlock = set(self.leftBlocks[i])
                leftCluster = set(self.leftClusters[j])
                if leftBlock.issubset(leftCluster):
                    # we know that block i is part of the column cluster j.
                    # Now, if we find a leftBlock k that is part of left cluster j, we know that they are related.
                    for k in range(self.n):
                        rightBlock = set(self.rightBlocks[k])
                        rightCluster = set(self.rightClusters[j])
                        if rightBlock.issubset(rightCluster):
                            C[i, k] = len(leftBlock) * len(rightBlock)
        return C

    ################################################################################################
    ########## Translating blockPermutations into a permutation for the original matrix ############
    ################################################################################################

    def getLeftPermutation(self):
        finalPermutation = []
        # print(self.leftBlocksPermutation)
        # print(self.leftBlocks)
        for blockIndex in self.leftBlocksPermutation:
            finalPermutation += self.leftBlocks[blockIndex]
        for node in range(self.rows):
            if node not in finalPermutation:
                finalPermutation.append(node)
        return finalPermutation

    def getRightPermutation(self):
        finalPermutation = []
        for blockIndex in self.rightBlocksPermutation:
            finalPermutation += self.rightBlocks[blockIndex]
        for node in range(self.cols):
            if node not in finalPermutation:
                finalPermutation.append(node)
        return finalPermutation

    #########################################################################################
    ############## Dispatching which ordering technique to what function. ###################
    #########################################################################################

    def order(self):
        self.adviser()

    #######################################################################
    ######################### BASELINE ALGORITHM ##########################
    #######################################################################

    def adviser(self):
        # left = rows = users
        # right = columns = perms
        # clusters = roles
        # UA all role-user associations

        users = self.leftBlocks
        perms = self.rightBlocks

        UA = []
        for b in range(len(users)):
            block = users[b]
            user = block[0]  # all users in the block share the same rank
            for r in range(len(self.leftClustersBlock)):
                c = self.leftClusters[r]  # c is cluster number r
                if user in c:
                    UA.append((b, block, r))

        PA = []
        for b in range(len(perms)):
            block = perms[b]
            user = block[0]  # all users in the block share the same rank
            for r in range(len(self.rightClustersBlock)):
                c = self.rightClusters[r]  # c is cluster number r
                if user in c:
                    PA.append((b, block, r))

        userOrder = self.sortSet(
            range(len(self.leftBlocks)),
            UA,
            PA,
            [i for i in range(len(self.leftClusters))],
        )
        permsOrder = self.sortSet(
            range(len(self.rightBlocks)),
            PA,
            UA,
            [i for i in range(len(self.rightClusters))],
        )
        self.leftBlocksPermutation = userOrder
        self.rightBlocksPermutation = permsOrder

    def sortSet(self, items, IA, IAbar, roles):
        # here items is itemsBar already, since the items have been grouped by block.
        sigma = []
        # we sort the blocks by the criterion in the paper.
        OGorder = items
        items = sorted(items, key=lambda x: -self.importanceScore(x, IA))

        for I in items:
            if len(sigma) < 2:
                sigma.append(I)
            else:
                if self.jacc(I, sigma[0], IA, IAbar) > self.jacc(
                    I, sigma[-1], IA, IAbar
                ):
                    p = 0
                    j = self.jacc(I, sigma[0], IA, IAbar)
                else:
                    p = len(sigma)
                    j = self.jacc(I, sigma[-1], IA, IAbar)
                for i in range(2, len(sigma)):
                    jprec = self.jacc(I, sigma[i - 1], IA, IAbar)
                    jsucc = self.jacc(I, sigma[i], IA, IAbar)
                    jcurr = self.jacc(sigma[i - 1], sigma[i], IA, IAbar)
                    if max(jprec, jsucc) > j and min(jprec, jsucc) >= jcurr:
                        p = i
                        j = max(jprec, jsucc)
                sigma.insert(p, I)
        return sigma

    def roleArea(self, r):
        return len(self.leftClusters[r]) * len(self.rightClusters[r])

    def importanceScore(self, item, IA):
        R = []
        for i, items, r in IA:
            if i == item:
                R.append(r)
        return sum([self.roleArea(r) for r in R])

    def jacc(self, i, j, IA, IAbar):
        # first we get the roles (clusters) of i and j
        ri, rj = [], []
        for item, items, role in IA:
            if item == i:
                ri.append(role)
            if item == j:
                rj.append(role)
        riUrj = []
        riNrj = []
        for r in ri:
            if r in rj:
                riNrj.append(r)
            riUrj.append(r)
        for r in rj:
            if r not in riUrj:
                riUrj.append(r)

        rU = 0
        rN = 0
        for u, items, r in IAbar:
            if r in riUrj:
                rU += len(items)
            if r in riNrj:
                rN += len(items)

        if rU == 0:
            return 0
        return rN / rU

    def plotMatrix(
            self, axs=None, showClusters=True, permutationRow=None, permutationCol=None,
            aspect=1.0, alpha=0.8
    ):
        if permutationRow == None:
            leftPermutation = self.getLeftPermutation()
        else:
            leftPermutation = permutationRow

        if permutationCol == None:
            rightPermutation = self.getRightPermutation()
        else:
            rightPermutation = permutationCol
        lowRankMatrix = np.zeros([self.rows, self.cols])

        orderedLeftBlocks = [
            [leftPermutation.index(element) for element in block]
            for block in self.leftBlocks
        ]
        orderedRightBlocks = [
            [rightPermutation.index(element) for element in block]
            for block in self.rightBlocks
        ]

        orderedLeftClusters = [
            [leftPermutation.index(element) for element in cluster]
            for cluster in self.leftClusters
        ]
        orderedRightClusters = [
            [rightPermutation.index(element) for element in cluster]
            for cluster in self.rightClusters
        ]

        if showClusters:
            for i in range(self.nbClusters):
                lcluster = orderedLeftClusters[i]
                rcluster = orderedRightClusters[i]
                lowRankMatrix[np.ix_(lcluster, rcluster)] = i + 1
        else:
            for i in range(self.m):
                for j in range(self.n):
                    if self.C[i, j] != 0:
                        leftBlock = orderedLeftBlocks[i]
                        rightBlock = orderedRightBlocks[j]
                        if list(self.leftBlocksPermutation).index(i) % 2 == 0:
                            lowRankMatrix[np.ix_(leftBlock, rightBlock)] = 1 + (
                                list(self.rightBlocksPermutation).index(j) % 2
                            )
                        else:
                            lowRankMatrix[np.ix_(leftBlock, rightBlock)] = 3 + (
                                list(self.rightBlocksPermutation).index(j) % 2
                            )

        orderedMatrix = lowRankMatrix

        if axs == None:
            return orderedMatrix
        else:
            # my_colors = ['#7F3C8D', '#11A579', '#3969AC', '#F2B701', '#E73F74', '#80BA5A', '#E68310', '#008695', '#CF1C90', '#f97b72', '#4b4b8f', '#A5AA99']  # bold
            my_colors = ['#E58606', '#5D69B1', '#52BCA3', '#99C945', '#CC61B0', '#24796C', '#DAA51B', '#2F8AC4', '#764E9F', '#ED645A', '#CC3A8E', '#A5AA99']  # vivid
            my_colors = ['#FFFFFF'] + my_colors
            cmap = colors.ListedColormap(my_colors)

            self.plotOriginalMatrix(axs=axs, aspect=aspect, alpha=0.9)  # we show ordered matrix first            
            # then add the rules
            axs.matshow(orderedMatrix, aspect=aspect, cmap=cmap, alpha=alpha)

            # axs.set_axis_off()

    def createBlockedClusters(self, clusters, blocks):
        blockedClusters = []
        for cluster in clusters:
            blockedCluster = []
            for block in blocks:
                if set(block).issubset(set(cluster)):
                    blockedCluster.append(blocks.index(block))
            blockedClusters.append(blockedCluster)
        return blockedClusters

    def plotOriginalMatrix(self, axs=None, permutationRow=None, permutationCol=None, alpha=1.0, aspect=1.0):
        mat = self.returnOriginalMatrix(permutationRow, permutationCol)

        if axs == None:
            print(mat)
        else:
            axs.matshow(mat, aspect=aspect, cmap=BW_CMAP, alpha=alpha)

    def returnOriginalMatrix(self, permutationRow=None, permutationCol=None):
        if permutationRow == None:
            leftPermut = self.getLeftPermutation()
        else:
            leftPermut = permutationRow
        if permutationCol == None:
            rightPermut = self.getRightPermutation()
        else:
            rightPermut = permutationCol
        mat = self.originalMatrix[leftPermut, :]
        mat[:] = mat[:, rightPermut]
        return mat
