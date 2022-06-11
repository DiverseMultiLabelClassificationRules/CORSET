import random
import numpy as np
from scipy import sparse as sp
from itertools import product
from collections import Counter
from .base import Sampler
from .assignment import TrieSampleAssignmentMixin, PRETTISampleAssignmentMixin
from .cftp import CFTP
from .pc import PCSampleSpaceConstructor
from dmrs.utils import convert_matrix_to_sets, convert_matrix_to_sets_v2
from dmrs.PRETTI.PRETTI_invertedIndex import InvertedIndex
from logzero import logger



class PruneGreedy(
        Sampler
        ):
    """
    sampling using the deterministic greedy algorithm preceeded by pruning and no CFTP 
    """

    def __init__(self, gamma=0.5, early_stop_epsilon = 0.1, pruning_epsilon = 0.001):
        self._positives = None
        self.gamma = gamma 
        self.early_stop_epsilon = early_stop_epsilon
        self.pruning_epsilon = pruning_epsilon
        self.rows = None
        self.InvInd = None 
        self.attribute_freq = None
        

        

    def _compute_pos_and_neg_info(self, y):
        self._positives, self._negagives = map(
            lambda arr: set(list(arr)), [(y > 0).nonzero()[0], (y == 0).nonzero()[0]]
        )
        
         
        self._pos_list = sorted(list(self._positives))
        self._neg_list = sorted(list(self._negagives))

    def _check_data(self, X, y):
        self.N, self.D = X.shape
        assert set(np.unique(y)) == {
            0,
            1,
        }, f"{np.unique(y)} contains elements other than {0, 1}"



    # @profile
    def fit(self, X: sp.csr_matrix, y: np.ndarray):
        """X: data records matrix
        y: a binary vector indicating the label of each row
        """

        # need the inverted index and attribute frequency for normalization in the greedy algorithm
        if self.rows == None: 
            self.rows = convert_matrix_to_sets_v2(X) 
        if self.InvInd == None: 
            self.InvInd = InvertedIndex(X).build_index()
        if self.attribute_freq == None: 
            self.attribute_freq = self.compute_frequencies() 

        self._check_data(X, y)
        self._compute_pos_and_neg_info(y)
        
        
    def prune_attributes(self): 
        res = set() 
        for attr in range(self.D): 
            if len(self.InvInd[attr].intersection(self._positives)) > 0:
                res.add(attr)
       
        return res 
    
    
    
    def sample_once(self):
        """if return_sample_index is True, return the sample index in the reduced sample space, instead of its content
        """
        res = self.prune_attributes() # Note that it is possible to optimize here by making sure we do the pruning only once 
        if len(res)==0: 
            res=set([i for i in range(self.D)])
   
        sample = self.greedy_choice(res) 
        return sample
          
        

    def sample(self, k):
        return [self.sample_once() for _ in range(k)]
    
    
    
    def greedy_choice(self, res): 
        
        ''' greedy algorithm to select a subset of attributes from a tuple of 
        positive and negative data records '''
        
        #print("res " + str(res)) 
        
        
        sampling_universe =  res # we sample from the difference here 
        chosen_attributes = [] 
        current_greedy_coverage = set() 
        discriminativities = [] 
        max_gain = float("-inf") 
        
        # select the first attribute 
        for attribute in sampling_universe: 
            gain = len( self.InvInd[attribute].intersection(self._positives))  - self.gamma * len(self.InvInd[attribute].intersection(self._negagives))
            if gain > max_gain: 
                best_attr = attribute 
                max_gain = gain 
            
        current_greedy_coverage.update(self.InvInd[best_attr]) 
        chosen_attributes.append(best_attr)
        sampling_universe.remove(best_attr) 
        discriminativities.append(max_gain)
        
        # select the other attributes 
        while len(sampling_universe) > 0:  
            max_gain = float("-inf") 
            for attribute in sampling_universe:                 
                gain =  len( current_greedy_coverage.intersection(self.InvInd[attribute]).intersection(self._positives)) - self.gamma * len(current_greedy_coverage.intersection(self.InvInd[attribute]).intersection(self._negagives))  
                if gain > max_gain: 
                    best_attr = attribute 
                    max_gain = gain 
                    
            
            
            current_greedy_coverage = current_greedy_coverage.intersection(self.InvInd[best_attr]) 
            
            if len(current_greedy_coverage) < self.early_stop_epsilon * len(self._positives):
                break 
            
            chosen_attributes.append(best_attr)
            sampling_universe.remove(best_attr) 
            discriminativities.append(max_gain)
        
        maximizer = np.argmax(discriminativities)
        attr_list = chosen_attributes[:(maximizer+1)]
                      
        return set(attr_list)
    
    
    
    def get_ground_truth_proba_for_pairs(self):
        """
        get the ground truth proba for each positive and negative DR pair

        do not run on large data sets since time complexity: O(N^2)
        """
        proba = {}
        total = 0
        for pos, neg in product(self._pos_list, self._neg_list):
            cnt = len(set(self.rows[pos]) - set(self.rows[neg]))
            proba[(pos, neg)] = cnt
            total += cnt

        for k in proba:
            proba[k] /= total

        return proba

    def get_ground_truth_proba_for_samples(self):
        """get the truth probability for each sample

        do not run on large data sets since time complexity: O(N^2)"""
        cnt = Counter()
        for pos, neg in product(self._pos_list, self._neg_list):
            for sample in set(self.rows[pos]) - set(self.rows[neg]):
                cnt[sample] += 1

        total = sum(cnt.values())

        proba = {}
        for k in cnt:
            proba[k] = cnt[k] / total

        return proba
    
    
    
    def compute_frequencies(self): 
        '''compute label and attributes and frequencies
        
        Returns
        ---------- 
            counter (dict) attribute freq        '''
        
        all_labels = []
        all_attributes = []
        for i in range(len(self.rows)): 
            list_attributes_bag = list(self.rows[i])
            all_attributes.extend(list_attributes_bag)
        
        return Counter(all_attributes) 
