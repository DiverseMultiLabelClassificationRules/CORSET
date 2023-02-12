import numpy as np
from .divmax_v2 import GreedyDivMaxV2
from .base import GreedyBase, RuleSampler
from ..samplers.discrim import CFTPDiscriminativitySampler
from ..samplers.uncovered_area import UncoveredAreaSampler


class GreedyDivMaxV3(GreedyDivMaxV2):
    """compared to v2, the quality function is a bit different, sqrt(coverage) is used instead of coverage"""
    def quality_of_rule(self, rule):
        c = rule.marginal_coverage(self.selected_rules)
        kl = rule.KL()
        return np.sqrt(c) * kl


class GreedyCFTPDivMaxV3(GreedyDivMaxV3):
    def __init__(
            self,
            min_feature_proba=0.8,
            min_label_proba=0.8,
            lambd=0.1,
            n_tails_per_iter=10,
            n_heads_per_tail=1,
            n_max_rules=10,
    ):
        """
        a wrapper class of GreedyDivMaxV3
        it uses CFTPDiscriminativitySampler as head_sampler and UncoveredAreaSampler as tail_sampler
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        head_sampler = CFTPDiscriminativitySampler(min_proba=min_feature_proba)
        tail_sampler = UncoveredAreaSampler(min_proba=min_label_proba)
        GreedyBase.__init__(self)
        RuleSampler.__init__(self, head_sampler, tail_sampler)
        
    
