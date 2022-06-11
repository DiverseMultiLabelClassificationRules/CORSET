from logzero import logger
from copy import copy
from itertools import combinations
from scipy import sparse as sp

from .base import GreedyBase, RuleSampler
from ..samplers.base import Sampler
from ..exceptions import NoMoreSamples, InsufficientCandidates


class GreedySubmod(GreedyBase, RuleSampler):
    def __init__(
        self,
        head_sampler: Sampler,
        tail_sampler: Sampler,
        lambd=0.1,
        n_tails_per_iter=10,
        n_heads_per_tail=1,
        n_max_rules=10
    ):
        """
        an multi-label learning algorithm that greedily selects rules according to the diversity maximization principle

        params:

        head_sampler: a sampler for head (features)
        tail_sampler: a sampler for tails (labels)
        lambd: weight for the diversity term in the training objective

        n_max_rules: maximum number of rules to find (which has higher priority over using tolerance to determine stopping)
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        GreedyBase.__init__(self)
        RuleSampler.__init__(self, head_sampler, tail_sampler)

    def should_stop(self):
        if self.num_rules_selected >= self.n_max_rules:
            logger.info(
                f"collected enough rules {self.num_rules_selected}, terminate"
            )
            return True
        else:
            return False

    def objective_of_rule(self, rule):
        mc = rule.marginal_coverage(self.selected_rules)
        mul = (1.0 if mc > 0 else 0.0)
        return mul * rule.KL() + self.lambd * mc

    def add_rule(self, rule):
        # update coverage
        self.tail_sampler.update_row_weights(rule.support, set(rule.tail))

        # add rule
        self.selected_rules.append(rule)

    # def objective(self, rules):
    #     """get the objective achieved by the input rules"""
    #     quality = sum(r.KL() for r in rules) 
    #     coverage = self.coverage_ratio() * self.total_num_label_occurences
    #     return quality + self.lambd * coverage

    def fit(self, X: sp.csr_matrix, Y: sp.csr_matrix):
        logger.info("start fitting")

        self.set_X_Y(X, Y)

        self.tail_sampler.fit(Y)

        self.candidate_rules = set()
        while True:
            try:
                candidate_rules = self.generate_candidate_rules()
                # save candidates for second round
                self.candidate_rules |= candidate_rules
            except NoMoreSamples:
                logger.info(
                    "no more samples can be generated, probably due to 100% label coverage\n"
                    "terminate"
                )
                break

            best_candidate = self.select_best_candidate(candidate_rules)
            logger.debug(
                f"choosing rule: {best_candidate.summary()} among {len(candidate_rules)} candidates"
            )
            logger.debug(
                "marginal coverage={:>5}, KL={:.2f}".format(
                    best_candidate.marginal_coverage(self.selected_rules),
                    best_candidate.KL(),
                )
            )

            if self.should_stop():
                break

            self.add_rule(best_candidate)

        logger.info(
            "fitting done: {} rules selected and coverage ratio {:.2%}".format(
                self.num_rules_selected, self.coverage_ratio()
            )
        )
