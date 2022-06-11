from logzero import logger
from copy import copy
from itertools import combinations
from scipy import sparse as sp

from .base import GreedyBase, RuleSampler
from ..samplers.base import Sampler
from ..samplers.discrim import CFTPDiscriminativitySampler
from ..samplers.uncovered_area import UncoveredAreaSampler
from ..exceptions import NoMoreSamples, InsufficientCandidates


class GreedyDivMax(GreedyBase, RuleSampler):
    def __init__(
        self,
        head_sampler: Sampler,
        tail_sampler: Sampler,
        lambd=0.1,
        n_tails_per_iter=100,
        n_heads_per_tail=10,
        n_max_rules=None,
        tolerance=1e-2,
    ):
        """
        an multi-label learning algorithm that greedily selects rules according to the diversity maximization principle

        params:

        head_sampler: a sampler for head (features)
        tail_sampler: a sampler for tails (labels)
        lambd: weight for the diversity term in the training objective

        tolerance: the minimum of threshold newly covered data records, which determines the stopping criterion
        n_max_rules: maximum number of rules to find (which has higher priority over using tolerance to determine stopping)

        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        GreedyBase.__init__(self)
        RuleSampler.__init__(self, head_sampler, tail_sampler)

    def should_stop(self, new_rule=None):
        # if n_max_rules is given, we use it
        if self.n_max_rules is not None:
            if self.num_rules_selected >= self.n_max_rules:
                logger.info(
                    f"collected enough rules {self.num_rules_selected}, terminate"
                )
                return True
            else:
                return False
        else:
            if new_rule is not None:
                mc = self.marginal_coverage_ratio(new_rule)
                if mc <= self.tolerance:
                    logger.info(
                        "margin coverage {} dropping below {}, terminate".format(
                            mc, self.tolerance
                        )
                    )
                    return True
                logger.debug(
                    "marginal coverage of it is {:.3%} (> {:.3%})".format(
                        mc, self.tolerance
                    )
                )

            return False

    # @profile
    def quality_of_rule(self, rule):
        # return rule.quality()
        c = rule.marginal_coverage(self.selected_rules)
        # c = rule.label_area()
        kl = rule.KL()
        return c * kl

    def diversity_of_rule(self, rule):
        return sum(rule.distance(other_rule) for other_rule in self.selected_rules)

    # @profile
    def objective_of_rule(self, rule):
        q = self.quality_of_rule(rule)
        d = self.diversity_of_rule(rule)
        return q + self.lambd * d

    def add_rule(self, rule):
        # update coverage
        self.tail_sampler.update_row_weights(rule.support, set(rule.tail))

        # add rule
        self.selected_rules.append(rule)

    def objective(self, rules):
        """get the objective achieved by the input rules"""
        quality = sum(self.quality_of_rule(r) for r in rules)
        diversity = sum(r1.distance(r2) for r1, r2 in combinations(rules, 2))
        return quality + self.lambd * diversity

    # @profile
    def fit_1st_round(self, X: sp.csr_matrix, Y: sp.csr_matrix):
        logger.info("running 1st round of fitting")

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
                "marginal coverage: {}, KL: {}".format(
                    best_candidate.marginal_coverage(self.selected_rules),
                    best_candidate.KL(),
                )
            )

            if self.should_stop(best_candidate):
                break

            self.add_rule(best_candidate)

        logger.info(
            "1st round done: {} rules selected and coverage ratio {:.2%}".format(
                self.num_rules_selected, self.coverage_ratio()
            )
        )
        self.obj_round_1 = self.objective(self.selected_rules)
        return copy(self.selected_rules)

    # @profile
    def fit_2nd_round(self):
        # TODO: filter out duplicate rules
        # TODO: remove selected rule from candidate set
        # TODO: make Rule support `set`
        logger.info("running 2nd round of fitting")
        self.selected_rules_round_1 = copy(self.selected_rules)
        self.selected_rules = []
        budget = len(self.selected_rules_round_1)

        if not hasattr(self, "candidate_rules"):
            raise RuntimeError(
                "candidate_rules not set, have you called fit_1st_round?"
            )

        if len(self.candidate_rules) == 0:
            raise InsufficientCandidates()

        for i in range(budget):
            best_candidate = self.select_best_candidate(self.candidate_rules)
            logger.debug(
                f"choosing rule: {best_candidate.summary()} among {len(self.candidate_rules)} candidates"
            )
            logger.debug(
                "marginal coverage: {}, KL: {}".format(
                    best_candidate.marginal_coverage(self.selected_rules),
                    best_candidate.KL(),
                )
            )
            self.selected_rules.append(best_candidate)

        logger.info(
            "2nd round done: {} rules selected and coverage ratio {:.2%}".format(
                self.num_rules_selected, self.coverage_ratio()
            )
        )
        self.obj_round_2 = self.objective(self.selected_rules)
        return copy(self.selected_rules)

    # @profile
    def fit(self, X, Y):
        rules_round_1 = self.fit_1st_round(X, Y)
        rules_round_2 = self.fit_2nd_round()

        self.obj = min(self.obj_round_1, self.obj_round_2)

        if self.obj_round_1 > self.obj_round_2:
            self.selected_rules = rules_round_1
            logger.info(
                "1st round fitting gives better objective: {} > {}".format(
                    self.obj_round_1, self.obj_round_2
                )
            )
        else:
            self.selected_rules = rules_round_2
            logger.info(
                "2nd round fitting gives better objective: {} >= {}".format(
                    self.obj_round_2, self.obj_round_1
                )
            )

class GreedyCFTPDivMax(GreedyDivMax):
    def __init__(
            self,
            min_feature_proba=0.8,
            min_label_proba=0.8,
            lambd=0.1,
            n_tails_per_iter=100,
            n_heads_per_tail=10,
            n_max_rules=None,
            tolerance=1e-2,
    ):
        """
        a wrapper class of GreedyDivMax

        it uses CFTPDiscriminativitySampler as head_sampler and UncoveredAreaSampler as tail_sampler
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        head_sampler = CFTPDiscriminativitySampler(min_proba=min_feature_proba)
        tail_sampler = UncoveredAreaSampler(min_proba=min_label_proba)
        GreedyBase.__init__(self)
        RuleSampler.__init__(self, head_sampler, tail_sampler)
