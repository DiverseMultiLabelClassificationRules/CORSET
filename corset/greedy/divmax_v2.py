from scipy import sparse as sp
from ..samplers.base import Sampler
from .diversity_maximization import GreedyDivMax
from .base import GreedyBase, RuleSampler
from ..exceptions import NoMoreSamples, InsufficientCandidates
from ..samplers.discrim import CFTPDiscriminativitySampler
from ..samplers.uncovered_area import UncoveredAreaSampler
from logzero import logger


class GreedyDivMaxV2(GreedyDivMax):
    def __init__(
        self,
        head_sampler: Sampler,
        tail_sampler: Sampler,
        lambd=0.1,
        n_tails_per_iter=10,
        n_heads_per_tail=1,
        n_max_rules=10,
    ):
        """
        compares to GreedyDivMax, this class differs by:

        - it only does the 1st round of fitting (for speed-up)
        - termination condition is determined by the number of rules being used ()
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)
        # logger.info("parameters: {}".format(kwargs))

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

    def fit_1st_round(self):
        raise RuntimeError('call fit instead')

    def fit_2nd_round(self):
        raise RuntimeError('2nd round of fitting is not available')

    # @profile
    def fit(self, X: sp.csr_matrix, Y: sp.csr_matrix):
        # TODO: mark that the model is fitted
        logger.info("fitting starts")
        self.set_X_Y(X, Y)

        self.tail_sampler.fit(Y)

        while True:
            try:
                candidate_rules = self.generate_candidate_rules()
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

            if self.should_stop():
                break

            self.add_rule(best_candidate)

        self.obj = self.objective(self.selected_rules)            
        logger.info(
            "fitting done: {} rules selected and coverage ratio {:.2%}".format(
                self.num_rules_selected, self.coverage_ratio()
            )
        )


class GreedyCFTPDivMaxV2(GreedyDivMaxV2):
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
        a wrapper class of GreedyDivMaxV2
        it uses CFTPDiscriminativitySampler as head_sampler and UncoveredAreaSampler as tail_sampler
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        head_sampler = CFTPDiscriminativitySampler(min_proba=min_feature_proba)
        tail_sampler = UncoveredAreaSampler(min_proba=min_label_proba)
        GreedyBase.__init__(self)
        RuleSampler.__init__(self, head_sampler, tail_sampler)
        
