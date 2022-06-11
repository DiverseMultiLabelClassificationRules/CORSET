import random
import numpy as np


class CFTP:
    """sampling using coupling from the past"""
    def __init__(
        self,
        W_pos_dict,
        W_neg_dict,
        pos_list,
        neg_list,
        data_records,
        max_iters=1024,
        random_state=None,
    ):
        """
        W_pos_dict: mapping from positive DR index to its weight
        W_neg_dict: mapping from negative DR index to its weight
        pos_list: list of positive DRs
        neg_list: list of negative DRs
        data_records: list of data records, each of which is a set of integers
        """
        kwargs = locals()
        del kwargs["self"]
        self.__dict__.update(**kwargs)

        self.pos_weights = np.array([self.W_pos_dict[p] for p in pos_list])
        self.neg_weights = np.array([self.W_neg_dict[p] for p in neg_list])
        self.pos_proba = self.pos_weights / self.pos_weights.sum()
        self.neg_proba = self.neg_weights / self.neg_weights.sum()

        self.reset_state()

    def add_data(self, n_iters, seed=None):
        """
        add data to the u_list, C_pos_list and C_neg_list such that their lengths are all n_iters
        """
        np.random.seed(seed)
        random.seed(seed)
        # extra samples we need to generate
        n_more_samples = n_iters - len(self.u_list)

        if n_iters <= 0:
            raise ValueError(f"{n_iters} <= {len(self.u_list)}")
        # sample uniforms
        u_arr = np.random.uniform(0, 1, n_more_samples)  # list of random numbers

        # prepend to previous samples
        self.u_list = list(u_arr) + self.u_list

        # sample C enumeratively using positive and negative weights
        # this is sampling WITH replacement
        # print("self.pos_proba: ", self.pos_proba)
        # print("self.neg_proba: ", self.neg_proba)
        
        
        C_pos_arr = np.random.choice(
            self.pos_list, size=n_more_samples, p=self.pos_proba
        )
        C_neg_arr = np.random.choice(
            self.neg_list, size=n_more_samples, p=self.neg_proba
        )

        # prepend to previous samples
        self.C_pos_list = list(C_pos_arr) + self.C_pos_list
        self.C_neg_list = list(C_neg_arr) + self.C_neg_list

        if (
            len(self.C_pos_list) != n_iters
            or len(self.C_neg_list) != n_iters
            or len(self.u_list) != n_iters
        ):
            raise ValueError(
                f"Wrong number of samples {len(self.C_pos_list)} != {n_iters}"
        )

    def reset_state(self):
        self.u_list = []  # list of random numbers from [0, 1]
        self.C_pos_list = []  # list of positive candidates generated
        self.C_neg_list = []  # list of positive candidates generated

    def simulate_backwards(self, return_history=True):
        """do backward simulation of the Markov chain

        if return_history is true, a list of dict is returned,
        dict keys are: pos, neg, D, W_C, W_D_bar, W_C_bar, W_D, u, and ratio is returned
        """
        W_D_bar, W_D = 1.0, 1.0
        D = None

        if return_history:
            hist = []

        for pos, neg, this_u in zip(self.C_pos_list, self.C_neg_list, self.u_list):
            W_C_bar = self.W_pos_dict[pos] * self.W_neg_dict[neg]

            dpos = set(self.data_records[pos])  # samples in pos DR
            dneg = set(self.data_records[neg])  # samples in neg DR

            W_C = len(dpos.difference(dneg))  # true weight of this pair

            ratio = (W_C * W_D_bar) / (W_C_bar * W_D)

            if return_history:
                hist.append(
                    dict(
                        pos=pos,
                        neg=neg,
                        u=this_u,
                        D=D,
                        W_C=W_C,
                        W_D_bar=W_D_bar,
                        W_C_bar=W_C_bar,
                        W_D=W_D,
                        ratio=ratio,
                    )
                )

            if this_u < ratio:
                # make a state transition
                D = (pos, neg)

                W_D, W_D_bar = W_C, W_C_bar

        if return_history:
            return D, hist
        else:
            return D

    def sample(self):
        i = 0
        # np.random.seed(self.random_state)
        D = None
        while D is None:
            n_iters = int(2**i)
            if n_iters > self.max_iters:
                print("reaching max number of iters")
                break
            
            # fix for the case there are no weights
            if self.pos_weights.sum() == 0 or self.neg_weights.sum() == 0: 
                return None 
            
            self.add_data(
                n_iters=n_iters,
                # seed=np.random.randint(999999)  # uncommenting this line seems to produce biased samples, not sure why
            )
            D = self.simulate_backwards(return_history=False)
            i += 1

        # print("self.C_pos_list: ", self.C_pos_list)
        # print("self.C_neg_list: ", self.C_neg_list)
        # print("self.u_list: ", self.u_list)

        self.reset_state()  # empty the lists
        return D

    def sample_k_times(self, n_samples):
        return [self.sample() for _ in range(n_samples)]
