import copy
import math
import numpy
from pymix import _C_mixextend
from core.distributions.normal import NormalDistribution
from core.priors.prior import PriorDistribution
from core.pymix_util.candidate_group import CandidateGroup
from core.pymix_util.dataset import DataSet
from core.pymix_util.errors import InvalidPosteriorDistribution, InvalidDistributionInput


class NormalGammaPrior(PriorDistribution):
    """
    Inverse-Gamma Normal distribution prior for univariate Normal distribution.
    """

    def __init__(self, mu, kappa, dof, scale):
        """
        Constructor

        @param mu: hyper-parameter mu
        @param kappa: hyper-parameter kappa
        @param dof: hyper-parameter dof
        @param scale: hyper-parameter scale
        """
        # parameters of the Normal prior on mu | sigma
        self.mu_p = float(mu)
        self.kappa = float(kappa)

        # parameters on the inverse-gamma prior on sigma
        self.dof = float(dof)
        self.scale = float(scale)

        self.constant_hyperparams = 1  # hyperparameters are constant

    def __str__(self):
        outstr = "NormalGamma: mu_p=" + str(self.mu_p) + ", kappa=" + str(self.kappa) + ", dof=" + str(self.dof) + ", scale=" + str(self.scale)
        return outstr

    def __eq__(self, other):
        if not isinstance(other, NormalGammaPrior):
            return False
        if self.mu_p != other.mu_p:
            return False
        if self.kappa != other.kappa:
            return False
        if self.dof != other.dof:
            return False
        if self.scale != other.scale:
            return False
        return True

    def __copy__(self):
        return NormalGammaPrior(self.mu_p, self.kappa, self.dof, self.scale)


    def pdf(self, n):

        if isinstance(n, NormalDistribution):
            res = _C_mixextend.get_log_normal_inverse_gamma_prior_density(self.mu_p, self.kappa, self.dof, self.scale, [n.mu], [n.sigma])[0]
            return res

        elif isinstance(n, list):
            # extract parameters, XXX better way to do this ?
            d_sigma = numpy.zeros(len(n))
            d_mu = numpy.zeros(len(n))
            for i, d in enumerate(n):
                d_sigma[i] = d.sigma
                d_mu[i] = d.mu

            # call to extension function
            return _C_mixextend.get_log_normal_inverse_gamma_prior_density(self.mu_p, self.kappa, self.dof, self.scale, d_mu, d_sigma)
        else:
            raise TypeError

    def posterior(self, m, x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self, x):
        raise NotImplementedError, "Needs implementation"

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):

        assert isinstance(dist, NormalDistribution)
        # data has to be reshaped for parameter estimation
        if isinstance(data, DataSet):
            x = data.internalData[:, 0]
        elif isinstance(data, numpy.ndarray):
            x = data[:, 0]
        else:
            raise TypeError, "Unknown/Invalid input to MStep."
        nr = len(x)
        sh = x.shape
        assert sh == (nr,)

        post_sum = numpy.sum(posterior)  # n_k
        if post_sum == 0.0:
            print dist
            raise InvalidPosteriorDistribution, "Sum of posterior is zero."

        # computing ML estimates for mu and sigma
        ml_mu = numpy.dot(posterior, x) / post_sum  # ML estimator for mu
        new_mu = ( (post_sum * ml_mu) + (self.kappa * self.mu_p) ) / ( post_sum + self.kappa)

        n_sig_num_1 = self.scale + ( (self.kappa * post_sum) / ( self.scale + post_sum ) ) * (ml_mu - self.mu_p) ** 2
        n_sig_num_2 = numpy.dot(posterior, (x - ml_mu) ** 2)
        n_sig_num = n_sig_num_1 + n_sig_num_2
        n_sig_denom = self.dof + post_sum + 3.0

        new_sigma = math.sqrt(n_sig_num / n_sig_denom)

        # assigning updated parameter values
        dist.mu = new_mu
        dist.sigma = new_sigma


    def mapMStepMerge(self, group_list):

        pool_req_stat = copy.copy(group_list[0].req_stat)
        pool_post_sum = group_list[0].post_sum
        pool_pi_sum = group_list[0].pi_sum

        for i in range(1, len(group_list)):
            pool_req_stat += group_list[i].req_stat
            pool_post_sum += group_list[i].post_sum
            pool_pi_sum += group_list[i].pi_sum

        new_mu = (pool_req_stat[0] + (self.kappa * self.mu_p) ) / ( pool_post_sum + self.kappa)

        y = (pool_req_stat[0] ) / ( pool_post_sum)
        n_sig_num_1 = self.scale + ( (self.kappa * pool_post_sum) / ( self.scale + pool_post_sum ) ) * (y - self.mu_p) ** 2
        n_sig_num_2 = (pool_req_stat[1]) - 2 * y * (pool_req_stat[0]) + y ** 2 * pool_post_sum
        n_sig_num = n_sig_num_1 + n_sig_num_2
        n_sig_denom = self.dof + pool_post_sum + 3.0

        new_sigma = math.sqrt(n_sig_num / n_sig_denom)
        new_dist = NormalDistribution(new_mu, new_sigma)

        return CandidateGroup(new_dist, pool_post_sum, pool_pi_sum, pool_req_stat)

    def mapMStepSplit(self, toSplitFrom, toBeSplit):

        split_req_stat = copy.copy(toSplitFrom.req_stat)
        split_req_stat -= toBeSplit.req_stat

        split_post_sum = toSplitFrom.post_sum - toBeSplit.post_sum
        split_pi_sum = toSplitFrom.pi_sum - toBeSplit.pi_sum

        new_mu = (split_req_stat[0] + (self.kappa * self.mu_p) ) / ( split_post_sum + self.kappa)

        y = (split_req_stat[0] ) / ( split_post_sum)
        n_sig_num_1 = self.scale + ( (self.kappa * split_post_sum) / ( self.scale + split_post_sum ) ) * (y - self.mu_p) ** 2
        n_sig_num_2 = (split_req_stat[1]) - 2 * y * (split_req_stat[0]) + y ** 2 * split_post_sum
        n_sig_num = n_sig_num_1 + n_sig_num_2
        n_sig_denom = self.dof + split_post_sum + 3.0
        new_sigma = math.sqrt(n_sig_num / n_sig_denom)

        new_dist = NormalDistribution(new_mu, new_sigma)

        return CandidateGroup(new_dist, split_post_sum, split_pi_sum, split_req_stat)

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";NormalGamma;" + str(self.mu_p) + ";" + str(self.kappa) + ";" + str(self.dof) + ";" + str(self.scale) + "\n"

    def isValid(self, x):
        if not isinstance(x, NormalDistribution):
            raise InvalidDistributionInput, "NormalGammaPrior: " + str(x)

    def setParams(self, x, K):
        """
        Get guesses for hyper-parameters according to the heuristics used in "Bayesian Regularization for Normal
        Mixture Estimation and Model-Based Clustering" (C.Fraley and A.E. Raftery)

        @param x: numpy data vector
        @param K: number of components
        """
        nr = len(x)
        assert x.shape == (nr, 1)
        x = x[:, 0]

        self.mu_p = x.mean()
        self.kappa = 0.01
        self.dof = 3.0

        self.scale = x.var() / (K ** 2)

