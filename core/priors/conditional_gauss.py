import math
from pymix import _C_mixextend
import numpy
from core.distributions.conditional_gauss import ConditionalGaussDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.priors.prior import PriorDistribution


class ConditionalGaussPrior(PriorDistribution):
    """
    Prior over ConditionalGaussDistribution. Assumes Normal prior over the covariance parameters w.

    """

    def __init__(self, nr_comps, p):
        """
            Constructor

            @param nr_comps: number of components in the mixture the prior is applied to
            @param p:  number of features in the ConditionalGaussDistribution the prior is applied to
        """

        self.constant_hyperparams = 0  # hyperparameters are updated as part of the mapEM
        self.nr_comps = nr_comps    # number of components in the mixture the prior is applied to
        self.p = p   # number of features in the ConditionalGaussDistribution the prior is applied to

        # no initial value needed, is updated as part of EM in updateHyperparameters
        self.beta = numpy.zeros((self.nr_comps, self.p))
        self.nu = numpy.zeros((self.nr_comps, self.p))

        # XXX initialization of sufficient statistics, necessary for hyperparameter updates
        self.post_sums = numpy.zeros(self.nr_comps)
        self.var = numpy.zeros((self.nr_comps, self.p))
        self.cov = numpy.zeros((self.nr_comps, self.p))
        self.mu = numpy.zeros((self.nr_comps, self.p))


    def __str__(self):
        return 'ConditionalGaussPrior(beta=' + str(self.beta) + ')'


    def pdf(self, d):
        if type(d) == list:
            N = numpy.sum(self.post_sums)

            res = numpy.zeros(len(d))
            for i in range(len(d)):
                for j in range(1, d[i].p):
                    pid = d[i].parents[j]
                    res[i] += (1.0 / self.cov[i, j] ** 2) / (self.nu[i, j] * (self.post_sums[i] / N))
                    res[i] += numpy.log(_C_mixextend.wrap_gsl_ran_gaussian_pdf(0.0,
                        math.sqrt((self.beta[i, j] * self.cov[i, j] ** 2) / (self.var[i, pid] * (self.post_sums[i] / N) )),
                        [d[i].w[j]]))
        else:
            raise TypeError, 'Invalid input ' + str(type(d))

        return res


    def posterior(self, m, x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self, x):
        raise NotImplementedError, "Needs implementation"


    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        assert not dist_ind == None # XXX debug

        post_sum = numpy.sum(posterior)
        self.post_sums[dist_ind] = post_sum

        # checking for valid posterior: if post_sum is zero, this component is invalid
        # for this data set
        if post_sum != 0.0:

            # reestimate mu
            for j in range(dist.p):
                # computing ML estimates for w and sigma
                self.mu[dist_ind, j] = numpy.dot(posterior, data[:, j]) / post_sum
                #self.var[dist_ind,j] = numpy.dot(posterior, (data[:,j] - dist.mu[j])**2 ) / post_sum
                self.var[dist_ind, j] = numpy.dot(posterior, (data[:, j] - self.mu[dist_ind, j]) ** 2) / post_sum

                if j > 0:  # w[0] = 0.0 is fixed
                    pid = dist.parents[j]
                    self.cov[dist_ind, j] = numpy.dot(posterior, (data[:, j] - self.mu[dist_ind, j]) * (data[:, pid] - self.mu[dist_ind, pid])) / post_sum

                    # update hyperparameters beta
                    self.beta[dist_ind, j] = post_sum / ( (( self.var[dist_ind, j] * self.var[dist_ind, pid]) / self.cov[dist_ind, j] ** 2) - 1 )

                    # update hyperparameters nu
                    self.nu[dist_ind, j] = - post_sum / (2 * dist.sigma[j] ** 2)

                    # update regression weights
                    dist.w[j] = self.cov[dist_ind, j] / (dist.sigma[pid] ** 2 * (1 + self.beta[dist_ind, j] ** -1 ) )

                    # update standard deviation
                    dist.sigma[j] = math.sqrt(self.var[dist_ind, j] - (dist.w[j] ** 2 * dist.sigma[pid] ** 2 * (1 + (1.0 / self.beta[dist_ind, j])) ) - self.nu[dist_ind, j] ** -1)
                    # update means
                    dist.mu[j] = self.mu[dist_ind, j] #- (dist.w[j] * self.mu[dist_ind,pid])

                else:
                    dist.sigma[j] = math.sqrt(self.var[dist_ind, j])  # root variance
                    dist.mu[j] = self.mu[dist_ind, j]


    def updateHyperparameters(self, dists, posterior, data):
        """
        Updates the hyperparamters in an empirical Bayes fashion as part of the EM parameter estimation.

        """
        assert len(dists) == posterior.shape[0]  # XXX debug

        # update component-specific hyperparameters
        for i in range(self.nr_comps):
            self.post_sums[i] = numpy.sum(posterior[i, :])
            for j in range(0, self.p):
                #  var_j = numpy.dot(posterior, (data[:,j] - dist.mu[j])**2 ) / post_sum
                self.var[i, j] = numpy.dot(posterior[i, :], (data[:, j] - dists[i].mu[j]) ** 2) / self.post_sums[i]

                if j > 0: # feature 0 is root by convention
                    pid_i_j = dists[i].parents[j]
                    self.cov[i, j] = numpy.dot(posterior[i, :], (data[:, j] - dists[i].mu[j]) * (data[:, pid_i_j] - dists[i].mu[pid_i_j])) / self.post_sums[i]
                    self.beta[i, j] = self.post_sums[i] / ( (( self.var[i, j] * self.var[i, pid_i_j]) / self.cov[i, j] ** 2) - 1 )
                    self.nu[i, j] = - self.post_sums[i] / (2 * dists[i].sigma[j] ** 2)


    def isValid(self, x):
        if not isinstance(x, ConditionalGaussDistribution):
            raise InvalidDistributionInput, "ConditionalGaussPrior: " + str(x)

