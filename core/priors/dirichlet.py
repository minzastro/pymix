import copy
from pymix import _C_mixextend
import numpy
from core.distributions.discrete import DiscreteDistribution
from core.distributions.multinomial import MultinomialDistribution
from core.pymix_util.candidate_group import CandidateGroup
from core.pymix_util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from core.priors.prior import PriorDistribution


class DirichletPrior(PriorDistribution):  # DirichletDistribution,
    """
    Dirichlet distribution as Bayesian prior for MultinomialDistribution and derived .
    """

    def __init__(self, M, alpha):
        """
        @param M: number of dimensions
        @param alpha: distribution parameters
        """
        assert M == len(alpha)
        #for a in alpha:
        #    assert a > 0.0, "Invalid parameter."

        self.M = M
        self.alpha = numpy.array(alpha, dtype='Float64')
        self.alpha_sum = numpy.sum(alpha) # assumes alphas to remain constant !
        self.p = M
        self.suff_p = M
        self.freeParams = M

        self.constant_hyperparams = 1  # hyperparameters are constant

    def __copy__(self):
        cp_alpha = copy.deepcopy(self.alpha)
        return DirichletPrior(self.M, cp_alpha)

    def __str__(self):
        return "DirichletPrior: " + str(self.alpha)

    def __eq__(self, other):
        if isinstance(other, DirichletPrior):
            if self.M == other.M and numpy.alltrue(self.alpha == other.alpha):
                return True
            else:
                return False
        else:
            return False

    def sample(self):
        """
        Samples from Dirichlet distribution
        """
        phi = _C_mixextend.wrap_gsl_dirichlet_sample(self.alpha, self.M)

        d = DiscreteDistribution(self.M, phi)
        return d


    def pdf(self, m):

        # XXX should be unified ...
        if isinstance(m, MultinomialDistribution):
            # use GSL implementation
            #res = pygsl.rng.dirichlet_lnpdf(self.alpha,[phi])[0] XXX
            try:
                res = _C_mixextend.wrap_gsl_dirichlet_lnpdf(self.alpha, [m.phi])
            except ValueError:
                print m
                print self
                raise
            return res[0]

        elif isinstance(m, list):
            in_l = [d.phi for d in m]
            # use GSL implementation
            res = _C_mixextend.wrap_gsl_dirichlet_lnpdf(self.alpha, in_l)
            return res
        else:
            raise TypeError

    def posterior(self, m, x):
        """
        Returns the posterior for multinomial distribution 'm' for multinomial count data 'x'
        The posterior is again Dirichlet.
        """
        assert isinstance(m, MultinomialDistribution)
        res = numpy.ones(len(x), dtype='Float64')
        for i, d in enumerate(x):
            post_alpha = self.alpha + d
            res[i] = numpy.log(_C_mixextend.wrap_gsl_dirichlet_pdf(post_alpha, [m.phi]))

        return res

    def marginal(self, x):
        """
        Returns the log marginal likelihood of multinomial counts 'x' (sufficient statistics)
        with Dirichlet prior 'self' integrated over all parameterizations of the multinomial.
        """
        # XXX should be eventually replaced by more efficient implementation
        # in Dirchlet mixture prior paper (K. Sjoelander,Karplus,..., D.Haussler)

        x_sum = sum(x)

        term1 = _C_mixextend.wrap_gsl_sf_lngamma(self.alpha_sum) - _C_mixextend.wrap_gsl_sf_lngamma(self.alpha_sum + x_sum)
        term2 = 0.0
        for i in range(self.p):
            term2 += _C_mixextend.wrap_gsl_sf_lngamma(self.alpha[i] + x[i]) - _C_mixextend.wrap_gsl_sf_lngamma(self.alpha[i])

        res = term1 + term2
        return res

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        # Since DiscreteDistribution is a special case of MultinomialDistribution
        # the DirichletPrior applies to both. Therefore we have to distinguish the
        # two cases here. The cleaner alternative would be to derive specialized prior
        # distributions but that would be unnecessarily complicated at this point.
        if isinstance(dist, DiscreteDistribution):
            ind = numpy.where(dist.parFix == 0)[0]
            fix_phi = 1.0
            dsum = 0.0
            for i in range(dist.M):
                if dist.parFix[i] == 1:
                    fix_phi -= dist.phi[i]
                    continue
                else:
                    i_ind = numpy.where(data == i)[0]
                    est = numpy.sum(posterior[i_ind]) + self.alpha[i] - 1
                    dist.phi[i] = est
                    dsum += est

            # normalizing parameter estimates
            dist.phi[ind] = (dist.phi[ind] * fix_phi) / dsum
        elif isinstance(dist, MultinomialDistribution):

            fix_phi = 1.0
            dsum = 0.0
            # reestimating parameters
            for i in range(dist.M):
                if dist.parFix[i] == 1:
                    #print "111"
                    fix_phi -= dist.phi[i]
                    continue
                else:
                    est = numpy.dot(data[:, i], posterior) + self.alpha[i] - 1
                    dist.phi[i] = est
                    dsum += est

            if dsum == 0.0:
                raise InvalidPosteriorDistribution, "Invalid posterior in MStep."

            ind = numpy.where(dist.parFix == 0)[0]
            # normalzing parameter estimates
            dist.phi[ind] = (dist.phi[ind] * fix_phi) / dsum

        else:
            raise TypeError, 'Invalid input ' + str(dist.__class__)


    def mapMStepMerge(self, group_list):
        #XXX only for DiscreteDistribution for now, MultinomialDistribution to be done
        assert isinstance(group_list[0].dist, DiscreteDistribution), 'only for DiscreteDistribution for now'

        pool_req_stat = copy.copy(group_list[0].req_stat)
        pool_post_sum = group_list[0].post_sum
        pool_pi_sum = group_list[0].pi_sum

        for i in range(1, len(group_list)):
            pool_req_stat += group_list[i].req_stat
            pool_post_sum += group_list[i].post_sum
            pool_pi_sum += group_list[i].pi_sum

        new_dist = copy.copy(group_list[0].dist)  # XXX copy necessary ?

        ind = numpy.where(group_list[0].dist.parFix == 0)[0]
        fix_phi = 1.0
        dsum = 0.0
        for i in range(group_list[0].dist.M):
            if group_list[0].dist.parFix[i] == 1:
                assert group_list[1].dist.parFix[i] == 1  # incomplete consistency check of parFix (XXX)

                fix_phi -= new_dist.phi[i]
                continue
            else:
                est = pool_req_stat[i] + self.alpha[i] - 1
                new_dist.phi[i] = est

                dsum += est

        # normalizing parameter estimates
        new_dist.phi[ind] = (new_dist.phi[ind] * fix_phi) / dsum

        return CandidateGroup(new_dist, pool_post_sum, pool_pi_sum, pool_req_stat)


    def mapMStepSplit(self, toSplitFrom, toBeSplit):
        #XXX only for DiscreteDistribution for now, MultinomialDistribution to be done
        assert isinstance(toSplitFrom.dist, DiscreteDistribution), 'only for DiscreteDistribution for now'

        split_req_stat = copy.copy(toSplitFrom.req_stat)
        split_req_stat -= toBeSplit.req_stat

        split_post_sum = toSplitFrom.post_sum - toBeSplit.post_sum
        split_pi_sum = toSplitFrom.pi_sum - toBeSplit.pi_sum

        new_dist = copy.copy(toSplitFrom.dist)  # XXX copy necessary ?

        ind = numpy.where(toSplitFrom.dist.parFix == 0)[0]
        fix_phi = 1.0
        dsum = 0.0
        for i in range(toSplitFrom.dist.M):
            if toSplitFrom.dist.parFix[i] == 1:

                fix_phi -= new_dist.phi[i]
                continue
            else:
                est = split_req_stat[i] + self.alpha[i] - 1
                new_dist.phi[i] = est
                dsum += est

        # normalizing parameter estimates
        new_dist.phi[ind] = (new_dist.phi[ind] * fix_phi) / dsum

        return CandidateGroup(new_dist, split_post_sum, split_pi_sum, split_req_stat)


    def isValid(self, x):
        if not isinstance(x, MultinomialDistribution):
            raise InvalidDistributionInput, "in DirichletPrior: " + str(x)
        else:
            if self.M != x.M:
                raise InvalidDistributionInput, "in DirichletPrior: " + str(x)

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";DirichletPr;" + str(self.M) + ";" + str(self.alpha.tolist()) + "\n"



