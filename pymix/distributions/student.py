# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:22:49 2015

@author: mints
"""

import numpy as np
from scipy.special import gamma, psi
from pymix.distributions.prob import ProbDistribution
from pymix.util.dataset import DataSet
import scipy.optimize as opt
from sage.utils import to_str


def xddot(par, matrix):
    z = np.dot(par, matrix)
    #print z.shape, par.shape, matrix.shape
    return np.sum(np.multiply(z, par))

def ddot(par, matrix):
    return np.sum(np.multiply(par, np.dot(par, matrix)), axis=1)

class MultivariateTDistribution(ProbDistribution):
    def __init__(self, p, mu, sigma, df):
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            p: dimension
            mu = mean (p dimensional numpy array or scalar)
            Sigma = scale matrix (pxp numpy array)
            df = degrees of freedom
        '''
        assert len(mu) == len(sigma) == len(sigma[0]) == p, str(len(mu))+ ' == ' + str(len(sigma)) + ' == ' + str(len(sigma[0])) + ' == '+ str(p)
        self.p = p
        self.suff_p = p
        self.df = df
        self.mu = np.array(mu, dtype='float')
        self.sigma = np.array(sigma, dtype='float')
        self.inv_sigma = np.linalg.inv(self.sigma)
        self.update_params()

    def update_params(self):
        self.dd = np.linalg.det(self.sigma);
        #self.ff = np.power(2*np.pi, -self.p*0.5)*np.power(self.dd,-0.5);
        self.freeParams = 2.*self.p + self.p**2
        self.factor = np.log(gamma((self.p+self.df)*0.5)) - \
                      np.log(gamma(self.df*0.5)) - \
                      0.5*self.p*np.log(self.df*np.pi) - \
                      0.5*np.log(self.dd)

    def __str__(self):
        return "Student's t:  [%s, %s, DoF: %s]" % (str(self.mu),
                                                    str(self.sigma.tolist()),
                                                    self.df)


    def pdf(self, data):
        x = self.data_numpy(data)
        result = self.factor - (self.p+self.df)*0.5*np.log(1. + ddot(x - self.mu, self.inv_sigma)/self.df)
        #import ipdb; ipdb.set_trace()
        #for i in xrange(len(x)):
        #    print x[i], result[i]
        return result

    def flatStr(self, offset):
        offset = "\t" * (offset + 1)
        return "%s;MultiNormal;%s;%s;%s;%s\n" % (offset, str(self.p),
                                                 str(self.mu.tolist()),
                                                 str(self.sigma.tolist()),
                                                 str(self.df))


    def MStep(self, posterior, data, mix_pi=None):
        x = self.data_numpy(data)
        post = posterior.sum() # sum of posteriors
        s = ddot(x-self.mu, self.inv_sigma)
        e1 = np.divide(self.df + self.p, self.df + s)
        e1post = posterior * e1
        e1post = e1post[:, np.newaxis]
        e1 = e1[:, np.newaxis]
        loge1 = psi(0.5*(self.df + self.p)) - np.log(0.5*(self.df + s))
        mu_1 = np.sum(x*e1post, axis=0)/np.sum(e1post, axis=0)
        tmp1 = x - self.mu
        tmp2 = tmp1 * e1post
        purr = tmp1[:, :, np.newaxis]*tmp2[:,np. newaxis, :]
        sigma_1 = np.sum(purr, axis=0)/post
        buzz = (np.sum(posterior*loge1) - np.sum(e1post, axis=0))/post
        #import ipdb; ipdb.set_trace()
        if np.any(buzz > -1.02):
            df_1 = self.df
        else:
            def fun1(t):
                return psi(0.5*t) - np.log(0.5*t) - 1 - buzz[0]
            df_1 = opt.brenth(fun1, 0.1, 50.)
        self.mu = mu_1
        self.sigma = sigma_1
        self.inv_sigma = np.linalg.inv(sigma_1)
        self.df = df_1
        self.update_params()
        #print self.mu, self.df, np.diag(self.sigma), buzz
