################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################

from ..distributions.product import ProductDistribution
from ..util.errors import InvalidDistributionInput
from ..models.mixture import MixtureModel
from .prior import PriorDistribution
from ..util.dataset import DataSet


class ProductDistributionPrior(PriorDistribution):
    """
    Prior for ProductDistribution objects. Basically only holds a list of priors for
    atomar distributions. Necessary for model hierarchy.
    """

    def __init__(self, priorList):
        """
        Constructor

        @param priorList: list of PriorDistribution objects
        """
        self.priorList = priorList
        self.dist_nr = len(priorList)

    def __getitem__(self, ind):
        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError, 'Index ' + str(ind)
        else:
            return self.priorList[ind]

    def __setitem__(self, ind, item):
        assert isinstance(item, PriorDistribution)

        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError
        else:
            self.priorList[ind] = item

    def __eq__(self, other):
        if not isinstance(other, ProductDistributionPrior):
            return False
        if self.dist_nr != other.dist_nr:
            return False
        for i in range(self.dist_nr):
            if not self.priorList[i] == other.priorList[i]:
                return False
        return True

    def pdf(self, dist):
        assert isinstance(dist, ProductDistribution)
        res = 0
        for i in range(self.dist_nr):
            res += self.priorList[i].pdf(dist.distList[i])

        return res

    def marginal(self, dist, posterior, data):
        assert isinstance(dist, ProductDistribution)
        res = 0
        for i in range(self.dist_nr):
            res += self.priorList[i].marginal(dist.distList[i], posterior, data.getInternalFeature(i))

        return res

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        assert dist.suff_dataRange and dist.suff_p, "Attributes for sufficient statistics not initialized."
        assert isinstance(data, DataSet)
        assert isinstance(dist, ProductDistribution)
        assert dist.dist_nr == len(self.priorList)

        for i in range(dist.dist_nr):
            if isinstance(dist.distList[i], MixtureModel):
            # XXX use of isinstance() should be removed, singleFeatureSubset(i) to replace getInternalFeature(i)  ?
                self.priorList[i].mapMStep(dist.distList[i], posterior, data.singleFeatureSubset(i), mix_pi, dist_ind)
            else:
                self.priorList[i].mapMStep(dist.distList[i], posterior, data.getInternalFeature(i), mix_pi, dist_ind)


    def isValid(self, p):
        if not isinstance(p, ProductDistribution):
            raise InvalidDistributionInput, 'Not a ProductDistribution.'
        if p.dist_nr != self.dist_nr:
            raise InvalidDistributionInput, 'Different dimensions in ProductDistributionPrior and ProductDistribution: ' + str(p.dist_nr) + ' != ' + str(self.dist_nr)
        for j in range(p.dist_nr):
            try:
                self[j].isValid(p[j])
            except InvalidDistributionInput, ex:
                ex.message += "\n\tin ProductDistributionPrior.priorList[" + str(j) + "]"
                raise

