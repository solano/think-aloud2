import pytest

import numpy as np
from numpy.linalg import qr
from scipy.stats import gmean
from think_aloud.compute_indices import volume

def sample_spherical(npoints, ndims):
    vec = np.random.randn(npoints, ndims)
    vec = (vec.T / np.linalg.norm(vec, axis=1)).T
    return vec

def test_vol():

    def test(ndims, npoints, tol=1):
        cloud = sample_spherical(npoints, ndims)
        semiaxes = np.abs(np.random.rand(ndims))
        #semiaxes = 0.8* np.ones(ndims)
        cloud = cloud * semiaxes
        vol = gmean(semiaxes)

        # Volume of MVEE should be less than the
        # volume calculated here, but on the same
        # order of magnitude
        if npoints > ndims:
            assert volume(cloud) <= vol
        assert abs(volume(cloud) - vol)/vol < tol
    
    
    for ndims in [20,30,40,150]:
        # deterministic test
        semiaxes = np.abs(np.random.rand(ndims))
        cloud = np.diag(semiaxes)
        cloud = np.vstack((cloud, -cloud))
        assert np.allclose(volume(cloud), gmean(semiaxes), 0.05)


        # random tests
        for npoints in range(ndims+10, ndims-10, -1):
            test(ndims, npoints)
        # In the limit with a lot of points, my test
        # should calculate the same volume as the MVEE
        for npoints in range(10*ndims, 10*ndims+1):
            test(ndims, npoints, tol=0.1)