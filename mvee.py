# By user unutbu in https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sin = np.sin
cos = np.cos

def mvee(points, tol = 0.001):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)        
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    return A, c

#some random points
points = np.array([[ 0.53135758, -0.25818091, -0.32382715], 
                   [ 0.58368177, -0.3286576,  -0.23854156,], 
                   [ 0.18741533,  0.03066228, -0.94294771], 
                   [ 0.65685862, -0.09220681, -0.60347573],
                   [ 0.63137604, -0.22978685, -0.27479238],
                   [ 0.59683195, -0.15111101, -0.40536606],
                   [ 0.68646128,  0.0046802,  -0.68407367],
                   [ 0.62311759,  0.0101013,  -0.75863324]])

# Singular matrix error!
# points = np.eye(3)

A, centroid = mvee(points)    
U, D, V = la.svd(A)    
rx, ry, rz = 1./np.sqrt(D)
u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

def ellipse(u,v):
    x = rx*cos(u)*cos(v)
    y = ry*sin(u)*cos(v)
    z = rz*sin(v)
    return x,y,z


E = np.dstack(ellipse(u,v))
E = np.dot(E,V) + centroid
x, y, z = np.rollaxis(E, axis = -1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.05)
ax.scatter(points[:,0],points[:,1],points[:,2])

plt.show()