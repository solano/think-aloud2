# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:09:37 2022

Calculates various indices from the points in embedding space.

@author: Solano Felicio
"""

# %% Import modules

from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.stats.mstats import gmean
import python_tsp.heuristics

row_embeddings = np.load("../data/row_embeddings.npy")
probe_embeddings = np.load("../data/probe_embeddings.npy")
subrow_embeddings = np.load("../data/subrow_embeddings.npy")


# %% Define useful functions

# Cosine distance between vectors or arrays of vectors
dist = spatial.distance.cosine

# Compute distance matrix for a set of points
def dist_mat(points):
    p = spatial.distance.pdist(points, dist)
    return spatial.distance.squareform(p)

# Traveling salesman problem solver
solve_tsp = python_tsp.heuristics.solve_tsp_simulated_annealing

# Compute circuitousness of path
def circuitousness(path):
    # Add dummy node to ensure endpoints are respected
    # cf. https://stackoverflow.com/questions/14527815/how-to-fix-the-start-and-end-points-in-travelling-salesmen-problem
    N = len(path)
    dummy = 100*np.ones(path[-1:].shape)
    path = np.r_[path, dummy]
    
    # fix distance of dummy to endpoints as zero
    dmat = dist_mat(path)
    dmat[0,N] = 0
    dmat[N,0] = 0
    dmat[N-1,N] = 0
    dmat[N,N-1] = 0
    
    pathdist = sum(dmat[i,i+1] for i in range(0,N-1))
    
    # find optimal route
    optpath, optdist = solve_tsp(dmat)
    
    return pathdist/optdist

# Computes speed based on spatialtemporal trajectory
# vecs : array of vectors in embedding-space, size N > 1
# dt : array of time intervals, size N-1
# Returns array of size N-1 representing speed.
# If times are not specified, assumes Δt = 1
def trajectory_speed(vecs, dt=1):
    N = len(vecs)
    if N <= 1: return None
        
    dx = pd.array([dist(vecs[i], vecs[i+1]) for i in range(N-1)])
        
    return dx/dt

# By user unutbu in https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
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

# Find MVEE in the singular case, following Toubia
def singular_mvee(points, tol=1e-3, eps=1e-3):
    """Finds the minimum volume enclosing ellipsoid for a list of
    points whose length does not exceed the dimensionality of the space."""
    
    # Verify whether this is really needed:
    T = len(points)    # number of points
    d = len(points[0]) # dimensionality = 1024
    if(T > d):
        return mvee(points, tol)
    
    #print(f"MVEE is singular, {T} points")
    
    # Proceed according to Toubia
    
    # First, recenter around one of the points, arbitrarily chosen to be
    # the last one, and exclude it. Form a d x (T - 1) matrix
    Y = (points - points[-1])[:-1].T
    
    # Second, perform a singular value decomposition of this matrix
    U, Sigma, V = la.svd(Y)
    
    # Dimensionality reduction: choose first K columns of U as basis of
    # subspace (Toubia's choice K = T-1 gives bad convergence here)
    # Our choice: K = number of singular values greater than eps
    K = 0
    for svalue in Sigma:
        if svalue > eps:
            K += 1

    print(f"MVEE is singular, {T} points, {K} dimensions")
    S1 = U[:, :K]
    subspace_points = np.dot(S1.T, Y).T
    
    # Append the origin to the list of points and finally compute MVEE
    subspace_points = np.r_[subspace_points, np.zeros((1, K))]
    return mvee(subspace_points, tol)

# Volume as defined by Toubia: geometric mean of semiaxis lengths.
# Singular values of A = semiaxis lengths
def matvol(A):
    D = la.svd(A, compute_uv=False)
    return gmean(D)

def volume(points):
    A, c = singular_mvee(points)
    return matvol(A)



# df is either df_probes or df_rows or df_subrows.
# Calculates volume of all points for each subject
def voldf(df, points):
    subjects = df.suj.unique()
    nbpoints = []
    volumes = []
    for subj in subjects:
        v = volume(points[df.suj==subj])
        nbpoints.append((df.suj==subj).sum())
        volumes.append(v)
    data = np.array([subjects, nbpoints, volumes]).T
    return pd.DataFrame(data=data,
                        columns = ['suj','nbpoints','vol'])

# df is either df_probes or df_rows or df_subrows.
# Calculates circuitousness of each trajectory
# trajlevel should be "prob" for row and subrow analyses,
# and "bloc" for probe analysis
def circdf(df, points, trajlevel):
    if trajlevel=="prob":
        columns = ['suj', 'bloc', 'prob']
    elif trajlevel=="bloc":
        columns = ['suj', 'bloc']
    else:
        raise Exception("invalid trajlevel")

    trajs = df.groupby(columns, group_keys=False)
    
    c = []
    for key, trajindexes in trajs.groups.items():
        nbpoints = len(trajindexes)
        if nbpoints > 1:
            print(key)
            c.append((*key,                 # trajectory id
                  nbpoints,
                  circuitousness(points[trajindexes])))      # circuitousness
    
    columns.extend(['nbpoints', 'circ'])
    return pd.DataFrame(data=c,
                        columns=columns)

# Here each nonempty row of text is considered a different phrase
# to be embedded. "Speed" is defined as the quotient of distance
# between consecutive rows and the time elapsed between them.
# Note that we don't know the time between probes, so there is
# one trajectory per probe (thus, multiple trajectories per subject).

# For each probe, compute trajectory jump lengths, intervals and speeds
def probe_to_trajectory(probe):
    # Only nonempty rows
    indices = (probe.SPEECH.isna() == False)
    
    vecs = row_embeddings[probe.index][indices]
    
    # Choose time for each row as average of start and end times
    start_time = probe.start_time[indices].reset_index(drop=True)
    end_time = probe.end_time[indices].reset_index(drop=True)
    time = (start_time + end_time)/2
    
    interv = pd.array([time[i+1]-time[i] for i in range(len(time)-1)])
    pause = pd.array([start_time[i+1]-end_time[i] for i in range(len(time)-1)])
    
    jumps = trajectory_speed(vecs)
    speed = trajectory_speed(vecs, interv)
    
    if jumps is None:
        return None # null trajectory
    
    return list(zip(jumps, interv, pause, speed))

# Here each probe gives a phrase. There is no temporal data,
# so we compute only jump lengths between probes. Each block gives
# a trajectory in embedding space.

# For each block, compute trajectory jump lengths
def block_to_trajectory(block):
    # Only nonempty phrases
    indices = (block.SPEECH.isna() == False)
    vecs = probe_embeddings[block.index][indices]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps

# Here each row of original data gives multiple phrases. There is no
# temporal data, so we compute only jump lengths. Each probe gives
# a trajectory in embedding space.

# For each probe, compute trajectory jump lengths
def probe_sub_to_trajectory(probe):
    # Only nonempty phrases
    indices = (probe.SPEECH.isna() == False)
    vecs = subrow_embeddings[probe.index][indices]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps
