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

# %% Read data

# --- Experience data in original segmentation level ---
df_rows = pd.read_csv("text_rows.csv", sep="\t")

# Group row-level data into probes
probes = df_rows.groupby(['suj','bloc','prob'], group_keys=False)

# --- Experience data segmented at probe level ---

df_probes = pd.read_csv("text_probes.csv", sep="\t")

# Group probe-level data into blocks
blocks = df_probes.groupby(['suj','bloc'], group_keys=False)

# --- Experience data segmented at subrow level ---
df_subrows = pd.read_csv("text_subrows.csv", sep="\t")

# Group subrow-level data into probes
probes_sub = df_subrows.groupby(['suj','bloc','prob'], group_keys=False)

# --- Subject data ---
subj_data = pd.read_csv("info_participants.csv", sep="\t")

# Correct some entries which are lowercase, replace 'H' -> 'M', simplify names
subj_data.genre = subj_data.genre.str.upper()
subj_data.genre = subj_data.genre.apply(lambda s: 'M' if s=='H' else s)
subj_data = subj_data.rename(columns={"total-MEWS":"MEWS"})

# %% Load embeddings

row_embeddings = np.load("row_embeddings.npy")
probe_embeddings = np.load("probe_embeddings.npy")
subrow_embeddings = np.load("subrow_embeddings.npy")


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
def volume(A):
    D = la.svd(A, compute_uv=False)
    return gmean(D)

def volp(points):
    A, c = singular_mvee(points)
    return volume(A)

# Get inattention and impulsivity scores separately
ADHD_inatt = lambda df: df[[f"ADHD-{n}" for n in (1,2,3,4,7,8,9,10,11)]].astype('int').sum(1)
ADHD_impuls = lambda df: df[[f"ADHD-{n}" for n in (5,6,12,13,14,15,16,17,18)]].astype('int').sum(1)

# Prepare some dataframe for exportation
def prepare_export(indexdf, columns):
    df = indexdf.merge(subj_data, how='left', left_on='suj', right_on='sujet')
    
    df.insert(0, 'ADHD_inatt', ADHD_inatt(df))
    df.insert(0, 'ADHD_impuls', ADHD_impuls(df))
    df = df[columns]
    
    return df

# %% Analysis of volume

# df is either df_probes or df_rows or df_subrows.
# Calculates volume of all points for each subject
def voldf(df, points):
    subjects = df.suj.unique()
    nbpoints = []
    volumes = []
    for subj in subjects:
        v = volp(points[df.suj==subj])
        nbpoints.append((df.suj==subj).sum())
        volumes.append(v)
    data = np.array([subjects, nbpoints, volumes]).T
    return pd.DataFrame(data=data,
                        columns = ['suj','nbpoints','vol'])

row_vol = voldf(df_rows, row_embeddings)
probe_vol = voldf(df_probes, probe_embeddings)
subrow_vol = voldf(df_subrows, subrow_embeddings)

# %% Prepare and export volume data

row_voldf = prepare_export(row_vol, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'nbpoints','vol'])
subrow_voldf = prepare_export(subrow_vol, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'nbpoints','vol'])
probe_voldf = prepare_export(probe_vol, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'nbpoints','vol'])

row_voldf.to_csv("volume_rows.csv", sep="\t", index=False)
subrow_voldf.to_csv("volume_subrows.csv", sep="\t", index=False)
probe_voldf.to_csv("volume_probes.csv", sep="\t", index=False)

# %% Scatterplots of volume as function of ADHD
plt.scatter(row_voldf.ADHD, row_voldf.vol)
plt.title("Row volumes as function of ADHD")
plt.show()

plt.scatter(subrow_voldf.ADHD, subrow_voldf.vol)
plt.title("Subrow volumes as function of ADHD")
plt.show()

plt.scatter(probe_voldf.ADHD, probe_voldf.vol)
plt.title("Probe volumes as function of ADHD")
plt.show()

# %% Analysis of circuitousness

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

row_circ = circdf(df_rows, row_embeddings, "prob")
subrow_circ = circdf(df_subrows, subrow_embeddings, "prob")
probe_circ = circdf(df_probes, probe_embeddings, "bloc")

# %% Prepare and export circuitousness data

row_circdf = prepare_export(row_circ, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'bloc','prob','nbpoints','circ'])
subrow_circdf = prepare_export(subrow_circ, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'bloc','prob','nbpoints','circ'])
probe_circdf = prepare_export(probe_circ, ['suj','age','genre','exp','level','topic',
                       'ADHD','ADHD_inatt','ADHD_impuls','MEWS',
                       'bloc','nbpoints','circ'])

row_circdf.to_csv("circ_rows.csv", sep="\t", index=False)
subrow_circdf.to_csv("circ_subrows.csv", sep="\t", index=False)
probe_circdf.to_csv("circ_probes.csv", sep="\t", index=False)

# %% Analysis of transitions at row level

# Here each nonempty row of text is considered a different phrase
# to be embedded. "Speed" is defined as the quotient of distance
# between consecutive rows and the time elapsed between them.
# Note that we don't know the time between probes, so there is
# one trajectory per probe (thus, multiple trajectories per subject).

# For each probe, compute trajectory jump lengths, intervals and speeds
def probe_to_trajectory(probe):
    # Only nonempty rows
    indexes = (probe.SPEECH.isna() == False)
    
    vecs = row_embeddings[probe.index][indexes]
    
    # Choose time for each row as average of start and end times
    start_time = probe.start_time[indexes].reset_index(drop=True)
    end_time = probe.end_time[indexes].reset_index(drop=True)
    time = (start_time + end_time)/2
    
    interv = pd.array([time[i+1]-time[i] for i in range(len(time)-1)])
    pause = pd.array([start_time[i+1]-end_time[i] for i in range(len(time)-1)])
    
    jumps = trajectory_speed(vecs)
    speed = trajectory_speed(vecs, interv)
    
    if jumps is None:
        return None # null trajectory
    
    return list(zip(jumps, interv, pause, speed))

trajectories_rows = probes.apply(probe_to_trajectory)

# Delete null trajectories
trajectories_rows = trajectories_rows[trajectories_rows.isnull()==False]

#nb_trajectories_rows = len(trajectories_rows)
#print(f"There are {nb_trajectories_rows} non-null trajectories out of {nb_probes}")


# Separate transitions as units

transitions_rows = []

for index,traj in trajectories_rows.iteritems():
    for datatraj in traj:
        transitions_rows.append((*index, *datatraj))
        
transitions_rows = pd.DataFrame(data = transitions_rows,
                           columns=['suj','bloc','prob','length','interv',
                                    'pause','speed'])

#nb_transitions_rows = len(transitions_rows)
#print(f"There are {nb_transitions_rows} transitions at row level")

# %% Prepare and export row-level transitions for statistical analysis in R

df = prepare_export(transitions_rows, ['suj','bloc','prob',
                                       'length','interv','pause','speed',
         'age','genre','exp','level','topic',
         'ADHD','ADHD_inatt', 'ADHD_impuls','MEWS'])

df.to_csv("row_as_embedding_transitions.csv", sep='\t', index=False)

# %% Analysis of transitions at probe level

# Here each probe gives a phrase. There is no temporal data,
# so we compute only jump lengths between probes. Each block gives
# a trajectory in embedding space.

# For each block, compute trajectory jump lengths
def block_to_trajectory(block):
    # Only nonempty phrases
    indexes = (block.SPEECH.isna() == False)
    vecs = probe_embeddings[block.index][indexes]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps

trajectories_prob = blocks.apply(block_to_trajectory)

# Delete null trajectories
trajectories_prob = trajectories_prob[trajectories_prob.isnull()==False]

# Separate transitions as units

transitions_prob = []

for index,traj in trajectories_prob.iteritems():
    for jumplength in traj:
        transitions_prob.append((*index, jumplength))
        
transitions_prob = pd.DataFrame(data = transitions_prob,
                           columns=['suj','bloc','length'])

# %% Prepare and export probe-level transitions for statistical analysis in R

df2 = prepare_export(transitions_prob, ['suj','bloc','length',
         'age','genre','exp','level','topic',
         'ADHD','ADHD_inatt', 'ADHD_impuls','MEWS'])

df2.to_csv("probe_as_embedding_transitions.csv", sep='\t', index=False)

# %% Analysis of transitions at subrow level

# Here each row of original data gives multiple phrases. There is no
# temporal data, so we compute only jump lengths. Each probe gives
# a trajectory in embedding space.

# For each probe, compute trajectory jump lengths
def probe_sub_to_trajectory(probe):
    # Only nonempty phrases
    indexes = (probe.SPEECH.isna() == False)
    vecs = subrow_embeddings[probe.index][indexes]
        
    jumps = trajectory_speed(vecs)
    
    if jumps is None:
        return None # null trajectory
    
    return jumps

trajectories_sub = probes_sub.apply(probe_sub_to_trajectory)

# Delete null trajectories
trajectories_sub = trajectories_sub[trajectories_sub.isnull()==False]

# Separate transitions as units

transitions_sub = []

for index,traj in trajectories_sub.iteritems():
    for jumplength in traj:
        transitions_sub.append((*index, jumplength))
        
transitions_sub = pd.DataFrame(data = transitions_sub,
                           columns=['suj','bloc','prob','length'])

# %% Prepare and export subrow-level transitions for statistical analysis in R

df2 = prepare_export(transitions_sub, ['suj','bloc','length',
         'age','genre','exp','level','topic',
         'ADHD','ADHD_inatt', 'ADHD_impuls','MEWS'])

df3.to_csv("subrow_as_embedding_transitions.csv", sep='\t', index=False)