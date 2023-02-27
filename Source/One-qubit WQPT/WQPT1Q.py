# This file contains functions for performing single qubit process tomography
# in the context of Wigner representation.

# # Wigner Quantum State Tomography for single qubit system

# $\textbf{Authors:}$ Amit Devra, Niklas Glaser, Dennis Huber, and Steffen J Glaser
#
# Code file for performing quantum state tomography for single qubit in the context of finite dimensional Wigner representation.
#
# Based on the study:

# %%
# import required packages
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint
import time
from fractions import Fraction
from math import gcd
from numpy import pi

from qiskit import IBMQ

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere, visualize_transition, plot_bloch_vector
from qiskit.circuit.library import *
from qiskit import (QuantumRegister, ClassicalRegister, QuantumCircuit,
    execute, Aer, compiler)
from qiskit.tools.monitor import job_monitor
from qiskit.tools import job_monitor, backend_monitor, backend_overview
from qiskit.quantum_info import Statevector
from qiskit.visualization import exceptions
from qiskit.visualization import latex as _latex
from qiskit.visualization import text as _text
from qiskit.visualization import utils
from qiskit.visualization import matplotlib as _matplotlib
import matplotlib as mpl
from matplotlib import cm, colors
import scipy
from scipy.special import sph_harm
# imports for plotting
import plotly.express as px
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %% Functions for analysis for single qubit process tomography result
# For generating basis we are using scipy.special.sph_harm function. Here, scipy.special.sph_harm(m,j) where m is the order and j is tha rank.
#
# The basis conersion i.e. from Pauli basis to Spherical tensor basis is provided in Appendix 4 of https://arxiv.org/abs/1409.5417#
#
# Creating a function for generating basis for single qubit. Input is number of theta (nth) and number of phi (nph) angle values.
# In[6]:
def single_basis(nth,nph):
    # polar angle
    theta = np.linspace(0, np.pi, nth)
    # azimuthal angle
    phi = np.linspace(0, 2*np.pi, nph)

    phi, theta = np.meshgrid(phi, theta)

# Generating the basis droplets for single qubit system
#     BId has scalar coeff is: 1/(2*np.sqrt(2)) which is arising from droplet f0_0. Check out Eqn. 54 of paper.
    BId = 0.5*(1/np.sqrt(2))*scipy.special.sph_harm(0, 0, phi, theta)

#     Bx and By has scalar coeff is: 0.5*(1/np.sqrt(2))*(1/np.sqrt(2)); the first part '0.5*(1/np.sqrt(2))' is
#     arising from droplet f1_1 (check out Eqn. 54 of paper).
#     Second part (1/np.sqrt(2)): conversion of complex to real harmonics.
    Bx = 0.5*(1/np.sqrt(2))*(1/np.sqrt(2))*(scipy.special.sph_harm(-1, 1, phi, theta)-scipy.special.sph_harm(1, 1, phi,theta))
    By = 0.5*(1/np.sqrt(2))*(1/np.sqrt(2))*1j*(scipy.special.sph_harm(-1, 1, phi, theta)+scipy.special.sph_harm(1, 1, phi,theta))

#    Bz has the same scalar coeff as for Bx and By; same reasoning. The last part 'np.sqrt(2)' is not clear.
#    But it should be here because to make radii of droplets same (as the conversion of complex to real requires divide by
#   sqrt(2) in Bx and By. So here to keep radii same we need multiply by sqrt(2)).
    Bz = 0.5*(1/np.sqrt(2))*(1/np.sqrt(2))*np.sqrt(2)*(scipy.special.sph_harm(0, 1, phi, theta))
    B0 = [BId,Bx,By,Bz]
    return B0
# %%
# New_Sampling: this function generates the weights for the equiangular sampling technique for calculating the overlap between two DROPS. This is based on Supplementary section 3 of the paper.
#
# Let $N$ is the number of points to sample in a sphere, and the angle increments by $d = \pi/N$. The phase or azimuthal angles ($\phi$) have 2$N$+1 equally spaced values between [0,2$\pi$] and similarly the polar angles ($\theta$) have $N$+1 equally spaced values between [0,$\pi$]. Let $\text{w}(l,\bar{l})$ be the sampling weight matrix of size (2$N$+1)$\times$($N$+1).

# Sampling points
# discretiziation of equiangular grids
def New_sampling(N):
    # discretization of grid
    d = np.pi/N
    # phase: distributed betwenn 0 and 2*pi
    phi = np.linspace(0, 2*np.pi, 2*N+1)
    # polar angle: distributed betwenn 0 and pi
    theta = np.linspace(0, np.pi, N+1)

    # ***samplings
    w = np.zeros((2*N+1,N+1))
    # for j=0 (theta=0, phi=0 i.e. north pole)
    w[0][0]=(1-np.cos(d/2))/(8*N)

    # for phi=0 and all theta values
    for k in range (1,N):
        theta_k=(k)*d
        w[0][k] = (np.cos(theta_k-(d/2))-np.cos(theta_k+(d/2)))/(8*N)

    # for k=N+1 or theta = pi (south pole)
    w[0][N] = (1-np.cos(d/2))/(8*N)

    # for j=2N+1
    w[2*N][0] = (1-np.cos(d/2))/(8*N)

    # for phi=2*pi and all theta values
    for k in range (1,N):
        theta_k=(k)*d
        w[2*N][k] = (np.cos(theta_k-(d/2))-np.cos(theta_k+(d/2)))/(8*N)

    # for k=N+1 or theta = pi (south pole)
    w[2*N][N] = (1-np.cos(d/2))/(8*N)

    # for remaining points
    for k in range (1,2*N):
        w[k][0] = (1-np.cos(d/2))/(4*N)

    for j in range (1,2*N):
        for k in range (1,N):
            theta_k=(k)*d
            w[j][k] = (np.cos(theta_k-(d/2))-np.cos(theta_k+(d/2)))/(4*N)

    for k in range (1,2*N):
        w[k][N] = (1-np.cos(d/2))/(4*N)

    return(w)
# %% DROPoverlap function: for calculation of overlap between the droplets. Based on Supplementary section 2 of paper.
# input droplets: f1, f2
# weights: W
def DROPoverlap(f1,f2,W):
    f1 = np.transpose(f1)
    f2 = np.transpose(f2)
    # **normalize droplet f1
    norm = W*np.conj(f1)*f1
    # sum over rows
    norm1 = np.sum(norm,axis=0)
    # summing over columns
    norm2 = np.sum(norm1)
    # normalizing f1
    f11 = f1/np.sqrt(norm2)
    #** calculation of overlap between f1 and f2
    c = W*np.conj(f11)*f2
    # sum over rows
    c1 = np.sum(c,axis=0)
    # sum over columns
    c2 = np.sum(c1)
    coeff = c2
    return(coeff)
# In[22]:
# For calculation of overlap with experimental droplets.
# required files: ideal basis, experimental droplets r values
def Experimental_process_analysis(basis,W,U_t,r0_expt,r1_expt):
    iB = basis
# calculating overlap of basis droplets
    coeff1=1j*np.zeros((4,4))
    for k in range(0,4):
        for j in range(0,4):
            coeff1[k][j] = DROPoverlap(iB[k],iB[j],W)


# overlap of experimental and simualted label l=0+l=1, and rank j=0+j=1 droplet
    coeff3 = 1j*np.zeros((1,4))
    for j in range (0,4):
        coeff3[0][j] = DROPoverlap(iB[j],r0_expt+r1_expt,W)/coeff1[j][j]

# define Pauli basis for reconstruction of density matrix
    PId = np.matrix([[1,0],
           [0,1]])
    Px = np.matrix([[0,1],
           [1,0]])
    Py = np.matrix([[0,-1j],
           [1j,0]])
    Pz = np.matrix([[1,0],
           [0,-1]])

# recreating process matrix
    U_r = coeff3[0][0]*PId+coeff3[0][1]*Px+coeff3[0][2]*Py+coeff3[0][3]*Pz
    U = np.matrix(U_r)
# Fidelity calculation
# U_t: target process matrix
    F = np.abs(np.trace(U*np.matrix(U_t).getH()))/2
    return(U,F)

# %%
# wrap function for angles used in plotting (plotly)
def wrap_angle(angles):
    wangle = np.empty(angles.shape)
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            wangle[i][j] = angles[i][j] - np.floor(angles[i][j]/(2*np.pi))*2*pi
    return(wangle)

# %%
# for plotting droplets
def Droplet_plot(res_theta,f0_0,f1_1,inter):
    # defining colormap here----
    colors = [ (0,a,0) for a in np.linspace(1,0,128)] + [(a,0,0) for a in np.linspace(0,1,128)]
    colors = [(0,1,0), (1,0,0)]

    v = np.exp(1.j * 2 * pi * np.linspace(0,1,255))
    x, y = v.real, v.imag
    r = np.power(np.maximum(0, np.minimum(1, ( x + y + 1) / 2)), 0.7)
    g = np.power(np.maximum(0, np.minimum(1, (-x + y + 1) / 2)), 0.7)
    yp = 0.195 * x + 0.981 * y;
    b = np.power(np.maximum( 0., -y) * np.minimum(1., 1.41 * np.maximum( 0. , -yp)), 0.7)
    # this can be directly used for normal matplotlib
    cmap = ListedColormap(list(zip(r,g,b)))

    # this is required for plotly
    cmap1 =list(zip(255*r,255*g,255*b))
    cmap2 = []
    for j in range(0,len(cmap1)):
        cmap2.append(px.colors.label_rgb(cmap1[j]))
    # colormap for plotly
    cmap3 = px.colors.validate_colors(cmap2,colortype='rgb')

    # label l={0}
    r0 = f0_0

    # label l={1}
    r1 = f1_1

    # theta and phi values
    res_phi = 2*res_theta-1
    thphi = []
    for theta in np.linspace(0,pi,res_theta):
        for phi in np.linspace(0,2*pi,res_phi):
            thphi.append([theta, phi])
    shape = (res_theta,res_phi)

    thphi_np = np.array(thphi)
    phi, theta = thphi_np[:,0].reshape(shape), thphi_np[:,1].reshape(shape)

    x = np.sin(phi)*np.cos(theta) * 1
    y = np.sin(phi)*np.sin(theta) * 1
    z = np.cos(phi) * 1

    #For matplotlib
    fig = plt.figure(figsize=(5,5))

    if inter==0:
        ax = fig.add_subplot(111, projection='3d')
        # Rank 0 droplet
        rank0 = ax.plot_surface(
            abs(r0) * x, abs(r0) * y, abs(r0) * z,  rstride=1, cstride=1, facecolors=cmap(np.mod(np.angle(r0),2*pi) / (2*pi)), alpha=1, linewidth=0)
        # Rank 1 droplet
        rank1 = ax.plot_surface(
            abs(r1) * x+0.3, abs(r1) * y, abs(r1) * z,  rstride=1, cstride=1, facecolors=cmap(np.mod(np.angle(r1),2*pi) / (2*pi)), alpha=1, linewidth=0)
        ax.set_xlim([-0.5,0.5])
        ax.set_ylim([-0.5,0.5])
        ax.set_zlim([-0.5,0.5])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.tight_layout()
        fig.show()

    # For plotly
    elif inter==1:
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, specs=[[{"type": "surface"},{"type": "surface"}]],
                                                        subplot_titles=('$f_{0}^{(\emptyset)}$', '$f_{1}^{(1)}$'))

        fig.add_trace(go.Surface(x=abs(r0)*x, y=abs(r0)*y, z=abs(r0)*z, surfacecolor = np.angle((r0)),
                                    opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 1,1)
        fig.add_trace(go.Surface(x=x*abs(r1), y=y*abs(r1), z=z*abs(r1), surfacecolor = wrap_angle(np.angle((r1))),
                                    opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 1,2)
        fig.show()

# %%
#Function for plotting expectation values
# rank 0 operator
# z_vals: j=0 (<x>+i<y>)
#rank 1 operator
# zz_vals: j=1 (<xz>+i<yz>)
def Expec_plot(z_vals,zz_vals):
    fig, axs = plt.subplots(2,2)
    # plt.figure()
    axs[0,0].plot(z_vals.real,label='real')
    axs[0,0].set_title('<x>')
    axs[0,0].set_ylim(-1,1)

    axs[0,1].plot(z_vals.imag, label = 'imag')
    axs[0,1].set_title('<y>')
    axs[0,1].set_ylim(-1,1)

    axs[1,0].plot(zz_vals.real,label='real')
    axs[1,0].set_title('<xz>')
    axs[1,0].set_ylim(-1,1)

    plt.plot(zz_vals.imag, label = 'imag')
    axs[1,1].set_title('<yz>')
    axs[1,1].set_ylim(-1,1)

    fig.tight_layout()
    for ax in axs.flat:
        ax.set(xlabel='n', ylabel='Expec')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return(plt.show())
# In[14]:
# Build the quantum circuits required for process tomography.
# All of the circuit elements are described in terms of a general single qubit rotation gate U3. More information about U3 is available here: https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html.

# ----Input----
# res_theta: number of theta (polar) angles for tomography
# Up: process/gate to tomograph.

# ----Output---
# circuit: circuits required for performing WQST for single qubit
def WQPT_1Q_circuits(res_theta,Up):
    # normal definition
    pi = np.pi

    # empty array for storing circuits
    circuits = []

    # Chosing resolution of theta and phi for the scanning purpose.
    res_phi = 2*res_theta-1
    N = res_theta*res_phi

    # **defining some rotations.
    # to_xgate takes the state from |0> to (|0>+|1>)*(1/sqrt(2))
    to_xgate = U3Gate(theta=-pi/2, phi=0, lam=0)

    # to_ygate takes the state from |0> to (|0>+i|1>)*(1/sqrt(2))
    to_ygate = U3Gate(theta=pi/2, phi=0, lam=pi/2)

    # Process or gate (U) to tomograph
    gate = Up

    # making the gate 'controlled Up(gate)'
    cU = gate.control()

    # **preparation of circuits**
    nnn = 0
    for theta in np.linspace(0,pi,res_theta):
        for phi in np.linspace(0,2*pi,res_phi):
            nnn+=1
            for i in range(4):
                circ_u = QuantumCircuit(2)

    #          hadamard for superposition state
                circ_u.h(0)

    #          x gate for mixed state
                if i % 2 == 1:
                    circ_u.x(1)
                circ_u.barrier()

                # controlled U
                circ_u.append(cU,[0,1])
                # controlled U
                circ_u.barrier()

                # gate for theta,phi rotation
                gate = U3Gate(theta=theta, phi=phi, lam=0).inverse()
                circ_u.append(gate, [1])
                circ_u.barrier()

                # rotation for measurement
                if (i < 2):
                    circ_u.append(to_xgate,[0])
                else:
                    circ_u.append(to_ygate,[0])

                circ_u.measure_all()
                # draw circuit for different values of theta and phi
                if nnn==1:
                    display(circ_u.draw())
                    # display(circ_u.draw(output='mpl'))
                circuits.append(circ_u)

    print("Total number of ciruits for WQPT for single qubit are:{}".format(len(circuits)))
    return(circuits)

# In[]
# Function for running quantum circuits
# ----Input----
# res_theta: resolution for polar angles (number)
# circuits: circuits for tomography (generated by WQST_1Q_circuits)
# device: simulator or experimental device to run on
# shots: number of shots for simulation/experiments
# inter: interactivity for plotting droplets (inter=0 gives non-interactive droplets)
# Ut: target unitary matrix. Required for estimating fidelity.

# ----Output----
# fidelity: experimental state fidelity
# density matrix: experimentally estimated density matrix
# plot of expectation values
# droplets (rank 0 and rank 1)
#
def WQPT_1Q_runner(res_theta,circuits,device,shots,inter,Ut):
    res_phi = 2*res_theta-1
    N = res_theta*res_phi

    # Defining chunks for storing circuits
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # giving a number of circuit you would like to run in a batch.
    # This ofcourse depends on the quantum device.
    circuit_chunks = list(chunks(circuits,300))

    jobs=[]
    # result of jobs are stored here
    jobs = [None]*len(circuit_chunks)

    # for executing the jobs
    for i, circuit_l in enumerate(circuit_chunks):
        jobs[i] = execute(circuit_l,device,shots=shots)

    #job monitor
    for i, job in enumerate(jobs):
        print(f'Job {i} of {len(jobs)}')
        display(job_monitor(job))

    # For counting zeros (0) and ones (1) from the data.
    cnt_list = []
    for job in jobs:
        cnt_list.extend(job.result().get_counts())

    # for storing expectation values
    # for j=0 (<x>+i<y>)
    z_vals = []
    # for j=1 (<xz>+i<yz>)
    zz_vals = []
    i=0
    # calculation of expectation values for different droplet operator described in paper's section 4B.
    for ii, cnts in enumerate(cnt_list):
        if ii % 4 == 0:
            val = [None] * 4
            val2 = [None] * 4

    #     linear terms (x and y)
    #  [v for k,v in cnts.items() if '0' in k[-1]]: this counts the number of 0 and 1 in a string (occurance of it)
        val[ii % 4] = (np.sum([v for k,v in cnts.items() if '0' in k[-1]]) / shots -
                        np.sum([v for k,v in cnts.items() if '1' in k[-1]]) / shots )

    #     bilinear terms (xz and yz)
        val2[ii % 4] = (np.sum([v for k,v in cnts.items() if ('00' in k or '11' in k)]) / shots -
                        np.sum([v for k,v in cnts.items() if ('01' in k or '10' in k)]) / shots )

        if ii % 4 == 3:
    # rank 0 operator
    # for j=0 (<x>+i<y>)
            z_vals.append(np.mean(val[0:2])+ 1.j*np.mean(val[2:4]))
    #rank 1 operator
    # for j=1 (<xz>+i<yz>)
            zz_vals.append(np.mean(val2[0:2])+ 1.j*np.mean(val2[2:4]))

    z_vals = np.array(z_vals)
    zz_vals = np.array(zz_vals)

    # for plotting the DROPS
    shape = (res_theta,res_phi)

    # combining expectation values to form tensor operators

    # rank-0 droplet function
    T00 = (np.sqrt((2 * 0 + 1) / (2 * pi))/(4))*(z_vals)
    # rank-1 droplet function
    T10 = (np.sqrt((2 * 1 + 1) / (2 * pi))/(4))*(zz_vals)

    # droplet functions
    f0_0 = np.reshape(T00,shape)
    f1_1 = np.reshape(T10,shape)

    # use Experimental_state_analysis(basis,W,U_t,r0_expt,r1_expt) for this analysis
    # basis: basis droplets generated using function single_basis(res_theta,res_phi)
    # W: sample weights for calculation of overlap
    # U_t: target process matrix
    # r0_expt: experimental droplet values of rank j=0
    # r1_expt: experimental droplet values of rank j=1
    r0_expt = f0_0
    r1_expt = f1_1

    L = Experimental_process_analysis(single_basis(res_theta,res_phi),New_sampling(res_theta-1),Ut,r0_expt,r1_expt)
    print('Process matrix is:',L[0])
    print('Process fidelity is:',L[1])
    return(Expec_plot(z_vals, zz_vals),Droplet_plot(res_theta,f0_0,f1_1,inter))
