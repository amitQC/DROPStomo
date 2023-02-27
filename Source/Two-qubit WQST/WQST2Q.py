# In[]
# This file contains the functions used for Wigner state tomography of two qubit quantum states.
# $\textbf{Authors:}$ Amit Devra, Dennis Huber, Niklas Glaser, and Steffen J Glaser.

# In[]
# In[]
# importing required packages
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

# In[]
# # Functions for analysis and plotting of two qubit state tomography result.
# For generating basis we are using scipy.special.sph_harm function. Here, scipy.special.sph_harm(m,j) where m is the order and j is tha rank.
#
# The basis conversion i.e. from Pauli basis to Spherical tensor basis is provided in Appendix 4 of https://arxiv.org/abs/1409.5417.

# In[2]:
# Creating a function for generating basis for two qubit. Input is number of theta (nth) and number of phi (nph) angle values.

def two_basis(nth,nph):
    # polar angle
    theta = np.linspace(0, np.pi, nth)
    # azimuthal angle
    phi = np.linspace(0, 2*np.pi, nph)

    phi, theta = np.meshgrid(phi, theta)

    # Generating the basis droplets for two qubit system
    # The constant scaling factors in front: some part is taken from the main paper i.e. conversion of pauli in tensors.
    # all other factors are taking into account to have the same max value as the ideal droplet (from scanning approach).

    BId = 2*0.5*scipy.special.sph_harm(0, 0, phi, theta)

    # for qubit 1
    B1x = 2*0.25*np.sqrt(2)*(scipy.special.sph_harm(-1, 1, phi, theta)-scipy.special.sph_harm(1, 1, phi,theta))
    B1y = 2*0.25*np.sqrt(2)*1j*(scipy.special.sph_harm(-1, 1, phi, theta)+scipy.special.sph_harm(1, 1, phi,theta))
    B1z = 2*0.25*np.sqrt(2)*np.sqrt(2)*(scipy.special.sph_harm(0, 1, phi, theta))

    # for qubit 2
    B2x = 2*0.25*np.sqrt(2)*(scipy.special.sph_harm(-1, 1, phi, theta)-scipy.special.sph_harm(1, 1, phi,theta))
    B2y = 2*0.25*np.sqrt(2)*1j*(scipy.special.sph_harm(-1, 1, phi, theta)+scipy.special.sph_harm(1, 1, phi,theta))
    B2z = 2*0.25*np.sqrt(2)*np.sqrt(2)*(scipy.special.sph_harm(0, 1, phi, theta))

    # Bilinear terms
    B1x2x = 4*0.125*2*((1/np.sqrt(3))*scipy.special.sph_harm(0,0,phi,theta)+0.5*scipy.special.sph_harm(-2,2,phi,theta)-(1/np.sqrt(6))*scipy.special.sph_harm(0,2,phi,theta)+0.5*scipy.special.sph_harm(2,2,phi,theta))
    B1x2y = 4*0.125*2*((1/np.sqrt(2))*scipy.special.sph_harm(0,1,phi,theta)+0.5*1j*scipy.special.sph_harm(-2,2,phi,theta)-0.5*1j*scipy.special.sph_harm(2,2,phi,theta))
    B1x2z = 4*0.125*2*((-1j/2)*scipy.special.sph_harm(-1,1,phi,theta)-0.5*1j*scipy.special.sph_harm(1,1,phi,theta)+0.5*scipy.special.sph_harm(-1,2,phi,theta)-0.5*scipy.special.sph_harm(1,2,phi,theta))

    B1y2x = 4*0.125*2*(-(1/np.sqrt(2))*scipy.special.sph_harm(0,1,phi,theta)+0.5*1j*scipy.special.sph_harm(-2,2,phi,theta)-0.5*1j*scipy.special.sph_harm(2,2,phi,theta))
    B1y2y = 4*0.125*2*((1/np.sqrt(3))*scipy.special.sph_harm(0,0,phi,theta)-0.5*scipy.special.sph_harm(-2,2,phi,theta)-(1/np.sqrt(6))*scipy.special.sph_harm(0,2,phi,theta)-(0.5)*scipy.special.sph_harm(2,2,phi,theta))
    B1y2z = 4*0.125*2*((1/2)*scipy.special.sph_harm(-1,1,phi,theta)-0.5*scipy.special.sph_harm(1,1,phi,theta)+0.5*1j*scipy.special.sph_harm(-1,2,phi,theta)+0.5*1j*scipy.special.sph_harm(1,2,phi,theta))

    B1z2x = 4*0.125*2*((1j/2)*scipy.special.sph_harm(-1,1,phi,theta)+0.5*1j*scipy.special.sph_harm(1,1,phi,theta)+0.5*scipy.special.sph_harm(-1,2,phi,theta)-0.5*scipy.special.sph_harm(1,2,phi,theta))
    B1z2y = 4*0.125*2*((-1/2)*scipy.special.sph_harm(-1,1,phi,theta)+0.5*scipy.special.sph_harm(1,1,phi,theta)+0.5*1j*scipy.special.sph_harm(-1,2,phi,theta)+1j*0.5*scipy.special.sph_harm(1,2,phi,theta))
    B1z2z = 4*0.125*2*((1/np.sqrt(3))*scipy.special.sph_harm(0,0,phi,theta)+np.sqrt(2/3)*scipy.special.sph_harm(0,2,phi,theta))

    B0 = [BId,B1x,B1y,B1z,B2x,B2y,B2z,B1x2x,B1x2y,B1x2z,B1y2x,B1y2y,B1y2z,B1z2x,B1z2y,B1z2z]
    return B0

# In[3]:
# Sampling weights
# New_Sampling: this function generates the weights for the equiangular sampling technique for calculating the overlap between two DROPS. This is based on Supplementary section 3 of the paper.
# Let $N$ is the number of points to sample in a sphere, and the angle increments by $d = \pi/N$. The phase or azimuthal angles ($\phi$) have 2$N$+1 equally spaced values between [0,2$\pi$] and similarly the polar angles ($\theta$) have $N$+1 equally spaced values between [0,$\pi$]. Let $\text{w}(l,\bar{l})$ be the sampling weight matrix of size (2$N$+1)$\times$($N$+1).
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
# In[4]:
# Function for calculation of scalar product between two droplets
# DROPoverlap function: for calculation of overlap between the droplets. Based on Supplementary section 2 of paper.
# ----input----
# f1,f2: droplets
# W: sampling weights
# Returns the overlap coefficient of two droplets.
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
# In[5]:
# Defining Pauli basis which will be used later.
# Pauli basis for two qubits.
PId = np.matrix([[1,0],
   [0,1]])
Px = np.matrix([[0,1],
       [1,0]])
Py = np.matrix([[0,-1j],
       [1j,0]])
Pz = np.matrix([[1,0],
       [0,-1]])

# Define two qubit Pauli basis
Pii = np.kron(PId,PId)
Pxi = np.kron(Px,PId)
Pyi = np.kron(Py,PId)
Pzi = np.kron(Pz,PId)
Pix = np.kron(PId,Px)
Piy = np.kron(PId,Py)
Piz = np.kron(PId,Pz)
Pxx = np.kron(Px,Px)
Pxy = np.kron(Px,Py)
Pxz = np.kron(Px,Pz)
Pyx = np.kron(Py,Px)
Pyy = np.kron(Py,Py)
Pyz = np.kron(Py,Pz)
Pzx = np.kron(Pz,Px)
Pzy = np.kron(Pz,Py)
Pzz = np.kron(Pz,Pz)
P0 = [Pii,Pxi,Pyi,Pzi,Pix,Piy,Piz,Pxx,Pxy,Pxz,Pyx,Pyy,Pyz,Pzx,Pzy,Pzz]
# In[7]:
# required files: ideal basis, experimental droplets r values
def Experimental_state_analysis(basis,W,rhoT,r0_expt,r1_expt,r2_expt,r12_expt):
    iB = basis
    # calculating overlap of basis droplets for normalizing basis
    coeff1=1j*np.zeros((16,16))
    for k in range(0,16):
        for j in range(0,16):
            coeff1[k][j] = DROPoverlap(iB[k],iB[j],W)

#    for storing coefficients between ideal and experimental droplet
    coeff2 = 1j*np.zeros((1,16))

# overlap of experimental and simualted rank j=0 droplet
# here we are only taking overlap with label l= 0, and j=0 droplet of both experiments and simulated ideal droplet
# dividing by coeff1 to normalize

    coeff2[0][0] = DROPoverlap(iB[0],r0_expt,W)/coeff1[0][0]

#     overlap of first qubit (l=1)
    for j in range (1,4):
        coeff2[0][j] = DROPoverlap(iB[j],r1_expt,W)/coeff1[j][j]
#     overlap of second qubit (l=2)
    for j in range (4,7):
        coeff2[0][j] = DROPoverlap(iB[j],r2_expt,W)/coeff1[j][j]
#     overlap of all bilinear terms (l=12)
    for j in range (7,16):
        coeff2[0][j] = DROPoverlap(iB[j],r12_expt,W)/coeff1[j][j]

# reconstruction of density matrix
    rho_p = coeff2[0][0]*P0[0]
    for j in range (1,16):
        rho_p = coeff2[0][j]*P0[j]+rho_p
    rho = 0.5*rho_p

# Fidelity calculation
#   rhoT: target density matrix
    F = np.trace(np.matrix(rhoT).getH()*rho)/(np.sqrt(np.trace(np.matrix(rhoT).getH()*rhoT))*np.sqrt(np.trace(np.matrix(rho).getH()*rho)))
    return(F,rho)

# In[]
# %%
# wrap function for angles used in plotting (plotly)
def wrap_angle(angles):
    wangle = np.empty(angles.shape)
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            wangle[i][j] = angles[i][j] - np.floor(angles[i][j]/(2*np.pi))*2*pi
    return(wangle)

def Droplet_plot(res_theta,f0_0,f1_1,f1_2,f0_12,f1_12,f2_12,inter):
    # defining colormap here----
    colors = [ (0,a,0) for a in np.linspace(1,0,128)] + [(a,0,0) for a in np.linspace(0,1,128)]
    colors = [(0,1,0), (1,0,0)]

    v = np.exp(1.j * 2 * pi * np.linspace(0,1,255))
    x, y = v.real, v.imag
    r = np.power(np.maximum(0, np.minimum(1, ( x + y + 1) / 2)), 0.7)
    g = np.power(np.maximum(0, np.minimum(1, (-x + y + 1) / 2)), 0.7)
    yp = 0.195 * x + 0.981 * y
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

    # Droplet values
    # label l={1}
    r1 = f1_1
    # label l={2}
    r2 = f1_2
    # label l={0}
    r0 = f0_0
    # label l={12}
    r12 = f0_12+f1_12+f2_12

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
        #Set colours and render
        ax = fig.add_subplot(111, projection='3d')
        # The droplets of different components are plotted in one figure
        ax.plot_surface(
            abs(r0) * x+1, abs(r0) * y, abs(r0) * z+2,  rstride=1, cstride=1, facecolors=cmap(np.angle(r0) / (2*pi)), alpha=1, linewidth=0)
        ax.plot_surface(
            abs(r1) * x+1, abs(r1) * y, abs(r1) * z+0.5,  rstride=1, cstride=1, facecolors=cmap(np.angle(r1) / (2*pi)), alpha=1, linewidth=0)
        ax.plot_surface(
            abs(r2) * x+1, abs(r2) * y, abs(r2) * z-0.5,  rstride=1, cstride=1, facecolors=cmap(np.angle(r2) / (2*pi)), alpha=1, linewidth=0)
        # right now, I am plotting all the bilinear terms.
        ax.plot_surface(
            abs(f0_12) * x+1, abs(f0_12) * y, abs(f0_12) * z-1.5,  rstride=1, cstride=1, facecolors=cmap(np.angle(f0_12) / (2*pi)), alpha=1, linewidth=0)
        ax.plot_surface(
            abs(f1_12) * x, abs(f1_12) * y, abs(f1_12) * z-0.5,  rstride=1, cstride=1, facecolors=cmap(np.angle(f1_12) / (2*pi)), alpha=1, linewidth=0)
        ax.plot_surface(
            abs(f2_12) * x-1, abs(f2_12) * y, abs(f2_12) * z+0.5,  rstride=1, cstride=1, facecolors=cmap(np.angle(f2_12) / (2*pi)), alpha=1, linewidth=0)

        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.set_xlim([-1.5,1.5])
        ax.set_ylim([-1.5,1.5])
        ax.set_zlim([-1.5,1.5])
        return(fig.show())

    #For plotly
    elif inter==1:
        fig = make_subplots(rows=2, cols=2, shared_yaxes=True, specs=[[{"type": "surface"},{"type": "surface"}],
                                                              [{"type": "surface"},{"type": "surface"}]],
                                                subplot_titles=("Qubit 1", "Qubit 2",
                                                               "Identity", "Correlations"))
        fig.add_trace(go.Surface(x=x*abs(r1), y=y*abs(r1), z=z*abs(r1), surfacecolor = wrap_angle(np.angle((r1))),
                           opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 1,1)
        fig.add_trace(go.Surface(x=x*abs(r2), y=y*abs(r2), z=z*abs(r2), surfacecolor = wrap_angle(np.angle((r2))),
                           opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 1,2)
        fig.add_trace(go.Surface(x=abs(r0)*x, y=abs(r0)*y, z=abs(r0)*z, surfacecolor = wrap_angle(np.angle((r0))),
                           opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 2,1)
        fig.add_trace(go.Surface(x=abs(r12)*x, y=abs(r12)*y, z=abs(r12)*z, surfacecolor = wrap_angle(np.angle((r12))),
                           opacity=1,colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0), 2,2)
        return(fig.show())
# In[]
# Function for plotting expectation values.
# These expectation values are calculated/computed experimentally.
def Expec_plot(I1z_vals,I2z_vals,I1zI2z_vals,I1xI2x_vals,I1yI2y_vals,I1yI2x_vals,I1xI2y_vals,Id_vals):
    fig, axs = plt.subplots(4,2)
    # setting axis limits
    ymin = -1
    ymax = 1.1

    axs[0,0].plot(I1z_vals)
    axs[0,0].set_title('I1z')
    axs[0,0].set_ylim([ymin,ymax])

    axs[0,1].plot(I2z_vals)
    axs[0,1].set_title('I2z')
    axs[0,1].set_ylim([ymin,ymax])

    axs[1,0].plot(I1zI2z_vals)
    axs[1,0].set_title('I1zI2z')
    axs[1,0].set_ylim([ymin,ymax])

    axs[1,1].plot(I1xI2x_vals)
    axs[1,1].set_title('I1xI2x')
    axs[1,1].set_ylim([ymin,ymax])

    axs[2,0].plot(I1yI2y_vals)
    axs[2,0].set_title('I1yI2y')
    axs[2,0].set_ylim([ymin,ymax])

    axs[2,1].plot(I1yI2x_vals)
    axs[2,1].set_title('I1yI2x')
    axs[2,1].set_ylim([ymin,ymax])

    axs[3,0].plot(I1xI2y_vals)
    axs[3,0].set_title('I1xI2y')
    axs[3,0].set_ylim([ymin,ymax])

    axs[3,1].plot(Id_vals)
    axs[3,1].set_title('Id')
    axs[3,1].set_ylim([ymin,ymax])
    fig.tight_layout()

    for ax in axs.flat:
        ax.set(xlabel='N', ylabel='Expec')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
# In[]
# Build the quantum circuits required for state tomography. All of the circuit elements are described in terms of a general single qubit rotation gate U3. More information about U3 is available here: https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html.

# ----Input----
# res_theta: number of theta (polar) angles for tomography
# Up: quantum gate for preparing a quantum state for tomography. The initial state for these
# quantum circuits are |00>.

# ----Output---
# circuit: circuits required for performing WQST for single qubit

def WQST_2Q_circuits(res_theta,Up):
    # normal definition
    pi = np.pi

    # empty array for storing circuits
    circuits = []

    # Chosing resolution of theta and phi for the scanning purpose.
    res_phi = 2*res_theta-1
    N = res_theta*res_phi

    #----defining some rotations----These are required for 2 Qubits.
    # to_xgate takes the state from |0> to (|0>+|1>)*(1/sqrt(2))
    to_xgate = U3Gate(theta=-pi/2, phi=0, lam=0)

    # to_ygate takes the state from |0> to (|0>+i|1>)*(1/sqrt(2))
    to_ygate = U3Gate(theta=pi/2, phi=0, lam=pi/2)

    # preparation of cirucits
    nnn=0
    for theta in np.linspace(0,pi,res_theta):
        for phi in np.linspace(0,2*pi,res_phi):
            nnn+=1
            for i in range(5):
    #         defining quantum circuit
                circ_q = QuantumCircuit(2)
    #         preparation block
    #         Bell state preparation
    #             circ_q.h(0)
    #             circ_q.cx(0,1)
                circ_q.append(Up, [[0],[1]])

                circ_q.barrier()
    #         rotation block
                rotgate = U3Gate(theta=theta, phi=phi, lam=0).inverse()
                circ_q.append(rotgate, [0])
                circ_q.append(rotgate, [1])

                circ_q.barrier()
    #            Detection-associated rotations block

    #          for I1xI2x
                if i==1:
                    circ_q.append(to_xgate,[0])
                    circ_q.append(to_xgate,[1])
    #          for I1yI2y
                if i==2:
                    circ_q.append(to_ygate,[0])
                    circ_q.append(to_ygate,[1])
    #          for I1xI2y
                if i==3:
                    circ_q.append(to_xgate,[0])
                    circ_q.append(to_ygate,[1])
    #          for I1yI2x
                if i==4:
                    circ_q.append(to_ygate,[0])
                    circ_q.append(to_xgate,[1])

    # measurement
                circ_q.measure_all()
    #   draw circuit for different values of theta and phi
                if nnn==1:
                    display(circ_q.draw())
                circuits.append(circ_q)
    # printing total number of quantum ciruits. This is equivalent to N in this case.
    print("Total number of ciruits for WQST for two-qubit are:{}".format(len(circuits)))
    return(circuits)

# In[]
# Function for running quantum circuits
# ----Input----
# res_theta: resolution for polar angles (number)
# circuits: circuits for tomography (generated by WQST_1Q_circuits)
# device: simulator or experimental device to run on
# shots: number of shots for simulation/experiments
# inter: interactivity for plotting droplets (inter=0 gives non-interactive droplets)
# rhoT: target density matrix. Required for estimating fidelity.

# ----Output----
# fidelity: experimental state fidelity
# density matrix: experimentally estimated density matrix
# plot of expectation values
# droplets (rank 0 and rank 1)

def WQST_2Q_runner(res_theta,circuits,device,shots,inter,rhoT):
    res_phi = 2*res_theta-1
    N = res_theta*res_phi
    # Defining chunks for storing circuits
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # total number of shots per experiment.
    # shots = 2**13

    # giving a number of circuit you would like to run in a batch.
    # for experiments you can only do 75 circuits in a batch. This ofcourse depends on the quantum device.
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
    # for identity
    Id_vals = []

    #for z measure
    I1z_vals = []
    I2z_vals = []
    I1zI2z_vals = []

    # for x and y measure
    I1xI2x_vals = []
    I1yI2y_vals = []
    I1xI2y_vals = []
    I1yI2x_vals = []

    i = 0
    # calculation of expectation values for different droplet operator described in paper's section 4B.
    for ii, cnts in enumerate(cnt_list):
        if ii % 5 == 0:
    #         creating empty matrices
            val  = [None]*5
            val1 = [None]*5
            val2 = [None]*5
            val0 = [None]*5

    # So below is the counting of 0 and 1 on 0th qubit.
    # And in here the bits are arranged in such a way that the bit corresponding
    # to 0th qubit is in last; hence k[-1]
    # Expectation values are calculated as showsn in the supplememtary section 1 of the paper.
        val[ii % 5] = (np.sum([v for k,v in cnts.items() if '0' in k[-1]]) / shots-
                    np.sum([v for k,v in cnts.items() if '1' in k[-1]])/shots)
        val1[ii % 5] = (np.sum([v for k,v in cnts.items() if '0' in k[0]]) / shots-
                    np.sum([v for k,v in cnts.items() if '1' in k[0]])/shots)
        val2[ii % 5] = (np.sum([v for k,v in cnts.items() if ('00' in k or '11' in k)]) / shots-
                    np.sum([v for k,v in cnts.items() if ('01' in k or '10' in k)])/shots)
        val0[ii % 5] = (np.sum([v for k,v in cnts.items() if ('00' in k or '11' in k)]) / shots+
                    np.sum([v for k,v in cnts.items() if ('01' in k or '10' in k)])/shots)

    # appending these vals in corresponding operators
        if ii % 5 == 0:
            I1z_vals.append(val[0])
            I2z_vals.append(val1[0])
            I1zI2z_vals.append(val2[0])
            Id_vals.append(val0[0])
        if ii % 5 == 1:
            I1xI2x_vals.append(val2[1])
        if ii % 5 == 2:
            I1yI2y_vals.append(val2[2])
        if ii % 5 == 3:
            I1xI2y_vals.append(val2[3])
        if ii % 5 == 4:
            I1yI2x_vals.append(val2[4])

    Id_vals=np.array(Id_vals)
    I1z_vals = np.array(I1z_vals)
    I2z_vals = np.array(I2z_vals)
    I1zI2z_vals = np.array(I1zI2z_vals)
    I1xI2x_vals = np.array(I1xI2x_vals)
    I1yI2y_vals = np.array(I1yI2y_vals)
    I1xI2y_vals = np.array(I1xI2y_vals)
    I1yI2x_vals = np.array(I1yI2x_vals)

     # for plotting the DROPS
    shape = (res_theta,res_phi)

    # combining expectation values to form tensor operators

    # For identity case
    # j=0
    T00 = Id_vals*np.sqrt((2 * 0 + 1) / (4 * pi))/(2)

    # For linear case
    # j = 1
    # qubit 1
    T10_1 = I1z_vals*np.sqrt((2 * 1 + 1) / (4 * pi))*(0.5)
    # qubit 2
    T10_2 = I2z_vals*np.sqrt((2 * 1 + 1) / (4 * pi))*(0.5)

    # For bilnear cases
    # j = 0
    T00_12 = (I1xI2x_vals+I1yI2y_vals+I1zI2z_vals)*np.sqrt((2 * 0 + 1) / (4 * pi))*(np.sqrt(1 /3)*0.5)
    # j = 1
    T10_12 = (I1xI2y_vals-I1yI2x_vals)*np.sqrt((2 * 1 + 1) / (4 * pi))*(np.sqrt(1 /2)*0.5)
    # j = 2
    T20_12 = (-I1xI2x_vals-I1yI2y_vals+2*I1zI2z_vals)*np.sqrt((2 * 2 + 1) / (4 * pi))*(np.sqrt(1 /6)*0.5)

    # droplets
    # j=0
    f0_0 = np.reshape(T00,shape)

    # for linear case (l = 1)
    # j = 1
    f1_1 = np.reshape(T10_1,shape)
    f1_2 = np.reshape(T10_2,shape)

    # for bilnear cases (l = 12)
    # j = 0
    f0_12 = np.reshape(T00_12,shape)
    # j = 1
    f1_12 = np.reshape(T10_12,shape)
    # j = 2
    f2_12 = np.reshape(T20_12,shape)

    r0_expt = np.reshape(f0_0,shape)
    r1_expt = np.reshape(f1_1,shape)
    r2_expt = np.reshape(f1_2,shape)
    r12_expt = np.reshape(f0_12+f1_12+f2_12,shape)

    # use Experimental_state_analysis(iB,W,rhoT,r0_expt,r1_expt) for this analysis
    # iB: basis droplets generated using function single_basis(res_theta,res_phi)
    # W: sample weights for calculation of overlap
    # rhoT: target density matrix
    # r0_expt: experimental droplet values of rank j=0
    # r1_expt: experimental droplet values of qubit 1 of rank j=1
    # r2_expt: experimental droplet values of qubit 2 of rank j=1
    # r12_expt: experimental droplet for bilinear (coherence)
    L = Experimental_state_analysis(two_basis(res_theta,res_phi),New_sampling(res_theta-1),rhoT,r0_expt,r1_expt,r2_expt,r12_expt)
    print('State fidelity is:',L[0])
    print('Experimental state is:', L[1])
    return(Expec_plot(I1z_vals,I2z_vals,I1zI2z_vals,I1xI2x_vals,I1yI2y_vals,I1yI2x_vals,I1xI2y_vals,Id_vals),Droplet_plot(res_theta,f0_0,f1_1,f1_2,f0_12,f1_12,f2_12,inter))
