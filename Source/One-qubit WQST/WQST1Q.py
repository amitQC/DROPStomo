#--------
# Wigner Quantum State Tomography for single qubit system
# Authors: Amit Devra, Niklas Glaser, Dennis Huber, and Steffen J Glaser
# Based on the study:
#--------
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
from matplotlib import cm, colors
import scipy
from scipy.special import sph_harm
# imports for plotting
import plotly.express as px
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#In[] Functions for analysis and plotting of single qubit state tomography result

# For generating basis we are using scipy.special.sph_harm function. Here, scipy.special.sph_harm(m,j) where m is the order and j is tha rank.
#
# The basis conversion, i.e., from Pauli basis to Spherical tensor basis is provided in Appendix 4 of https://arxiv.org/abs/1409.5417#
#
# Creating a function for generating basis for single qubit. Input is number of theta (nth) and number of phi (nph) angle values.
# In[2]:
# nth=8
# nph=15
def Single_basis(nth,nph):
    # polar angle
    theta = np.linspace(0, np.pi, nth)
    # azimuthal angle
    phi = np.linspace(0, 2*np.pi, nph)

    phi, theta = np.meshgrid(phi, theta)

    # Generating the basis droplets for single qubit system
    BId = 2*(1/np.sqrt(2))*scipy.special.sph_harm(0, 0, phi, theta)
    Bx = np.sqrt(2)*np.sqrt(1/2)*(scipy.special.sph_harm(-1, 1, phi, theta)-scipy.special.sph_harm(1, 1, phi,theta))
    By = np.sqrt(2)*np.sqrt(1/2)*1j*(scipy.special.sph_harm(-1, 1, phi, theta)+scipy.special.sph_harm(1, 1, phi,theta))
    Bz = np.sqrt(2)*(scipy.special.sph_harm(0, 1, phi, theta))
    B0 = [BId,Bx,By,Bz]
    return B0
# New_Sampling: this function generates the weights for the equiangular sampling technique for calculating the overlap between two DROPS. This is based on Supplementary section 3 of the paper.
#
# Let $N$ is the number of points to sample in a sphere, and the angle increments by $d = \pi/N$. The phase or azimuthal angles ($\phi$) have 2$N$+1 equally spaced values between [0,2$\pi$] and similarly the polar angles ($\theta$) have $N$+1 equally spaced values between [0,$\pi$]. Let $\text{w}(l,\bar{l})$ be the sampling weight matrix of size (2$N$+1)$\times$($N$+1).

# In[3]:


# Sampling points
# discretiziation of equiangular grids
# N = 7
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

# DROPoverlap function: for calculation of overlap between the droplets. Based on Supplementary section 2 of paper.

# In[4]:
# input droplets
# Sampling weights
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
    # calculation of overlap between f1 and f2
    c = W*np.conj(f11)*f2
    # sum over rows
    c1 = np.sum(c,axis=0)
    # sum over columns
    c2 = np.sum(c1)
    coeff = c2
    return(coeff)

# For calculation of overlap with experimental droplets
# In[5]:
# required files: ideal basis, experimental droplets r values
def Experimental_state_analysis(basis,W,rhoT,r0_expt,r1_expt):
    iB = basis
# calculating overlap of basis droplets
    coeff1=1j*np.zeros((4,4))
    for k in range(0,4):
        for j in range(0,4):
            coeff1[k][j] = DROPoverlap(iB[k],iB[j],W)

# overlap of experimental and simualted rank j=0 droplet
# here we are only taking overlap with label l= 0, and j=0 droplet of both experiments and simulated ideal droplet
# dividing by coeff1 to normalize
    coeff2 = DROPoverlap(iB[0],r0_expt,W)/coeff1[0][0]

# overlap of experimental and simualted label l = 1, and rank j=1 droplet
    coeff3 = 1j*np.zeros((1,3))
    for j in range (0,3):
        coeff3[0][j] = DROPoverlap(iB[j+1],r1_expt,W)/coeff1[j+1][j+1]

# define Pauli basis for reconstruction of density matrix
    PId = np.matrix([[1,0],
           [0,1]])
    Px = np.matrix([[0,1],
           [1,0]])
    Py = np.matrix([[0,-1j],
           [1j,0]])
    Pz = np.matrix([[1,0],
           [0,-1]])

# recreating density matrix
# generate l=0 and j=0 component
    rho0 = coeff2*PId
# generate l=1 and j=1 component
    rho1 = coeff3[0][0]*Px+coeff3[0][1]*Py+coeff3[0][2]*Pz
# combine them together. Divide by 2 to normalize
    rho = np.matrix((rho0+rho1))


# Fidelity calculation
# rhoT: target density matrix
    F = np.trace(np.matrix(rhoT).getH()*rho)/(np.sqrt(np.trace(np.matrix(rhoT).getH()*rhoT))*np.sqrt(np.trace(np.matrix(rho).getH()*rho)))
    return(F,rho)


# --For plotting droplets--
#
# Using two methods for plotting: matplotlib (in which droplets are not interactive), plotly (for interactive droplets).
#
# First, this is the colorscheme for the spherical droplets (DROPS representation) which will be required later.
# You can find more about DROPS color representaion here: https://spindrops.org/color_code.html?highlight=color

# Function for matplotlib
# %%
# wrap function for angles used in plotting (plotly)
def wrap_angle(angles):
    wangle = np.empty(angles.shape)
    for i in range(len(angles)):
        for j in range(len(angles[i])):
            wangle[i][j] = angles[i][j] - np.floor(angles[i][j]/(2*np.pi))*2*pi
    return(wangle)


def getPhaseColor(value):
    x, y = np.cos(value), np.sin(value)
    r = np.power(np.maximum(0, np.minimum(1, ( x + y + 1) / 2)), 0.7)
    g = np.power(np.maximum(0, np.minimum(1, (-x + y + 1) / 2)), 0.7)
    yp = 0.195 * x + 0.981 * y;
    b = np.power(np.maximum( 0., -y) * np.minimum(1., 1.41 * np.maximum( 0. , -yp)), 0.7)
    return (255*r,255*g,255*b)


# Weighted averaging of RGB fragment colors for manual shading, the smaller the spherical function value
# the smaller its weighting -> compensate artifacts at nodal points
def getAverageRGB(vertex1, vertex2, vertex3, weight1, weight2, weight3):
    rgb1 = getPhaseColor(vertex1)
    rgb2 = getPhaseColor(vertex2)
    rgb3 = getPhaseColor(vertex3)

    # adjustable weighting factor
    power = 1
    total_weight = np.power(weight1,power) + np.power(weight2,power) + np.power(weight3,power)

    r_av = int((np.power(weight1,power)*rgb1[0] + np.power(weight2,power)*rgb2[0] + np.power(weight3,power)*rgb3[0])/total_weight)
    g_av = int((np.power(weight1,power)*rgb1[1] + np.power(weight2,power)*rgb2[1] + np.power(weight3,power)*rgb3[1])/total_weight)
    b_av = int((np.power(weight1,power)*rgb1[2] + np.power(weight2,power)*rgb2[2] + np.power(weight3,power)*rgb3[2])/total_weight)
    return (r_av,g_av,b_av)


# This functions triangulates the experimental values in the correct order to plot the tomography without any gaps
def DrawSphericalFunction(x, y, z, res_theta, res_phi, values, figure, subplot_row, subplot_col):
    x = x * abs(values)
    y = y * abs(values)
    z = z * abs(values)
    angles = wrap_angle(np.angle((values)))
    angles = np.reshape(angles, (res_theta*res_phi,1))
    weights = np.reshape(abs(values), (res_theta*res_phi,1))

    # Calculate all possible vertices from value array and scanning positions
    x_ = []
    y_ = []
    z_ = []
    for i in range(res_theta):
        for j in range(res_phi):
            x_.append(x[i][j])
            y_.append(y[i][j])
            z_.append(z[i][j])

    # Now the complicated part, efficiently define the banding around the scanned sphere and triangulate the vertices
    # Should leave no gaps
    i_ = []
    j_ = []
    k_ = []
    facecolors = []
    for i in range(res_theta - 1):
        for j in range(res_phi):
            if ((i+1)*res_phi + (j+1)) == res_theta*res_phi:
                continue
            idx1 = i*res_phi + j
            idx2 = (i+1)*res_phi + (j+1)
            idx3 = (i+1)*res_phi + j
            idx4 = i*res_phi + (j+1)
            i_.append(idx1)
            j_.append(idx2)
            k_.append(idx3)
            # get the triangle fragment weighted RGB color
            facecolors.append(px.colors.label_rgb(getAverageRGB(angles[idx1][0],
                                                                angles[idx2][0],
                                                                angles[idx3][0],
                                                                weights[idx1][0],
                                                                weights[idx2][0],
                                                                weights[idx3][0])))
            i_.append(idx1)
            j_.append(idx4)
            k_.append(idx2)
            # get the triangle fragment weighted RGB color
            facecolors.append(px.colors.label_rgb(getAverageRGB(angles[idx1][0],
                                                                angles[idx2][0],
                                                                angles[idx4][0],
                                                                weights[idx1][0],
                                                                weights[idx2][0],
                                                                weights[idx4][0])))
    # return the plot, directly insert in specified parent subplot
    return figure.add_trace(go.Mesh3d(
            x=x_,
            y=y_,
            z=z_,
            i=i_,
            j=j_,
            k=k_,
            facecolor = facecolors,
            lighting=dict(ambient=1, specular=0, diffuse=0, roughness=0, fresnel=0)
            ), subplot_row, subplot_col)


# In[7]:
# Colorscheme
def Droplet_plot(res_theta,f0_0,f1_1,inter):
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
    #     -------

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

    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)

    #For matplotlib
    fig = plt.figure(figsize=(5,5))

    if inter==0:
        #Set colours and render
        ax = fig.add_subplot(111, projection='3d')

        # plotting rank 0 and rank 1 droplets.
        ax.plot_surface(
            abs(r0) * x+1, abs(r0) * y, abs(r0) * z,  rstride=1, cstride=1, facecolors=cmap(np.angle(r0)/(2*pi)), alpha=1, linewidth=0)
        ax.plot_surface(
            abs(r1) * x, abs(r1) * y, abs(r1) * z,  rstride=1, cstride=1, facecolors=cmap(np.angle(r1)/(2*pi)), alpha=1, linewidth=0)

        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        ax.set_zlim([-1,1])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        plt.tight_layout()
        return(fig.show())

    #For plotly
    elif inter==1:
        # Make a subplot
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "mesh3d"},{"type": "mesh3d"}]],
                            subplot_titles=("Identity", "Qubit 1"))
        # Rank 0
        DrawSphericalFunction(x, y, z, res_theta, res_phi, r0, fig, 1, 1)
        # Rank 1
        DrawSphericalFunction(x, y, z, res_theta, res_phi, r1, fig, 1, 2)
        # Add the colorbar
        fig.add_trace(go.Mesh3d(
                x=[None],
                y=[None],
                z=[None], intensity = [None],
                colorscale=cmap3,cmax=2*np.pi,cmid=np.pi,cmin=0, showscale=True, colorbar=dict(
                title="Phase",
                titleside="top",
                tickmode="array",
                tickvals=[0, 1.5708, 3.1416, 4.7124, 6.2832],
                labelalias={0: "0", 1.5708: "π/2", 3.1416: "π", 4.7124:"3π/2", 6.2832:"2π"},
                ticks="outside"
            )), 1, 2)
        # The default axis limit for all dimensions
        axeslimit = .5
        # update all axes
        fig.update_layout(scene=dict(xaxis=dict(range=[-axeslimit, axeslimit]), yaxis=dict(range=[-axeslimit, axeslimit]), zaxis=dict(range=[-axeslimit, axeslimit]), aspectmode='cube'), scene2=dict(xaxis=dict(range=[-axeslimit, axeslimit]), yaxis=dict(range=[-axeslimit, axeslimit]), zaxis=dict(range=[-axeslimit, axeslimit]), aspectmode='cube'))
        return fig.show()

# In[]
## Function for plotting expectation values
def Expec_plot(Expec0,Expec1):
    plt.figure(figsize=(10,6))  # adding this line sets the figure size
    plt.plot(Expec0, '-o', linewidth = 1, label = 'Id')
    plt.plot(Expec1, '-o', linewidth = 1, label = 'z')
    plt.xlabel('N (scanning angles)', fontsize = 12)
    plt.ylabel('Expec vals', fontsize = 12)
    ymin = -1
    ymax = 1.1
    plt.ylim([ymin, ymax])
    plt.xlim([0,len(Expec0)])
    plt.legend()
    plt.title('Expectation Values', fontsize = 16)
    return(plt.show())


# In[14]:
# Build the quantum circuits required for state tomography. All of the circuit elements are described in terms of a general single qubit rotation gate U3. More information about U3 is available here: https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html.

# ----Input----
# res_theta: number of theta (polar) angles for tomography
# Up: quantum gate for preparing a quantum state for tomography. The initial state for these
# quantum circuits are |0>.

# ----Output---
# circuit: circuits required for performing WQST for single qubit
def WQST_1Q_circuits(res_theta,Up):
    # normal definitions
    pi = np.pi

    # empty array for storing circuits
    circuits = []

    # Chosing resolution of theta and phi for the scanning purpose.
    res_phi = 2*res_theta-1
    N = res_theta*res_phi

    #----defining some rotations----These are required for 2Qubits.
    # to_xgate takes the state from |0> to (|0>+|1>)*(1/sqrt(2))
    to_xgate = U3Gate(theta=-pi/2, phi=0, lam=0)

    # to_ygate takes the state from |0> to (|0>+i|1>)*(1/sqrt(2))
    to_ygate = U3Gate(theta=pi/2, phi=0, lam=pi/2)
    # **Preparation of circuits**
    nnn = 0
    for theta in np.linspace(0,pi,res_theta):
        for phi in np.linspace(0,2*pi,res_phi):
            nnn+=1
    #         define size of quantum circuit
            circ_q = QuantumCircuit(1)

    #       Preparation of desired quantum state using Up gate
            circ_q.append(Up,[0])
    #       you can also use some specific gate such as Hadamard (h).
            # circ_q.z(0)


    #     rotation for scanning purpose
            rotgate = U3Gate(theta=theta, phi=phi, lam=0).inverse()
            circ_q.append(rotgate, [0])

    #     adding measurement
            circ_q.measure_all()

    #   draw circuit for different values of theta and phi
            if nnn==1:
                display(circ_q.draw())
                # display(circ_q.draw(output='mpl'))
            circuits.append(circ_q)

    # printing total number of quantum ciruits. This is equivalent to N in this case.
    print("Total number of ciruits for WQST for single qubit are:{}".format(len(circuits)))
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

def WQST_1Q_runner(res_theta,circuits,device,shots,inter,rhoT):
    res_phi = 2*res_theta-1
    N = res_theta*res_phi

    # Defining chunks for storing circuits
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # total number of shots per experiment.
    # shots = 8192

    # giving a number of circuit you would like to run in a batch.
    circuit_chunks = list(chunks(circuits,res_phi*res_theta))

    jobs=[]
    # result of jobs are stored here
    jobs = [None]*len(circuit_chunks)

    # Executing the jobs
    for i, circuit_l in enumerate(circuit_chunks):
        jobs[i] = execute(circuit_l,device,shots=shots)

    #job monitor
    for i, job in enumerate(jobs):
        print(f'Job {i} of {len(jobs)}')
        display(job_monitor(job))

    print(len(jobs))

    # ---for calculation of expectation values---
    # For counting zeros (0) and ones (1) from the data.
    cnt_list = []
    for job in jobs:
        cnt_list.extend(job.result().get_counts())

    # Expectation values with rank j=0
    Expec0 = []
    # Expectation values with rank j=1
    Expec1 = []

    i = 0

    # # calculation of expectation values for different droplet operator f_{0}^{0} and f_{1}^{1}
    for ii, cnts in enumerate(cnt_list):
        val = [None]*N
        val1 = [None]*N

    # Here the counting of 0 and 1 on 0th qubit.
    # Expectation values are calculated as shown in the supplememtary section 1 of the paper.
    # for Iz (rank 1)
        val[ii] = (np.sum([v for k,v in cnts.items() if '0' in k])/shots-np.sum([v for k,v in cnts.items() if '1' in k])/shots)
    # for id (rank 0)
        val1[ii] = (np.sum([v for k,v in cnts.items() if '0' in k])/shots+
                np.sum([v for k,v in cnts.items() if '1' in k])/shots)
    # appending them in files
        Expec1.append(val[ii])
        Expec0.append(val1[ii])
    # expectation values
    Expec0 = np.array(Expec0)
    Expec1 = np.array(Expec1)

    # for plotting the DROPS
    shape = (res_theta,res_phi)
    # combining expectation values to form droplet operators

    # for j=0
    T00 = Expec0*np.sqrt((2 * 0 + 1) / (4 * pi))/(np.sqrt(2))
    # j=1
    T10 = Expec1*np.sqrt((2 * 1 + 1) / (4 * pi))/(np.sqrt(2))

    f0_0 = np.reshape(T00,shape)
    f1_1 = np.reshape(T10,shape)
    # for analysis of droplets (calculating fidelity and density matrix)
    # experimental droplet data
    r0_expt = np.reshape(f0_0,shape)
    r1_expt = np.reshape(f1_1,shape)

    # use Experimental_state_analysis(iB,W,rhoT,r0_expt,r1_expt) for this analysis
    # iB: basis droplets generated using function single_basis(res_theta,res_phi)
    # W: sample weights for calculation of overlap
    # rhoT: target density matrix
    # r0_expt: experimental droplet values of rank j=0
    # r1_expt: experimental droplet values of rank j=1

    L = Experimental_state_analysis(Single_basis(res_theta,res_phi),New_sampling(res_theta-1),rhoT,r0_expt,r1_expt)
    print('State fidelity is:',L[0])
    print('Experimental state is:', L[1])
    # plotting the expectation values
    return(Droplet_plot(res_theta,f0_0,f1_1,inter),Expec_plot(Expec0, Expec1))
