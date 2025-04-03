#filter interface

import matplotlib.pyplot as plt
import numpy as np
from pymatreader import read_mat
import spiceypy as sp
from mpl_toolkits.mplot3d import Axes3D
from process import rotational, Model

from EKF import run_extended_kalman_filter
from CKF import run_cubature_kalman_filter
from SRCKF import run_square_root_CKF
from USKF import run_unscented_schmidt_KF
from UKF import run_unscented_kalman_filter
from hybrid import run_hybrid

#reference trajectory

Xtruth = np.load("NRHOTruthPoints.npy") # x y z generated from HALO propagator
#note, may choose to try to integrate HALO such that there is no need to manually save then call these points

tk = np.load("Time.npy") #time data for each measurement

#set noise here
Rk = [] #measurement noise --> I think its just bias and noise as a matrix for the state???

#convert to XREF by adding measurement noise:
#set measurement bias (systematic), select arbitrarily
rangebias_std = 7.5e-3 #kilometres

#set measurement noise (random), select arbitrarily 
rangenoise_std = 3e-3 #kilometres

#generate measurement data by incorporating bias and noise
XREF = Xtruth + rangebias_std + np.random.normal(0, rangenoise_std, Xtruth.shape) #convert truth to measurement by adding measurement noise

#process noise
Q = #process noise covariance matrix
GAMMA = #process noise mapping matrix
Qd = GAMMA @ Q @ np.transpose(GAMMA)

#for USKF only - consider parameters
c = #consider parameters
Pcc = #consider parameter covariance


#select filter to run here

#1 to run, 0 to skip
run_EKF = 0
run_CKF = 0
run_SRCKF = 0
run_USKF = 0
run_UKF = 0


#select hybridisation here

#options: 'EKF-CKF', 'EKF-SRCKF', 'EKF-USKF', 'EKF-UKF'
#         'CKF-USKF', 'CKF-UKF'
#note 'stable-unstable' regions
hybridisation = 0
stable = 'EKF'
unstable = 'CKF'   

#set boundary conditions here
x1 = 
y1 =
z1 =
bc1 = np.array([x1 y1 z1])

x2 = 
y2 =
z2 = 
bc2 = np.array([x2 y2 z2])


#run filters
# Dictionary to store filter results
filter_results = {}
    
if run_EKF:
    filter_results['EKF'] = run_extended_kalman_filter(Xtruth, tk)
    
if run_CKF:
    filter_results['CKF'] = run_cubature_kalman_filter(Xtruth, tk, Qd)
    
if run_SRCKF:
    filter_results['SRCKF'] = run_square_root_CKF(Xtruth, tk, Qd)
    
if run_USKF:
    filter_results['USKF'] = run_unscented_schmidt_KF(Xtruth, tk, c, Pcc, Qd)
    
if run_UKF:
    filter_results['UKF'] = run_unscented_kalman_filter(Xtruth, tk, Qd)

if hybridisation:
    filter_results['hybrid'] = run_hybrid(Xtruth, tk, stable, unstable, bc1, bc2, Qd, c, Pcc, Qd)
    
    

#visualisation
#similar to HALO propagator visualiser (HALO actually repurposed)

#adjust names to fit naming conventions:
T = tk
Ssat = filter_results[['X', 'Y', 'Z']].values
Struth = Xtruth[['X', 'Y', 'Z']].values

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')

### User's choice
MoonSphere =  0                                 #1 if the Moon is drawn as a sphere, 0 for a point
RotationalF = 1                                 #Plot the sequential in the Earth-Moon rotational frame (0=inertial)
Converged =   0                                 #Plot the initial and converged trajectory (for optimization mode)
Earth =       0                                 #Plot the earth (only in rotat. frame)
model =       1                                 #Plot the CR3BP trajectory (for optimization mode)
path = "statesNRHOCapstone.csv"                 #Path of the model (initial condition for optimization)

# ### Load the computed data - needs to be altered for simply plotting reference trajectory and whatever trajectory is output from filters

# Process the change of frame and the model if needed
SConv = None
SEarth = None

if RotationalF: 
    Ssat,SEarth,SConv = rotational(Ssat,T,Converged,SConv)
    
if model: ModP = Model(path,RotationalF,T)
Plot = Ssat

# Plot the filter results
ax.plot(Ssat[:,0], Ssat[:,1], Ssat[:,2], 'b-', label="Filter Results", zorder=10)
ax.scatter(Ssat[0,0], Ssat[0,1], Ssat[0,2], c='b')

# Plot the truth data
ax.plot(Struth[:,0], Struth[:,1], Struth[:,2], 'r-', label="Truth Data", zorder=10)
ax.scatter(Struth[0,0], Struth[0,1], Struth[0,2], c='r')

# Plot depending on user's choice
if Converged: 
    ax.plot(SConv[:,0],SConv[:,1],SConv[:,2],"orange",zorder=10,label="Traj. converged")
    ax.scatter(SConv[0,0],SConv[0,1],SConv[0,2],c="orange")
if model: 
    ax.plot(ModP[0],ModP[1],ModP[2],c="g",label="CR3BP")
    ax.scatter(ModP[0][0],ModP[1][0],ModP[2][0],c="g")
if earth:
    if not RotationalF:
        sp.furnsh("de430.bsp")
        SEarth = np.zeros((len(T),3))
        for i in range(len(T)):
            SEarth[i] = sp.spkezr("EARTH",T[i],"J2000","NONE","MOON")[0][:3]
    else: ax.scatter(-389703,0,0,c="b",label="Earth_CR3BP")
    ax.plot(SEarth[:,0],SEarth[:,1],SEarth[:,2],c="c")
    ax.scatter(SEarth[0,0],SEarth[0,1],SEarth[0,2],c="c")
if earth: ax.plot(SEarth[:,0],SEarth[:,1],SEarth[:,2],c="c",label="Earth")
        
###Graph
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')

if MoonSphere:   #Plot moon
    rM = 1738
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = rM * np.outer(np.cos(u), np.sin(v))
    y = rM * np.outer(np.sin(u), np.sin(v))
    z = rM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,cmap = "gray",zorder=0)
else: ax.scatter(0,0,0,c="gray",label = "Moon")

plt.legend()
plt.axis("equal")
sp.kclear()
plt.show()


## results!

pos_error = np.linalg.norm(Ssat-Struth, axis=1)
rmse_error = np.sqrt(pos_error)

#plot

fig_error = plt.figure(figsize=(10, 6))
ax_error = fig_error.add_subplot(111)
ax_error.plot(T, pos_error, 'r-')
ax_error.set_xlabel('Time')
ax_error.set_ylabel('Position Error (km)')
ax_error.set_title('Filter Position Error vs Truth')
ax_error.grid(True)
plt.show()





