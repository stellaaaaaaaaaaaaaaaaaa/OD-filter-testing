#filter interface

import matplotlib.pyplot as plt
import numpy as np
#from pymatreader import read_mat
import spiceypy as sp
#from mpl_toolkits.mplot3d import Axes3D
from process import rotational, Model
import time
import pandas as pd
import datetime

from EKF import run_extended_kalman_filter
from CKF import run_cubature_kalman_filter
from SRCKF import run_square_root_CKF
from USKF import run_unscented_schmidt_KF
from UKF import run_unscented_kalman_filter
from hybrid import run_hybrid
from DSNSim import DSN_generation

start_time = time.time()
#reference trajectory

#NRHOtruth = pd.read_csv('statesNRHOCapstone.csv')
Xtruth = np.load('NRHOTruthPoints.npy')
tk = np.load('Time.npy')

#every hundredth value for filter debugging, comment/uncomment to utilise
Xtruthreduced = Xtruth[0::50]
tkreduced = tk[0::50]

#remove NRHO departure --> this is where errors increase massively, skew RMSE results
Xtruthshort = Xtruthreduced[0:900]
tkshort = tkreduced[0:900]

Xtruth = Xtruthshort
tk = tkshort

#set noise here https://ntrs.nasa.gov/api/citations/20200011550/downloads/20200011550.pdf

# #set measurement bias (systematic), select arbitrarily
rangebias_std = 7.5e-3 #metres to kilometres 
velocity_bias_std = 2.5e-6 #range rate mesasurement bias, mm/s to km/s

#set measurement noise (random), select arbitrarily 
rangenoise_std = 3e-3 #kilometres
velocity_noise_std = 1e-6 #kilometers/sec (0.1mm/s), range rate noise

#measurement noise covariance - reflect what the filter will realistically produce error wise - need to tune to anticipate such that the filter doesn't fail (Rk and Qd)
#using identity matrix gives best results atm with steadily increasing error
Rk = np.eye(2)
scale = 7

Rk[0,0] = (0.008*scale)**2  # 8m standard deviation  
Rk[1,1] = (0.0027e-3*20)**2  # 2.7mm/s standard deviation


#generate one bias value per component
bias = np.zeros((1, 6))  # One bias vector for all time steps
bias[0, :3] = np.random.normal(0, rangebias_std, 3)  # Position biases
bias[0, 3:] = np.random.normal(0, velocity_bias_std, 3)  # Velocity biases

# Generate random noise (different for each time step and component)
noise = np.zeros_like(bias)
noise[:, :3] = np.random.normal(0, rangenoise_std, 3)  # Position noise
noise[:, 3:] = np.random.normal(0, velocity_noise_std, 3)  # Velocity noise

#generate measurement data by incorporating bias and noise
X_initial = Xtruth[0] + bias.flatten() + noise.flatten() # convert truth to measurement
X_initial = Xtruth[0] #will this reduce the spike?

#generating range and range rate measurements
DSN_Sim_measurements, ground_station_state = DSN_generation(Xtruth, tk)

#set value for initialised covariance
initial_covar = 1

#process noise
dt_debug = 1500 #every 100th value
dt_current = tk[1]-tk[0]
time_scale = np.sqrt(dt_current/dt_debug)

Q = np.eye(6)  # process noise covariance matrix
Q[:3, :3] *= (0.1 * time_scale)**2  # Position process noise -- for EKF, it performs well at high values - suggests might be compensating for STM error
Q[3:, 3:] *= (0.0001 * time_scale)**2  # Velocity process noise 
GAMMA = np.eye(6)  # process noise mapping matrix
Qd = GAMMA @ Q @ np.transpose(GAMMA)


#for USKF only - consider parameters NOTE code only works atm with size 6 arrays (otherwise matrix operations fail)
c = np.array([ #consider parameter means, assume zero for now
    1, #rel mass error
    1, #SRP
    0,
    0,
    0,
    0
    ]) 
Pcc = c = np.array([ #consider parameter covariance, take variance
    0.3**2, #rel mass error
    0.3**2, #SRP  #https://ntrs.nasa.gov/api/citations/20220000182/downloads/NRHO_OD.pdf
    0,
    0,
    0,
    0
    ])


#select filter to run here

#1 to run, 0 to skip
run_EKF = 0
run_CKF = 0
run_SRCKF = 0
run_USKF = 0
run_UKF = 1


#select hybridisation here

#options: 'EKF-CKF', 'EKF-SRCKF', 'EKF-USKF', 'EKF-UKF'
#         'CKF-USKF', 'CKF-UKF'
# #note 'stable-unstable' regions
hybridisation = 0
stable = 'EKF'
unstable = 'CKF'   
H_criteria = 33 #set upper limit of differential entropy before switching to the more intensive, yet more accurate filter


#run filters
# Dictionary to store filter results
filter_results = {}
    
if run_EKF:
    print("Running Extended Kalman Filter...")
    ekf_results, covariance_results, residual_results, entropy_results = run_extended_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria, 0)
    filter_results['EKF'] = {
        'states': ekf_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("EKF complete!")
    
if run_CKF:
    print("Running Cubature Kalman Filter...")
    ckf_results, covariance_results, residual_results, entropy_results = run_cubature_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria, 0)
    filter_results['CKF'] = {
        'states': ckf_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("CKF complete!")
    
if run_SRCKF:
    print("Running Square Root Cubature Kalman Filter...")
    srckf_results, covariance_results, residual_results, entropy_results = run_square_root_CKF(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria, 0)
    filter_results['SRCKF'] = {
        'states': srckf_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("SRCKF complete!")
    
if run_USKF:
    print("Running Unscented Schmidt Kalman Filter...")
    uskf_results, covariance_results, residual_results, entropy_results = run_unscented_schmidt_KF(Xtruth, tk, c, Pcc, Rk, Qd, initial_covar, 0, H_criteria, 0)
    filter_results['USKF'] = {
        'states': uskf_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("USKF complete!")
    
if run_UKF:
    print("Running Unscented Kalman Filter...")
    ukf_results, covariance_results, residual_results, entropy_results = run_unscented_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria, 0)
    filter_results['UKF'] = {
        'states': ukf_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("UKF complete!")

if hybridisation:
    print("Running Hybrid Filter: {stable} - {unstable}")
    hybrid_results, covariance_results, residual_results, entropy_results = run_hybrid(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, stable, unstable, H_criteria, c, Pcc, Qd, Rk, initial_covar)
    filter_results['hybrid'] = {
        'states': hybrid_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("{stable} - {unstable} complete!")
    

#visualisation
#similar to HALO propagator visualiser (HALO actually repurposed)

# Extract time array
T = tk


# Check if filter results dictionary has any entries
if filter_results:
    # Select the first available filter for visualization
    first_filter = next(iter(filter_results))
    Ssat = filter_results[first_filter]['states'][:, :3]  # Position components only
    print(f"Using {first_filter} results for visualization")
else:
    print("Warning: No filter results available")
    Ssat = np.zeros_like(Xtruth[:, :3])  # Create empty array for visualization

# Extract truth position components
Struth = Xtruth[:, :3]

# Visualization parameters
MoonSphere = 1 # 1 if the Moon is drawn as a sphere, 0 for a point
RotationalF = 1  # Plot in Earth-Moon rotational frame (0=inertial)
Converged = 0    # Plot initial and converged trajectory (for optimization)
Earth = 0        # Plot the Earth (only in rotational frame)
model = 0        # Plot the CR3BP trajectory (for optimization)
path = "statesNRHOCapstone.csv"  # Path of the model (if needed)

# Process the change of frame
SConv = None
SEarth = None

try:
    if RotationalF: 
        Ssat, SEarth, SConv = rotational(Ssat, T, Converged, SConv)
        Struth, _, _ = rotational(Struth, T, 0, None)
    
    if model:
        ModP = Model(path, RotationalF, T)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    # Plot the filter results
    ax.plot(Ssat[:, 0], Ssat[:, 1], Ssat[:, 2], 'b-', label="Filter Results", linewidth=2)
    ax.scatter(Ssat[0, 0], Ssat[0, 1], Ssat[0, 2], c='b', s=50)

    # Plot the truth data
    ax.plot(Struth[:, 0], Struth[:, 1], Struth[:, 2], 'r-', label="Truth Data", linewidth=2)
    ax.scatter(Struth[0, 0], Struth[0, 1], Struth[0, 2], c='r', s=50)

    # Plot optional elements
    if Converged: 
        ax.plot(SConv[:, 0], SConv[:, 1], SConv[:, 2], "orange", linewidth=2, label="Traj. converged")
        ax.scatter(SConv[0, 0], SConv[0, 1], SConv[0, 2], c="orange", s=50)
        
    if model: 
        ax.plot(ModP[0], ModP[1], ModP[2], c="g", label="CR3BP")
        ax.scatter(ModP[0][0], ModP[1][0], ModP[2][0], c="g", s=50)
        
    if Earth:
        if not RotationalF:
            sp.furnsh("visualisation processes/input/de430.bsp")
            SEarth = np.zeros((len(T), 3))
            for i in range(len(T)):
                SEarth[i] = sp.spkezr("EARTH", T[i], "J2000", "NONE", "MOON")[0][:3]
        else: 
            ax.scatter(-389703, 0, 0, c="b", s=100, label="Earth_CR3BP")
            
        ax.plot(SEarth[:, 0], SEarth[:, 1], SEarth[:, 2], c="c", label="Earth")
        ax.scatter(SEarth[0, 0], SEarth[0, 1], SEarth[0, 2], c="c", s=50)
            
    # Plot the Moon
    if MoonSphere:
        rM = 1738  # Moon radius in km
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = rM * np.outer(np.cos(u), np.sin(v))
        y = rM * np.outer(np.sin(u), np.sin(v))
        z = rM * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.5, zorder=0)
    else: 
        ax.scatter(0, 0, 0, c="gray", s=100, label="Moon")

    # Set axis labels and legend
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit Determination Results')
    plt.legend()

    # Try to set equal aspect ratio
    # This is approximate and may not work perfectly in 3D
    limits = []
    for dim in [0, 1, 2]:
        min_val = min(np.min(Ssat[:, dim]), np.min(Struth[:, dim]))
        max_val = max(np.max(Ssat[:, dim]), np.max(Struth[:, dim]))
        mid_val = (min_val + max_val) / 2
        limit = max(abs(max_val - mid_val), abs(min_val - mid_val))
        limits.append((mid_val - limit, mid_val + limit))
    
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])
    
    plt.tight_layout()
    plt.show()

    # Calculate and plot position error
    if filter_results:
        for filter_name, result in filter_results.items():
            # Calculate position error (Euclidean distance)
            pos_error = np.sqrt(np.sum((result['states'][:, :3] - Xtruth[:, :3])**2, axis=1))
            
            # Plot error
            fig_error = plt.figure(figsize=(10, 6))
            ax_error = fig_error.add_subplot(111)
            ax_error.plot(T, pos_error, linewidth=2)
            ax_error.set_xlabel('Time')
            ax_error.set_ylabel('Position Error (km)')
            ax_error.set_title(f'{filter_name} Position Error vs Truth')
            ax_error.grid(True)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean(pos_error**2))
            ax_error.text(0.05, 0.95, f'RMSE: {rmse:.6f} km', 
                        transform=ax_error.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            print(f"{filter_name} RMSE: {rmse:.6f} km")
            
            # Calculate and display velocity error if available
            if result['states'].shape[1] >= 6:  # Check if velocity data exists
                vel_error = np.sqrt(np.sum((result['states'][:, 3:6] - Xtruth[:, 3:6])**2, axis=1))
                vel_rmse = np.sqrt(np.mean(vel_error**2))
                
                fig_vel_error = plt.figure(figsize=(10, 6))
                ax_vel_error = fig_vel_error.add_subplot(111)
                ax_vel_error.plot(T, vel_error, 'b-', linewidth=2)
                ax_vel_error.set_xlabel('Time')
                ax_vel_error.set_ylabel('Velocity Error (km/s)')
                ax_vel_error.set_title(f'{filter_name} Velocity Error vs Truth')
                ax_vel_error.grid(True)
                
                ax_vel_error.text(0.05, 0.95, f'Velocity RMSE: {vel_rmse:.6f} km/s', 
                                transform=ax_vel_error.transAxes, 
                                bbox=dict(facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.show()
                
                print(f"{filter_name} Velocity RMSE: {vel_rmse:.6f} km/s")
except Exception as e:
    print(f"Error in visualization: {e}")
    import traceback
    traceback.print_exc()
    # Plot position error over time
    # Plot error metrics for all active filters
    if filter_results:
        # Create position error subplot
        fig_error = plt.figure(figsize=(12, 10))
        ax_pos = fig_error.add_subplot(211)  # Top subplot for position error
        ax_vel = fig_error.add_subplot(212)  # Bottom subplot for velocity error
        
        # Colors to match trajectory plot
        filter_colors = {
            'EKF': 'blue',
            'CKF': 'green',
            'SRCKF': 'purple',
            'USKF': 'orange',
            'UKF': 'cyan'
        }
        
        # Print a header for the RMSE summary
        print("\n" + "="*50)
        print("Filter Performance Summary")
        print("="*50)
        print(f"{'Filter':<10} {'Position RMSE (km)':<20} {'Velocity RMSE (km/s)':<20}")
        print("-"*50)
        
        for filter_name, result in filter_results.items():
            # Calculate position error (Euclidean distance)
            pos_error = np.sqrt(np.sum((result['states'][:, :3] - Xtruth[:, :3])**2, axis=1))
            
            # Calculate velocity error (Euclidean distance)
            vel_error = np.sqrt(np.sum((result['states'][:, 3:] - Xtruth[:, 3:])**2, axis=1))
            
            # Calculate RMSEs
            pos_rmse = np.sqrt(np.mean(pos_error**2))
            vel_rmse = np.sqrt(np.mean(vel_error**2))
            
            # Plot position error
            color = filter_colors.get(filter_name, 'blue')
            ax_pos.plot(tk, pos_error, '-', color=color, linewidth=2, label=f"{filter_name}")
            
            # Plot velocity error
            ax_vel.plot(tk, vel_error, '-', color=color, linewidth=2, label=f"{filter_name}")
            
            # Print RMSE values to console
            print(f"{filter_name:<10} {pos_rmse:<20.6f} {vel_rmse:<20.6f}")
        
        # Configure position error subplot
        ax_pos.set_xlabel('Time')
        ax_pos.set_ylabel('Position Error (km)')
        ax_pos.set_title('Position Error vs Truth')
        ax_pos.grid(True)
        ax_pos.legend()
        
        # Configure velocity error subplot
        ax_vel.set_xlabel('Time')
        ax_vel.set_ylabel('Velocity Error (km/s)')
        ax_vel.set_title('Velocity Error vs Truth')
        ax_vel.grid(True)
        ax_vel.legend()
        
        # Finish the table
        print("="*50)
        
        plt.tight_layout()
        plt.show()
else:
    print("No filters selected to run.")
    
#total elapsed time to run filter
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = int(elapsed_time // 60)
elapsed_seconds = elapsed_time % 60

print("Performance Summary")
print(f"Total execution time: {elapsed_minutes} minutes {elapsed_seconds:.2f} seconds")


# # Process the change of frame and the model if needed
# SConv = None
# SEarth = None

# if RotationalF: 
#     Ssat,SEarth,SConv = rotational(Ssat,T,Converged,SConv)
    
# if model: ModP = Model(path,RotationalF,T)
# Plot = Ssat

# # Plot the filter results
# ax.plot(Ssat[:,0], Ssat[:,1], Ssat[:,2], 'b-', label="Filter Results", zorder=10)
# ax.scatter(Ssat[0,0], Ssat[0,1], Ssat[0,2], c='b')

# # Plot the truth data
# ax.plot(Struth[:,0], Struth[:,1], Struth[:,2], 'r-', label="Truth Data", zorder=10)
# ax.scatter(Struth[0,0], Struth[0,1], Struth[0,2], c='r')

# # Plot depending on user's choice
# if Converged: 
#     ax.plot(SConv[:,0],SConv[:,1],SConv[:,2],"orange",zorder=10,label="Traj. converged")
#     ax.scatter(SConv[0,0],SConv[0,1],SConv[0,2],c="orange")
# if model: 
#     ax.plot(ModP[0],ModP[1],ModP[2],c="g",label="CR3BP")
#     ax.scatter(ModP[0][0],ModP[1][0],ModP[2][0],c="g")
# if earth:
#     if not RotationalF:
#         sp.furnsh("de430.bsp")
#         SEarth = np.zeros((len(T),3))
#         for i in range(len(T)):
#             SEarth[i] = sp.spkezr("EARTH",T[i],"J2000","NONE","MOON")[0][:3]
#     else: ax.scatter(-389703,0,0,c="b",label="Earth_CR3BP")
#     ax.plot(SEarth[:,0],SEarth[:,1],SEarth[:,2],c="c")
#     ax.scatter(SEarth[0,0],SEarth[0,1],SEarth[0,2],c="c")
# if earth: ax.plot(SEarth[:,0],SEarth[:,1],SEarth[:,2],c="c",label="Earth")
        
# ###Graph
# ax.set_xlabel('X (km)')
# ax.set_ylabel('Y (km)')
# ax.set_zlabel('Z (km)')

# if MoonSphere:   #Plot moon
#     rM = 1738
#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 50)
#     x = rM * np.outer(np.cos(u), np.sin(v))
#     y = rM * np.outer(np.sin(u), np.sin(v))
#     z = rM * np.outer(np.ones(np.size(u)), np.cos(v))
#     ax.plot_surface(x, y, z,cmap = "gray",zorder=0)
# else: ax.scatter(0,0,0,c="gray",label = "Moon")

# plt.legend()
# plt.axis("equal")
# sp.kclear()
# plt.show()


# ## results!

# residual = np.linalg.norm(Ssat-Struth, axis=1)
# rmse_error = np.sqrt(pos_error)

# #plot

# fig_error = plt.figure(figsize=(10, 6))
# ax_error = fig_error.add_subplot(111)
# ax_error.plot(T, pos_error, 'r-')
# ax_error.set_xlabel('Time')
# ax_error.set_ylabel('Position Error (km)')
# ax_error.set_title('Filter Position Error vs Truth')
# ax_error.grid(True)
# plt.show()





