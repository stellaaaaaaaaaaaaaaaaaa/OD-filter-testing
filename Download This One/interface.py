#filter interface

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as sp
from process import rotational, Model
import time
import pandas as pd
import datetime

from EKF import run_extended_kalman_filter
from CKF import run_cubature_kalman_filter
from UKF import run_unscented_kalman_filter
from hybrid import run_hybrid, run_hybrid_time_based
from DSNSim import DSN_generation

start_time = time.time()
#reference trajectory

Xtruthload = np.load('NRHOTruthPoints.npy')
tkload = np.load('Time.npy')

#every hundredth value for filter debugging, comment/uncomment to utilise
Xtruthreduced = Xtruthload[0::100]
tkreduced = tkload[0::100]

Xtruth = Xtruthreduced[0:475]
tk = tkreduced[0:475]

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
dt_current = tk[1]-tk[0]

Rk[0,0] = (0.008/3)**2  # 8m 3-standard deviation  
Rk[1,1] = (0.0027e-3/3)**2  # 2.7mm/s 3-standard deviation

#generate one bias value per component
bias = np.zeros((1, 6))  #one bias vector for all time steps
bias[0, :3] = np.random.normal(0, rangebias_std, 3)  #position biases
bias[0, 3:] = np.random.normal(0, velocity_bias_std, 3)  #velocity biases

#generate random noise (different for each time step and component)
noise = np.zeros_like(bias)
noise[:, :3] = np.random.normal(0, rangenoise_std, 3)  #position noise
noise[:, 3:] = np.random.normal(0, velocity_noise_std, 3)  #velocity noise

#generate measurement data by incorporating bias and noise
X_initial = Xtruth[0] + bias.flatten() + noise.flatten() #convert truth to measurement
X_initial = Xtruth[0] #will this reduce the spike? -- it ended up not *really* mattering

#generate range and range rate measurements
DSN_Sim_measurements, ground_station_state = DSN_generation(Xtruth, tk)

#set value for initialised covariance
initial_covar = 0.01

#process noise

Q = np.eye(6)  #process noise covariance matrix
Q[:3, :3] *= (1e-1)**2  #position process noise km
Q[3:, 3:] *= (1e-4)**2  #velocity process noise km
GAMMA = np.eye(6)  #process noise mapping matrix
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
#take note that the USKF and SRCKF are not working/properly not implemented, future work

#1 to run, 0 to skip
run_EKF = 0
run_CKF = 0
run_UKF = 0

#select hybridisation here

#options: 'EKF-CKF', 'EKF-SRCKF', 'EKF-USKF', 'EKF-UKF'
# #note 'stable-unstable' regions
entropy_hybridisation = 0
stable = 'EKF'
unstable = 'CKF'   

H_criteria_stable = 28.5
H_criteria_unstable = 17 #set upper limit of differential entropy before switching to the more intensive, yet more accurate filter

time_hybridisation = 0

switch_times = [7.5752778e8, 7.5772224e8, 7.581112e8, 7.583344e8, 7.586389e8] #list of times to switch between filters


#run filters
filter_results = {}
    
if run_EKF:
    print("Running Extended Kalman Filter...")
    ekf_results, ekf_covariance_results, ekf_residual_results, ekf_entropy_results = run_extended_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria_stable, 0)
    filter_results['EKF'] = {
        'states': ekf_results,
        'covariances': ekf_covariance_results, 
        'residuals': ekf_residual_results,
        'entropy': ekf_entropy_results
            }
    print("EKF complete!")
    
if run_CKF:
    print("Running Cubature Kalman Filter...")
    ckf_results, ckf_covariance_results, ckf_residual_results, ckf_entropy_results = run_cubature_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria_unstable, 0)
    filter_results['CKF'] = {
        'states': ckf_results,
        'covariances': ckf_covariance_results, 
        'residuals': ckf_residual_results,
        'entropy': ckf_entropy_results
            }
    print("CKF complete!")
    
if run_UKF:
    print("Running Unscented Kalman Filter...")
    ukf_results, ukf_covariance_results, ukf_residual_results, ukf_entropy_results = run_unscented_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, 0, H_criteria_unstable, 0)
    filter_results['UKF'] = {
        'states': ukf_results,
        'covariances': ukf_covariance_results, 
        'residuals': ukf_residual_results,
        'entropy': ukf_entropy_results
            }
    print("UKF complete!")

if entropy_hybridisation:
    print("Running Entropy Based Hybrid Filter: {stable} - {unstable}")
    hybrid_results, covariance_results, residual_results, entropy_results = run_hybrid(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, stable, unstable, H_criteria_stable, H_criteria_unstable, c, Pcc, Qd, Rk, initial_covar)
    filter_results['hybrid'] = {
        'states': hybrid_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("{stable} - {unstable} complete!")
    
if time_hybridisation:
    print("Running Time Based Hybrid Filter: {stable} - {unstable}")
    hybrid_results, covariance_results, residual_results, entropy_results = run_hybrid_time_based(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, stable, unstable, switch_times, Pcc, Qd, Rk, initial_covar)
    filter_results['hybrid'] = {
        'states': hybrid_results,
        'covariances': covariance_results, 
        'residuals': residual_results,
        'entropy': entropy_results
            }
    print("{stable} - {unstable} complete!")
    

#visualisation
#similar to HALO propagator visualiser 
#much of the code repurposed from HALO visualise.py

#Extract time array
T = tk


if filter_results:
    first_filter = next(iter(filter_results))
    Ssat = filter_results[first_filter]['states'][:, :3]  #position only
    print(f"Using {first_filter} results for visualization")
else:
    print("Warning: No filter results available")
    Ssat = np.zeros_like(Xtruth[:, :3])  

#truth position
Struth = Xtruth[:, :3]

#visualisation parameters as in HALO
MoonSphere = 1 # 1 if the Moon is drawn as a sphere, 0 for a point
RotationalF = 1  #plot in Earth-Moon rotational frame (0=inertial)
Converged = 0    #plot initial and converged trajectory (for optimization)
Earth = 0        #plot the Earth (only in rotational frame)
model = 0        #plot the CR3BP trajectory (for optimization)
path = "statesNRHOCapstone.csv"  #path of the model (if needed)


SConv = None
SEarth = None

try:
    if RotationalF: 
        Ssat, SEarth, SConv = rotational(Ssat, T, Converged, SConv)
        Struth, _, _ = rotational(Struth, T, 0, None)
    
    if model:
        ModP = Model(path, RotationalF, T)


    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    #plot results
    ax.plot(Ssat[:, 0], Ssat[:, 1], Ssat[:, 2], 'b-', label="Filter Results", linewidth=2)
    ax.scatter(Ssat[0, 0], Ssat[0, 1], Ssat[0, 2], c='b', s=50)

    #plot truth 
    ax.plot(Struth[:, 0], Struth[:, 1], Struth[:, 2], 'r-', label="Truth Data", linewidth=2)
    ax.scatter(Struth[0, 0], Struth[0, 1], Struth[0, 2], c='r', s=50)

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
            
    #Moon
    if MoonSphere:
        rM = 1738  #moon radius in km
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = rM * np.outer(np.cos(u), np.sin(v))
        y = rM * np.outer(np.sin(u), np.sin(v))
        z = rM * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.5, zorder=0)
    else: 
        ax.scatter(0, 0, 0, c="gray", s=100, label="Moon")

    #graph labels
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit Determination Results')
    plt.legend()


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

    if filter_results:
        for filter_name, result in filter_results.items():
            pos_error = np.sqrt(np.sum((result['states'][:, :3] - Xtruth[:, :3])**2, axis=1))
            
            #pos error
            fig_error = plt.figure(figsize=(10, 6))
            ax_error = fig_error.add_subplot(111)
            ax_error.plot(T, pos_error, linewidth=2)
            ax_error.set_xlabel('Time')
            ax_error.set_ylabel('Position Error (km)')
            ax_error.set_title(f'{filter_name} Position Error vs Truth')
            ax_error.grid(True)
            
            #RMSE
            rmse = np.sqrt(np.mean(pos_error**2))
            ax_error.text(0.05, 0.95, f'RMSE: {rmse:.6f} km', 
                        transform=ax_error.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            print(f"{filter_name} RMSE: {rmse:.6f} km")
            
            #plot vel error
            if result['states'].shape[1] >= 6:  
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
#plot trajectory
    if filter_results:
        fig_error = plt.figure(figsize=(12, 10))
        ax_pos = fig_error.add_subplot(211)  
        ax_vel = fig_error.add_subplot(212)  
        
    
        filter_colors = {
            'EKF': 'blue',
            'CKF': 'green',
            'SRCKF': 'purple',
            'USKF': 'orange',
            'UKF': 'cyan'
        }
        
        print("\n" + "="*50)
        print("Filter Performance Summary")
        print("="*50)
        print(f"{'Filter':<10} {'Position RMSE (km)':<20} {'Velocity RMSE (km/s)':<20}")
        print("-"*50)
        
        for filter_name, result in filter_results.items():
            #pos error
            pos_error = np.sqrt(np.sum((result['states'][:, :3] - Xtruth[:, :3])**2, axis=1))
            
            #vel error
            vel_error = np.sqrt(np.sum((result['states'][:, 3:] - Xtruth[:, 3:])**2, axis=1))
            
            #RMSE
            pos_rmse = np.sqrt(np.mean(pos_error**2))
            vel_rmse = np.sqrt(np.mean(vel_error**2))
            
            #plot pos error
            color = filter_colors.get(filter_name, 'blue')
            ax_pos.plot(tk, pos_error, '-', color=color, linewidth=2, label=f"{filter_name}")
            
            #plot vel error
            ax_vel.plot(tk, vel_error, '-', color=color, linewidth=2, label=f"{filter_name}")
            
            #print RMSE 
            print(f"{filter_name:<10} {pos_rmse:<20.6f} {vel_rmse:<20.6f}")
        
        ax_pos.set_xlabel('Time')
        ax_pos.set_ylabel('Position Error (km)')
        ax_pos.set_title('Position Error vs Truth')
        ax_pos.grid(True)
        ax_pos.legend()
        
        ax_vel.set_xlabel('Time')
        ax_vel.set_ylabel('Velocity Error (km/s)')
        ax_vel.set_title('Velocity Error vs Truth')
        ax_vel.grid(True)
        ax_vel.legend()
        
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

print(f"Total operation time: {elapsed_minutes} minutes {elapsed_seconds:.2f} seconds")







