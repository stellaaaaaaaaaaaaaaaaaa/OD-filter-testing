# Extended Kalman Filter (EKF) Algorithm - Improved Implementation

def run_extended_kalman_filter(XREF, tk, Rk, Qd, initial_covar):

    import numpy as np
    from CR3BP import NRHOmotion, STM
    
    # Set up empty arrays to store filter results
    state_dim = 6  # State dimension is 6 (position and velocity)
    n_points = len(XREF)
    
    ekf_results = np.zeros_like(XREF)
    covariance_results = np.zeros((n_points, state_dim, state_dim))
    residual_results = np.zeros_like(XREF)
    
    # Initialize first state with reference measurement
    ekf_results[0] = XREF[0].copy()
    
    # Initialize covariance matrix with different values for position and velocity
    P0 = np.eye(state_dim)
    # Position uncertainty (first 3 diagonal elements)
    P0[:3, :3] *= initial_covar
    # Velocity uncertainty (last 3 diagonal elements) - increased by random factor
    P0[3:, 3:] *= initial_covar * 10
    covariance_results[0] = P0
    
    # Progress tracking
    progress_interval = max(1, n_points // 20)
    
    for i in range(n_points - 1):
        
        # Print progress
        if i % progress_interval == 0:
            print(f"EKF progress: {i}/{n_points-1} steps ({i/(n_points-1)*100:.1f}%)")
        
        # Current time and next time (in physical units)
        tkminus1 = tk[i]
        tk_next = tk[i+1]
        dt = tk_next - tkminus1
        
        # Previous state estimate and covariance (state in physical units)
        Xkminus1 = ekf_results[i].copy()
        Pkminus1 = covariance_results[i].copy()
        
        # Step 1: Propagate state forward using CR3BP dynamics
        Xkprop = NRHOmotion(Xkminus1, tkminus1, tk_next)
        
        # Step 2: Calculate state transition matrix
        Phi = STM(Xkprop, dt)
        
        # Step 3: Time update (prediction)
        xbark = Xkprop
        
        # Predicted covariance
        Pbark = Phi @ Pkminus1 @ Phi.T + Qd
        
        # Ensure covariance matrix is symmetric
        Pbark = (Pbark + Pbark.T) / 2
        
        # Step 4: Measurement update (correction)
        Hk = np.eye(state_dim)  # Direct state measurements
        zk = XREF[i+1]  # Current measurement
        
        # Innovation (measurement residual)
        yk = zk - Hk @ xbark
        
        # Innovation covariance
        Sk = Hk @ Pbark @ Hk.T + Rk
        Sk_inv = np.linalg.inv(Sk)
        
        # Kalman gain
        Kk = Pbark @ Hk.T @ Sk_inv
        
        # Update state estimate
        Xk = xbark + Kk @ yk
        
        # Update estimate covariance using Joseph form for better numerical stability
        I = np.eye(state_dim)
        temp = I - Kk @ Hk
        Pk = temp @ Pbark @ temp.T + Kk @ Rk @ Kk.T
        
        # Ensure updated covariance remains symmetric
        Pk = (Pk + Pk.T) / 2
        
        # Store results
        ekf_results[i+1] = Xk
        covariance_results[i+1] = Pk
        residual_results[i+1] = zk - Hk @ Xk  # Post-update residual
        

    print("EKF complete!")
    return ekf_results, covariance_results, residual_results

# def run_extended_kalman_filter(XREF, tk, Rk, Qd, initial_covar):

#     import numpy as np
#     from CR3BP import NRHOmotion, STM
    
#     #set up empty array to store filter results - 'estimated state'
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
#     # Modified loop to only iterate up to the second-to-last element
#     for i in range(len(XREF) - 1):
    
#         #step 1: initialisation (a priori)
#         XREFminus1 = XREF[i] #previous reference state determined from latest measurement
#         state_dim = 3
        
#         if len(covariance_results) == 0:
#             covariance_results = np.array([0])
            
#         for j, row in enumerate(covariance_results):
#             if j == 0:
#                 Pkminus1 = np.eye(state_dim) * initial_covar
#             else:
#                 Pkminus1 = covariance_results[j-1] #previous covariance/uncertainty determined from measurement
        
#         tkminus1 = tk[i] #time at previous reference state/measurement
#         tk_next = tk[i+1] #next time read (renamed to avoid shadowing)
        
        
#         #step 2: read next observation 
#         #use CR3BP function to obtain XREFk (current reference state) and STM(tk, tk-1)
#         # Fixing function call to match definition
#         XREFk = NRHOmotion(XREFminus1, tkminus1, tk_next)
#         STMk = STM(XREFk, tk_next)
        
        
#         #step 3: time update
#         xbark = STMk @ XREFminus1 #compute state
#         Pbark = STMk @ Pkminus1 @ np.transpose(STMk) + Qd #compute covariance (this is where process noise is added)
        
        
#         #step 4: measurement update
        
#         Hk = np.eye(3) #observation mapping matrix
#         yk = XREF[i+1]
#         Kk = Pbark @ np.transpose(Hk) @ np.linalg.inv(Hk@Pbark@np.transpose(Hk)+Rk) #compute gain
#         xxk = xbark + Kk@(yk-Hk@xbark) #compute state deviation
#         Xk = XREFk + xxk #compute state
#         Pk = (np.identity(3)-Kk@Hk)@Pbark #compute covariance  # Fixed from 3 to 6 for matrix dimensions
        
        
#         #step 5: EKF update
#         #"if estimate has converged let XREFk = Xk and xk = 0"
#         if np.array_equal(XREFk, Xk):  # Updated to compare arrays properly
#             xxk = 0
            
#         filter_results[i] = Xk
#         covariance_results[i] = Pk
#         residual_results[i] = XREF[i+1] - Xk #residual
    
#     return filter_results, covariance_results, residual_results

# def run_extended_kalman_filter(XREF, tk, Rk, Qd, initial_covar):

#     import numpy as np
#     from CR3BP import NRHOmotion, STM
    
#     #set up empty array to store filter results - 'estimated state'
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
#     # Modified loop to only iterate up to the second-to-last element
#     for i in range(len(XREF) - 1):
    
#         #step 1: initialisation (a priori)
#         XREFminus1 = XREF[i] #previous reference state determined from latest measurement
#         state_dim = 3
        
#         if len(covariance_results) == 0:
#             covariance_results = np.array([0])
            
#         for j, row in enumerate(covariance_results):
#             if j == 0:
#                 Pkminus1 = np.eye(state_dim) * initial_covar
#             else:
#                 Pkminus1 = covariance_results[j-1] #previous covariance/uncertainty determined from measurement
        
#         tkminus1 = tk[i] #time at previous reference state/measurement
#         tk_next = tk[i+1] #next time read (renamed to avoid shadowing)
        
        
#         #step 2: read next observation 
#         #use CR3BP function to obtain XREFk (current reference state) and STM(tk, tk-1)
#         # Fixing function call to match definition
#         XREFk = NRHOmotion(XREFminus1, tkminus1, tk_next)
#         STMk = STM(XREFk, tk_next)
        
        
#         #step 3: time update
#         xbark = STMk @ XREFminus1 #compute state
#         Pbark = STMk @ Pkminus1 @ np.transpose(STMk) + Qd #compute covariance (this is where process noise is added)
        
        
#         #step 4: measurement update
        
#         Hk = np.eye(3) #observation mapping matrix
#         yk = XREF[i+1]
#         Kk = Pbark @ np.transpose(Hk) @ np.linalg.inv(Hk@Pbark@np.transpose(Hk)+Rk) #compute gain
#         xxk = xbark + Kk@(yk-Hk@xbark) #compute state deviation
#         Xk = XREFk + xxk #compute state
#         Pk = (np.identity(3)-Kk@Hk)@Pbark #compute covariance  # Fixed from 3 to 6 for matrix dimensions
        
        
#         #step 5: EKF update
#         #"if estimate has converged let XREFk = Xk and xk = 0"
#         if np.array_equal(XREFk, Xk):  # Updated to compare arrays properly
#             xxk = 0
            
#         filter_results[i] = Xk
#         covariance_results[i] = Pk
#         residual_results[i] = XREF[i+1] - Xk #residual
    
#     return filter_results, covariance_results, residual_results