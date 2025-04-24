#Extended Kalman Filter (EKF) Algorithm

#least computationally complex
#unlikely to be adequate for NRHO OD, however, may be useful for hybridised filters

def run_extended_kalman_filter(XREF, tk, Rk, Qd, initial_covar):
   
    import numpy as np
    from CR3BP import NRHOmotion, STM
    
    
    # Set up empty arrays to store filter results - 'estimated state'
    state_dim = 6  # Should be 6 (position and velocity)
    ekf_results = np.zeros_like(XREF)
    covariance_results = np.zeros((len(XREF), state_dim, state_dim)) #6 by 6 matrices for the length of XREF
    residual_results = np.zeros_like(XREF)
    
    # Initialize first state with reference measurement
    ekf_results[0] = XREF[0]
    covariance_results[0] = np.eye(state_dim) * initial_covar
    
    # Modified loop to iterate through all measurements
    for i in range(len(XREF) - 1):
        
        # Step 1: initialization (a priori)
        Xkminus1 = ekf_results[i].copy()  # Previous estimated state
        Pkminus1 = covariance_results[i].copy()  # Previous covariance
        
        tkminus1 = tk[i]  # Time at previous reference state/measurement
        tk_next = tk[i+1]  # Next time
        
        
        # Step 2: propagate state and get state transition matrix
        try:
            # Propagate the state to the next time step
            Xkprop = NRHOmotion(Xkminus1, tkminus1, tk_next)
            
            # Get state transition matrix
            time_diff = tk_next - tkminus1
            STMk = STM(Xkprop, time_diff)  # Pass time difference
            
            # Step 3: time update (prediction)
            xbark = STMk @ Xkminus1  # Compute predicted state
            Pbark = STMk @ Pkminus1 @ np.transpose(STMk) + Qd  # Compute predicted covariance
            
            # Step 4: measurement update (correction)
            Hk = np.eye(state_dim)  # Observation mapping matrix
            yk = XREF[i+1]  # Current measurement read
            
            # Compute Kalman gain
            S = Hk @ Pbark @ np.transpose(Hk) + Rk
            Kk = Pbark @ np.transpose(Hk) @ np.linalg.inv(S)
            
            # Update state and covariance
            innovation = yk - Hk @ xbark  # Innovation
            xxk = Kk @ innovation  # State correction
            Xk = xbark + xxk  # Compute state
            Pk = (np.eye(state_dim) - Kk @ Hk) @ Pbark  # Updated covariance
            
            #step 5: EKF update
#           #"if estimate has converged let XREFk = Xk and xk = 0"
            if np.array_equal(yk, Xk):  # Updated to compare arrays properly
                xxk = 0
            
            # Store results
            ekf_results[i+1] = Xk
            covariance_results[i+1] = Pk
            residual_results[i+1] = XREF[i+1] - ekf_results[i+1]  # Pre-update residual
            
        except Exception as exc:
            # Print detailed error information for aid in debugging
            import traceback
            traceback.print_exc()
            
            # Just copy previous state as fallback, to allow results to continue to run
            ekf_results[i+1] = ekf_results[i]
            covariance_results[i+1] = covariance_results[i]
            residual_results[i+1] = residual_results[i]
    
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