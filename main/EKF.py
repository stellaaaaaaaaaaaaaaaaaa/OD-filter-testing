# Extended Kalman Filter (EKF) Algorithm - Improved Implementation

def run_extended_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, is_it_hybrid, H_criteria, stable):

    import numpy as np
    from pointprop import point_propagation, STM_from_states
    from Hk import compute_Hk
    
    # Set up empty arrays to store filter results
    state_dim = 6  # State dimension is 6 (position and velocity)
    n_points = len(Xtruth)
    
    ekf_results = np.zeros_like(Xtruth)
    covariance_results = np.zeros((n_points, state_dim, state_dim))
    residual_results = np.zeros((n_points, 2))
    entropy_results = np.zeros(len(Xtruth))

    # Initialize first state with reference measurement
    ekf_results[0] = X_initial
    
    if isinstance(initial_covar, np.ndarray) and initial_covar.shape == (6, 6):
    # initial_covar is already a full covariance matrix from previous filter
        covariance_results[0] = initial_covar
    else:
    # Do normal initialization for scalar initial_covar
        P0 = np.eye(6)
        P0[:3, :3] *= initial_covar**2
        P0[3:, 3:] *= initial_covar/1000**2
        covariance_results[0] = P0
    
    
    # Progress tracking
    progress_interval = max(1, n_points // 20)
    
    for i in range(n_points - 1):
        
        if i % progress_interval == 0:
            print(f"EKF progress: {i}/{n_points-1} steps ({i/(n_points-1)*100:.1f}%)")
        
        # Current time and next time 
        tkminus1 = tk[i]
        tk_next = tk[i+1]
        dt = tk_next - tkminus1
        
        # Previous state estimate and covariance 
        Xkminus1 = ekf_results[i].copy()
        Pkminus1 = covariance_results[i].copy()
        
        # Step 1: Propagate state + calculate range and range rate
        Xkprop = point_propagation(Xkminus1, tkminus1, tk_next)
        
        position = Xkprop[0:3] - ground_station_state[i+1, 0:3]       
        velocity = Xkprop[3:6] - ground_station_state[i+1, 3:6] 
        
        Xk_range = np.linalg.norm(position)
        Xk_range_rate = np.dot(position, velocity) / np.linalg.norm(position) 
        
        expected_meas = np.array([Xk_range, Xk_range_rate])
        
        # Step 2: Calculate state transition matrix
        
        Phi = STM_from_states(Xkminus1, Xkprop, tkminus1, tk_next)
        condition_number = np.linalg.cond(Phi)
        print(f"condition number = {condition_number}")
        
        # Step 3: Time update (prediction)
        xbark = Xkprop
        
        # Predicted covariance
        Pbark = Phi @ Pkminus1 @ Phi.T + Qd
        
        # Ensure covariance matrix is symmetric
        Pbark = (Pbark + Pbark.T) / 2
        
        # Step 4: Measurement update (correction)
        Hk = compute_Hk(position, velocity, Xk_range, Xk_range_rate)  # 2 by 6 matrix, Jacobian using state prediction xbark and station state
        zk = np.array(DSN_Sim_measurements[i+1]) # Current range and range rate measurement
        
        # Innovation (measurement residual)
        yk = zk - expected_meas
        
        # Innovation covariance
        Sk = Hk @ Pbark @ Hk.T + Rk
        Sk_inv = np.linalg.inv(Sk)
        
        # Kalman gain
        Kk = Pbark @ Hk.T @ Sk_inv
        
        # Update state estimate
        correction = Kk @ yk #limit to combat rising errors towards perilune
        # max_cor = 10 #km
        
        # if np.linalg.norm(correction[:3]) > max_cor:
        #     correction = correction * max_cor / np.linalg.norm(correction[:3])
        #     Xk = xbark + correction
        
        Xk = xbark + correction
        
        # Update estimate covariance using Joseph form for better numerical stability
        I = np.eye(state_dim)
        temp = I - Kk @ Hk
        Pk = temp @ Pbark @ temp.T + Kk @ Rk @ Kk.T
        
        # Ensure updated covariance remains symmetric
        Pk = (Pk + Pk.T) / 2
        
        #calculate entropy [important for hybrid implementation BUT may be used to compare results]
        d = Pk.shape[0] #dimension of covariance matrix
        logval = (2*np.pi*np.e) ** d * np.linalg.det(Pk)
        H = 0.5 * np.log(abs(logval)) 
        
        # Store results
        ekf_results[i+1] = Xk
        covariance_results[i+1] = Pk
        residual_results[i+1] = yk # Post-update residual
        entropy_results[i+1] = abs(H)
        
        if is_it_hybrid == 1 and abs(H) > H_criteria and stable == 1:
             print("entropy > criteria, stable region finished, swapping to unstable filter")
             return (ekf_results[:i+2], covariance_results[:i+2], 
                     residual_results[:i+2], entropy_results[:i+2])
             return ekf_results, covariance_results, residual_results, entropy_results
            

    print("EKF complete!")
    return ekf_results, covariance_results, residual_results, entropy_results