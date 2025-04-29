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
        
        # Current time and next time 
        tkminus1 = tk[i]
        tk_next = tk[i+1]
        dt = tk_next - tkminus1
        
        # Previous state estimate and covariance 
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
