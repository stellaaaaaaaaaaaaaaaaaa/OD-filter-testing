#Unscented Schmidt Kalman Filter (USKF) Algorithm

#more complex
#more likely to perform well for the more unstable parts of the NRHO
#incorporates consider parameters - for orbit determination this may include:
    #3BP perturbations, gravitational uncertainties, SRP, ground station biases

def run_unscented_schmidt_KF(XREF, tk, c, Pcc, Rk, Qd, initial_covar):
    """
    Run Unscented Schmidt Kalman Filter algorithm for orbit determination
    
    Parameters:
    -----------
    XREF : numpy.ndarray
        Reference trajectory (truth + noise)
    tk : numpy.ndarray
        Time array for measurements
    c : numpy.ndarray
        Consider parameters (parameters not estimated but affecting dynamics)
    Pcc : numpy.ndarray
        Covariance of consider parameters
    Rk : numpy.ndarray
        Measurement noise covariance matrix
    Qd : numpy.ndarray
        Process noise covariance matrix
    initial_covar : float
        Initial covariance value
        
    Returns:
    --------
    filter_results : numpy.ndarray
        Estimated states from the filter
    covariance_results : numpy.ndarray
        Covariance matrices at each time step
    residual_results : numpy.ndarray
        Residuals (difference between measurements and estimates)
    """
    import numpy as np
    import time
    from scipy import linalg
    from CR3BP import NRHOmotion
    
    # Print shapes for debugging
    print(f"USKF - XREF shape: {XREF.shape}")
    print(f"USKF - tk shape: {tk.shape}")
    print(f"USKF - c shape: {c.shape}")
    print(f"USKF - Pcc shape: {Pcc.shape}")
    print(f"USKF - Rk shape: {Rk.shape}")
    print(f"USKF - Qd shape: {Qd.shape}")
    
    # Set up empty arrays to store filter results
    state_dim = XREF.shape[1] if XREF.ndim > 1 else 6  # Should be 6 (position and velocity)
    consider_dim = c.shape[0] if c.ndim > 0 else c.size  # Number of consider parameters
    aug_dim = state_dim + consider_dim  # Augmented state dimension
    
    uskf_results = np.zeros_like(XREF)
    covariance_results = np.zeros((len(XREF), state_dim, state_dim))
    residual_results = np.zeros_like(XREF)
    
    # Initialize first state with reference measurement
    uskf_results[0] = XREF[0]
    covariance_results[0] = np.eye(state_dim) * initial_covar
    
    # Unscented transform parameters
    alpha = 0.1  # Small positive value (determines spread of sigma points)
    beta = 2.0     # Optimal for Gaussian distributions
    kappa = 0.0    # Secondary scaling parameter
    
    lambda_param = alpha**2 * (aug_dim + kappa) - aug_dim
    gamma = np.sqrt(aug_dim + lambda_param)  # Scaling factor for sigma points
    
    # Calculate weights
    Wm = np.zeros(2*aug_dim + 1)  # Weights for mean
    Wc = np.zeros(2*aug_dim + 1)  # Weights for covariance
    
    Wm[0] = lambda_param / (aug_dim + lambda_param)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    
    for i in range(1, 2*aug_dim + 1):
        Wm[i] = 1.0 / (2 * (aug_dim + lambda_param))
        Wc[i] = Wm[i]
    
    print(f"USKF - Starting filter with {2*aug_dim + 1} sigma points")
    
    # Set up initial augmented covariance
    P_aug = np.zeros((aug_dim, aug_dim))
    P_aug[:state_dim, :state_dim] = covariance_results[0]
    P_aug[state_dim:, state_dim:] = Pcc
    
    # Modified loop to iterate through all measurements
    for i in range(len(XREF) - 1):
        print(f"USKF - Processing time step {i}/{len(XREF)-2}")
        
        try:
            # Step 1: initialization (a priori)
            x_prev = uskf_results[i].copy()  # Previous estimated state
            P_prev = covariance_results[i].copy()  # Previous state covariance
            
            tkminus1 = tk[i]  # Time at previous state
            tk_next = tk[i+1]  # Next time
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            # Step 2: Augment state with consider parameters
            z = np.zeros(aug_dim)
            z[:state_dim] = x_prev
            z[state_dim:] = c
            
            # Update augmented covariance
            P_aug[:state_dim, :state_dim] = P_prev
            
            # Step 3: Generate sigma points
            try:
                # Add small regularization term for numerical stability
                regularized_P = P_aug + np.eye(aug_dim) * 1e-10
                L = linalg.cholesky(regularized_P, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed, using eigendecomposition")
                # Alternative: use eigendecomposition for better stability
                eigvals, eigvecs = linalg.eigh(regularized_P)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
                
            # Allocate sigma points matrix
            sigma_points = np.zeros((2*aug_dim + 1, aug_dim))
            
            # Set first sigma point at the mean
            sigma_points[0] = z
            
            # Generate remaining sigma points
            for j in range(aug_dim):
                sigma_points[j+1] = z + gamma * L[:, j]
                sigma_points[j+aug_dim+1] = z - gamma * L[:, j]
            
            # Step 4: Time update - propagate sigma points
            # Only propagate the state part of each sigma point, not consider parameters
            prop_sigma_points = np.zeros_like(sigma_points)
            
            # Set a timeout for propagation - max 10 seconds per time step
            start_time = time.time()
            timeout = 10.0
            
            for j in range(2*aug_dim + 1):
                # Check if we're exceeding timeout
                if time.time() - start_time > timeout:
                    print(f"  Warning: Propagation timeout after {j} points")
                    # If we've propagated at least half the points, continue
                    if j > aug_dim:
                        # For remaining points, use simple linear propagation
                        for k in range(j, 2*aug_dim + 1):
                            # Simple linear propagation for state part
                            dt = tk_next - tkminus1
                            state_part = sigma_points[k, :state_dim].copy()
                            state_part[:3] += state_part[3:] * dt
                            
                            # Copy propagated state and original consider params
                            prop_sigma_points[k, :state_dim] = state_part
                            prop_sigma_points[k, state_dim:] = sigma_points[k, state_dim:]
                        break
                    else:
                        # Not enough points propagated - use linear propagation for all
                        print("  Using linear propagation for all points")
                        dt = tk_next - tkminus1
                        for k in range(2*aug_dim + 1):
                            state_part = sigma_points[k, :state_dim].copy()
                            state_part[:3] += state_part[3:] * dt
                            
                            prop_sigma_points[k, :state_dim] = state_part
                            prop_sigma_points[k, state_dim:] = sigma_points[k, state_dim:]
                        break
                
                try:
                    # Extract state part of sigma point
                    state_part = sigma_points[j, :state_dim]
                    
                    # Propagate state (consider params influence dynamics)
                    # In a real implementation, you would pass consider params to NRHOmotion
                    prop_state = NRHOmotion(state_part, tkminus1, tk_next)
                    
                    # Update the propagated sigma point
                    prop_sigma_points[j, :state_dim] = prop_state
                    prop_sigma_points[j, state_dim:] = sigma_points[j, state_dim:]  # Consider params unchanged
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    # Use linear propagation as fallback
                    dt = tk_next - tkminus1
                    state_part = sigma_points[j, :state_dim].copy()
                    state_part[:3] += state_part[3:] * dt
                    
                    prop_sigma_points[j, :state_dim] = state_part
                    prop_sigma_points[j, state_dim:] = sigma_points[j, state_dim:]
            
            # Step 5: Compute predicted state and covariance
            # Predicted augmented state
            z_pred = np.zeros(aug_dim)
            for j in range(2*aug_dim + 1):
                z_pred += Wm[j] * prop_sigma_points[j]
            
            # Predicted augmented covariance
            P_aug_pred = np.zeros((aug_dim, aug_dim))
            for j in range(2*aug_dim + 1):
                diff = prop_sigma_points[j] - z_pred
                P_aug_pred += Wc[j] * np.outer(diff, diff)
            
            # Add process noise to state part only
            P_aug_pred[:state_dim, :state_dim] += Qd
            
            # Step 6: Measurement prediction
            # For this implementation, we'll assume direct measurement of state (H = I)
            # Generate sigma points for measurement prediction
            try:
                # Add small regularization term for numerical stability
                regularized_Pk = P_aug_pred + np.eye(aug_dim) * 1e-10
                Lk = linalg.cholesky(regularized_Pk, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed for measurement prediction")
                # Alternative: use eigendecomposition for better stability
                eigvals, eigvecs = linalg.eigh(regularized_Pk)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-10)
                Lk = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # New sigma points based on predicted state
            meas_sigma_points = np.zeros((2*aug_dim + 1, aug_dim))
            meas_sigma_points[0] = z_pred
            
            for j in range(aug_dim):
                meas_sigma_points[j+1] = z_pred + gamma * Lk[:, j]
                meas_sigma_points[j+aug_dim+1] = z_pred - gamma * Lk[:, j]
            
            # Transform points through measurement model (identity for direct state measurement)
            meas_points = np.zeros((2*aug_dim + 1, state_dim))
            for j in range(2*aug_dim + 1):
                # Measurement is just the state part of each sigma point
                meas_points[j] = meas_sigma_points[j, :state_dim]
            
            # Predicted measurement mean
            y_pred = np.zeros(state_dim)
            for j in range(2*aug_dim + 1):
                y_pred += Wm[j] * meas_points[j]
            
            # Step 7: Innovation covariance
            Pyy = np.zeros((state_dim, state_dim))
            for j in range(2*aug_dim + 1):
                diff = meas_points[j] - y_pred
                Pyy += Wc[j] * np.outer(diff, diff)
            
            # Add measurement noise
            Pyy += Rk
            
            # Step 8: Cross-correlation matrix
            Pzy = np.zeros((aug_dim, state_dim))
            for j in range(2*aug_dim + 1):
                diff_z = meas_sigma_points[j] - z_pred
                diff_y = meas_points[j] - y_pred
                Pzy += Wc[j] * np.outer(diff_z, diff_y)
            
            # Step 9: Schmidt Kalman gain
            try:
                # Standard Kalman gain for augmented state
                K_aug = Pzy @ linalg.inv(Pyy)
            except linalg.LinAlgError:
                print("  Warning: Matrix inversion failed, using pseudo-inverse")
                K_aug = Pzy @ linalg.pinv(Pyy)
            
            # Split Kalman gain into state and consider parts
            Kx = K_aug[:state_dim, :]  # Gain for state
            Kc = K_aug[state_dim:, :]  # Gain for consider parameters
            
            # Schmidt Kalman Filter: Set consider parameter gain to zero
            Kc_schmidt = np.zeros_like(Kc)
            
            # Current measurement
            yk = XREF[i+1]
            
            # Innovation
            innovation = yk - y_pred
            
            # Update state (consider parameters remain unchanged)
            x_updated = z_pred[:state_dim] + Kx @ innovation
            
            # Schmidt-Kalman covariance update
            P_xx = P_aug_pred[:state_dim, :state_dim]
            P_xc = P_aug_pred[:state_dim, state_dim:]
            P_cx = P_aug_pred[state_dim:, :state_dim]
            
            # Covariance update for state part only
            P_updated = P_xx - Kx @ Pyy @ Kx.T - Kx @ Pyy @ Kc_schmidt.T - Kc_schmidt @ Pyy @ Kx.T
            
            # Consider correlation update
            P_xc_updated = P_xc - Kx @ Pyy @ Kc_schmidt.T
            
            # Ensure covariance remains symmetric and positive definite
            P_updated = (P_updated + P_updated.T) / 2  # Ensure symmetry
            
            # Eigenvalue check and correction
            eigvals = linalg.eigvalsh(P_updated)
            if np.any(eigvals < 1e-10):
                print("  Warning: Covariance has negative eigenvalues, applying correction")
                eigvals, eigvecs = linalg.eigh(P_updated)
                eigvals = np.maximum(eigvals, 1e-10)
                P_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Store results
            uskf_results[i+1] = x_updated
            covariance_results[i+1] = P_updated
            residual_results[i+1] = innovation
            
            # Update augmented covariance for next step
            P_aug[:state_dim, :state_dim] = P_updated
            P_aug[:state_dim, state_dim:] = P_xc_updated
            P_aug[state_dim:, :state_dim] = P_xc_updated.T
            
        except Exception as exc:
            # Print detailed error information
            import traceback
            print(f"Error at time step {i}:")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error message: {str(exc)}")
            print("  Traceback:")
            traceback.print_exc()
            
            print(f"  Using fallback: copying previous state")
            # Just copy previous state as fallback
            uskf_results[i+1] = uskf_results[i]
            covariance_results[i+1] = covariance_results[i]
            residual_results[i+1] = np.zeros(state_dim)
    
    return uskf_results, covariance_results, residual_results

# #Unscented Schmidt Kalman Filter (USKF) Algorithm

# #more complex
# #more likely to perform well for the more unstable parts of the NRHO
# #incorporates consider parameters - for orbit determination this may include:
#     #3BP perturbations, gravitational uncertainties, SRP, ground station biases

# def run_unscented_schmidt_KF(XREF, tk, c, Pcc, Rk, Qd, initial_covar):
        
#     import numpy as np
#     from CR3BP import NRHOmotion
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
#     for i, row in enumerate(XREF):
    
#         #step 1: compute weights (via unscented transform)
#         nx = 6
#         nc = 4 #assuming using all consider parameters aforementioned
#         L = (nx+nc)
#         gamma = np.sqrt(L)
#         Wi = 1/(2*(nx+nc))
        
        
#         #step 2: initialisation (a priori)
#         tkminus1 = tk[i] #time at last observation
#         XREFminus1 = XREF[i] #previous reference state determined from last read
        
#         if len(covariance_results) == 0:
#             covariance_results = np.array([0])
            
#         for i, row in enumerate(covariance_results):
#             if i == 0:
#                 Pkminus1 = np.eye(L) * initial_covar
#             else:
#                 Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from last read
        
#         tk = tk[i+1]
#         Pzz = Pkminus1
        
#         #step 3: combine XREFk and consider parameter vector to get 'z'
#         z = np.vstack((XREFminus1, c))
        
#         #measurement mapping matrices
#         #hx, hc, assume no change, identity matrices
       
        
#         #step 4: compute sigma point matrix, Szz via Cholesky
#         Szz = np.linalg.cholesky(Pzz, lower=True)
#         sigma_matrix = np.zeros((L, 2*L+1)) #empty matrix
        
#         sigma_matrix[:, 0] = z #first sigma point
        
#         for i in range(L):
#             sigma_matrix[:, i+1] = z + gamma*Szz[:, i]
            
#         for i in range(L):
#             sigma_matrix[:, i+L+1] = z - gamma*Szz[:, i]
        
        
#         #step 5: time update !! 
#         #this is different because each sigma point (13) is propagated with the equations of motion
#         Zk = np.zeros((L, 2*L+1)) #empty matrix
#         for i in range(2*L+1):
#             Zk[:, i+1] = NRHOmotion(sigma_matrix[:, i], Pkminus1, tkminus1, tk, Rk)
            
        
#         #step 6: process noise step
#         #compute a priori state xk and covariance Pk for tk
#         zksigma = np.zeros((L, 2*L+1))
#         for i in range(2*L):
#             zksigma = Wi*Zk[:,i]
        
#         shape = (6,1)
#         Zbark = np.zeros(shape) #initialise
#         Zbark = np.sum(zksigma, axis=1)
        
#         #process noise covariance
        
#         Pzzksigma = np.zeros((L, 2*L+1))
#         for i in range(2*L):
#             Pzzksigma[:, i] = Wi*np.dot(np.subtract(sigma_matrix[:, i], Zbark), np.transpose(np.subtract(sigma_matrix[:, i], Zbark)))
            
#         shape = (6,1)
#         Pzzk = np.zeros(shape) #initialise
#         Pzzk = np.sum(Pzzksigma, axis=1) + Qd
        
#         Szznew = np.linalg.cholesky(Pzzk, lower=True)
#         sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
#         sigma_matrix_new[:, 0] = Zbark       
        
#         for i in range(L):
#             sigma_matrix_new[:, i+1] = Zbark + gamma*Szznew[:, i]
            
#         for i in range(L):
#             sigma_matrix_new[:, i+L+1] = Zbark - gamma*Szznew[:, i]
        
        
#         #step 7: predicted measurement
#         #use the UT to compute
        
#         upsilon = np.zeros((L, 2*L+1)) #empty matrix
#         for i in range(L):
#             upsilon[:, i+1] = NRHOmotion(sigma_matrix_new[:, i], Pkminus1, tkminus1, tk)
        
#         Ybarksigma = np.zeros((L, 2*L+1))
        
#         for i in range(2*L):
#             Ybarksigma[:,i+1] = Wi*upsilon[:, i]
            
#         shape = (6,1)
#         Ybark = np.zeros(shape) #initialise
#         Ybark = np.sum(Ybarksigma, axis=1)
        
        
#         #step 8: innovation and cross-correlation
        
#         innovation_op = np.zeros((L, 2*L+1))
#         crosscor_op = np.zeros((L, 2*L+1))
        
#         for i in range(2*L):
#             innovation_op[:, i] = Wi*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
#             crosscor_op[:, i] = Wi*np.dot(np.subtract(sigma_matrix[:,i], Zbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            
#             Pyy = Rk + np.sum(innovation_op, axis=1) #innovation
#             Pzy = np.sum(crosscor_op, axis=1) #cross-correlation
    
        
#         #step 9: corrector update
#         Kz = Pzy @ np.linalg.inv(Pyy) #Kalman gain
        
#         Kx = np.array([[Kz[0,0]],
#                        [Kz[0,1]],
#                        [Kz[0,2]],
#                        [Kz[0,3]],
#                        [Kz[0,4]],
#                        [Kz[0,5]]])
        
#         Kc = np.array([[Kz[0,6]],
#                        [Kz[0,7]],
#                        [Kz[0,8]],
#                        [Kz[0,9]]])
        
#         KX = np.array([[Kx],
#                        [0]])
        
#         Yk = XREF[i+1]
#         Zkfinal = Zbark + KX @ np.subtract(Yk, Ybark) #final state estimation
        
#         Kc = Kz[0,1]
#         uncert = np.array([[Kx @ Pyy @ np.transpose(Kx), Kx @ Pyy @ np.transpose(Kc)],
#                           [Kc @ Pyy @ np.transpose(Kx), 0]])
        
#         Pzzk = Pzz - uncert #final covariance
         
#         #repeat for next observation
#         #need to store data in an array such that the filter and the truth data can be plotted against each other
        
#         filter_results[i] = Zkfinal
#         covariance_results[i] = Pzzk
#         residual_results[i] = XREF[i+1] - Zkfinal #residual
        
        
#     return filter_results, covariance_results, residual_results
        
