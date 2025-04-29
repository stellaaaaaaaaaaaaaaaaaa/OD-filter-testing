#Unscented Kalman Filter (UKF) Algorithm

#more complex
#more likely to perform well for the more unstable parts of the NRHO

def run_unscented_kalman_filter(XREF, tk, Rk, Qd, initial_covar):
  
    import numpy as np
    import time
    from scipy import linalg
    from CR3BP import NRHOmotion
    

    # Set up empty arrays to store filter results
    state_dim = 6  # Should be 6 (position and velocity)
    
    ukf_results = np.zeros_like(XREF)
    covariance_results = np.zeros((len(XREF), state_dim, state_dim))
    residual_results = np.zeros_like(XREF)
    
    # Initialize first state with reference measurement
    ukf_results[0] = XREF[0]
    covariance_results[0] = np.eye(state_dim) * initial_covar
    
    # Unscented transform parameters
    L = state_dim
    alpha = 0.1  # Small positive value (determines spread of sigma points)
    beta = 2.0   # Optimal for Gaussian distributions
    kappa = 3.0 - L  # Secondary scaling parameter
    
    lambda_param = (alpha**2) * (L + kappa) - L
    gamma = np.sqrt(L + lambda_param)  # Scaling factor for sigma points
    
    # Calculate weights
    W0m = lambda_param / (L + lambda_param)
    W0c = W0m + (1 - alpha**2 + beta)
    
    Wim = 1.0 / (2.0 * (L + lambda_param))
    Wic = Wim
    
    Wm = np.ones(2*L + 1) * Wim
    Wm[0] = W0m
    
    Wc = np.ones(2*L + 1) * Wic
    Wc[0] = W0c
    
    # Modified loop to iterate through all measurements
    for k in range(len(XREF) - 1):
        print(f"UKF - Processing time step {k}/{len(XREF)-2}")
        
        try:
            # Step 1: initialization (a priori)
            x_prev = ukf_results[k].copy()  # Previous estimated state
            P_prev = covariance_results[k].copy()  # Previous covariance
            
            tkminus1 = tk[k]  # Time at previous state
            tk_next = tk[k+1]  # Next time
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            # compute square root of covariance via Cholesky decomposition
            try:
                # Add small regularization term for stability
                regularised_P = P_prev + np.eye(L) * 1e-8
                sqrt_P = linalg.cholesky(regularised_P, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed, using eigendecomposition")
                # use eigendecomposition for stability
                eigvals, eigvecs = linalg.eigh(regularised_P)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)  # Increased minimum eigenvalue
                sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Step 2: Generate sigma points
            sigma_points = np.zeros((2*L + 1, L))
            
            sigma_points[0] = x_prev
            for j in range(L):
                sigma_points[j+1] = x_prev + gamma * sqrt_P[:, j]
                sigma_points[j+L+1] = x_prev - gamma * sqrt_P[:, j]
            
            # Step 4: Time update - propagate sigma points
            prop_sigma_points = np.zeros_like(sigma_points)
            
            for j in range(2*L + 1):
                try:
                    prop_sigma_points[j] = NRHOmotion(sigma_points[j], tkminus1, tk_next)
                    
                    # Check for non-finite values after propagation
                    if not np.all(np.isfinite(prop_sigma_points[j])):
                        raise ValueError("Propagation resulted in non-finite values")
                        
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    # Use linear propagation as fallback
                    dt = tk_next - tkminus1
                    state = sigma_points[j].copy()
                    state[:3] += state[3:] * dt
                    prop_sigma_points[j] = state
            
            # Step 5: Compute predicted state and covariance
            # Predicted state is weighted average of propagated points
            x_pred = np.zeros(L)
            for j in range(2*L + 1):
                x_pred += Wm[j] * prop_sigma_points[j]
            
            # Predicted covariance
            P_pred = np.zeros((L, L))
            for j in range(2*L + 1):
                diff = prop_sigma_points[j] - x_pred
                P_pred += Wc[j] * np.outer(diff, diff)
            
            # Add process noise
            P_pred += Qd
            
            # Step 6: Generate new sigma points
            try:
                # Add small regularisation term for numerical stability
                regularised_Pk = P_pred + np.eye(L) * 1e-8  
                sqrt_Pk = linalg.cholesky(regularised_Pk, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed for predicted covariance")
                # Alternative: use eigendecomposition for better stability
                eigvals, eigvecs = linalg.eigh(regularised_Pk)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)  # Increased minimum eigenvalue
                sqrt_Pk = eigvecs @ np.diag(np.sqrt(eigvals))
            
            #generate new sigma points
            new_sigma_points = np.zeros((2*L + 1, L))
            new_sigma_points[0] = x_pred
            
            for j in range(L):
                new_sigma_points[j+1] = x_pred + gamma * sqrt_Pk[:, j]
                new_sigma_points[j+L+1] = x_pred - gamma * sqrt_Pk[:, j]
            
            # Step 7: Measurement prediction
            # For direct state measurement, measurement function is identity
            meas_points = np.zeros_like(new_sigma_points)
            
            for j in range(2*L + 1):
                # Measurement is the same as state (identity measurement model)
                meas_points[j] = new_sigma_points[j]
            
            # Predicted measurement mean
            y_pred = np.zeros(L)
            for j in range(2*L + 1):
                y_pred += Wm[j] * meas_points[j]
            
            # Step 8: Innovation covariance and cross-correlation
            Pyy = np.zeros((L, L))
            Pxy = np.zeros((L, L))
            
            for j in range(2*L + 1):
                diff_y = meas_points[j] - y_pred
                diff_x = new_sigma_points[j] - x_pred
                
                Pyy += Wc[j] * np.outer(diff_y, diff_y)
                Pxy += Wc[j] * np.outer(diff_x, diff_y)
            
            # Add measurement noise
            Pyy += Rk
            
            # Step 9: Kalman gain and state update
            Kk = Pxy @ linalg.inv(Pyy)

            # Current measurement
            yk = XREF[k+1]
            
            # Innovation
            innovation = yk - y_pred
            
            # final state
            x_updated = x_pred + Kk @ innovation
                
            P_updated = P_pred - Kk @ Pyy @ Kk.T
            
            # Eigenvalue check and correction
            eigvals = linalg.eigvalsh(P_updated)
            if np.any(eigvals < 1e-8):  
                print("  Warning: Covariance has negative eigenvalues, applying correction")
                eigvals, eigvecs = linalg.eigh(P_updated)
                eigvals = np.maximum(eigvals, 1e-8)  # Increased minimum eigenvalue
                P_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Limit maximum covariance values
            max_cov_value = 1e6  # Adjust based on your system scale
            P_updated = np.minimum(P_updated, max_cov_value)
            
            ukf_results[k+1] = x_updated
            covariance_results[k+1] = P_updated
            residual_results[k+1] = innovation
            
        except Exception as exc:
            import traceback
            print(f"Error at time step {k}:")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error message: {str(exc)}")
            print("  Traceback:")
            traceback.print_exc()
            
            print(f"  Using fallback: copying previous state")
            ukf_results[k+1] = ukf_results[k]
            covariance_results[k+1] = covariance_results[k]
            residual_results[k+1] = np.zeros(state_dim)
    
    return ukf_results, covariance_results, residual_results

# #Unscented Kalman Filter (UKF) Algorithm

# #more complex
# #more likely to perform well for the more unstable parts of the NRHO

# def run_unscented_kalman_filter(XREF, tk, Rk, Qd, initial_covar):

#     import numpy as np
#     from CR3BP import NRHOmotion
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
    
#     for i, row in enumerate(XREF):
    
#         #step 1: compute weights (via unscented transform)
#         L = 6 #number of states, 3 position, 3 velocity
        
#         #tuning parameters for Gaussian initial PDF
#         alpha = 1 #determines spread of sigma points
#         beta = 2
#         kappa = 3 - L
        
#         lbda = (alpha**2)*(L+kappa) - L
#         gamma = np.sqrt(L + lbda)
        
#         W0m = lbda/(L+lbda)
#         W0c = lbda/(L+lbda) + 1 - alpha**2 + beta
        
#         Wim = 1/(2*(L+lbda))
#         Wic = Wim
        
        
#         #step 2: initialisation (a priori)
#         XREFminus1 = XREF[i] #previous reference state determined from last read
        
#         if len(covariance_results) == 0:
#             covariance_results = np.array([0])
            
#         for i, row in enumerate(covariance_results):
#             if i == 0:
#                 Pkminus1 = np.eye(L) * initial_covar
#             else:
#                 Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from last read
        
#         tkminus1 = tk[i]
#         tk = tk[i+1]
 
        
#         #step 3: compute sigma point matrix, Proot via Cholesky
#         Proot = np.linalg.cholesky(Pkminus1, lower=True)
#         sigma_matrix = np.zeros((L, 2*L+1)) #empty matrix
        
#         sigma_matrix[:, 0] = XREFminus1 #first sigma point
        
#         for i in range(L):
#             sigma_matrix[:, i+1] = XREFminus1 + gamma*Proot[:, i] #mean + scaled root of covariance
            
#         for i in range(L):
#             sigma_matrix[:, i+L+1] = XREFminus1 - gamma*Proot[:, i] #mean - scaled root of covariance
        
        
#         #step 4: time update !!
#         #this is different because each sigma point (13) is propagated with the equations of motion
#         xbark = np.zeros((L, 2*L+1)) #empty matrix
#         for i in range(2*L+1):
#             xbark[:, i+1] = NRHOmotion(sigma_matrix[:, i], Pkminus1, tkminus1, tk)
            
        
#         #step 5: process noise step
#         #compute a state xk and covariance Pk for tk
#         Xksigma = np.zeros((L, 2*L+1))
#         for i in range(2*L+1):
#             if i == 0:
#                 Xksigma[:,i] = W0m*xbark[:,i]
#             else:
#                 Xksigma[:,i] = Wim*xbark[:,i]
        
#         shape = (6,1)
#         Xbark = np.zeros(shape) #initialise
#         Xbark = np.sum(Xksigma, axis=1)
        
#         #process noise covariance
        
#         Pksigma = np.zeros((L, 2*L+1))
#         for i in range(2*L+1):
#             if i == 0:
#                 Pksigma[:, i] = W0c*np.dot(np.subtract(xbark[:,i], Xbark), np.transpose(np.subtract(xbark[:,i], Xbark)))
#             else:
#                 Pksigma[:, i] = Wic*np.dot(np.subtract(xbark[:,i], Xbark), np.transpose(np.subtract(xbark[:,i], Xbark)))
            
#         shape = (6,1)
#         Pbark = np.zeros(shape) #initialise
#         Pbark = np.sum(Pksigma, axis=1) + Qd
        
#         Prootnew = np.linalg.cholesky(Pbark, lower=True)
#         sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
#         sigma_matrix_new[:, 0] = Xbark #first sigma point
        
#         for i in range(L):
#             sigma_matrix_new[:, i+1] = Xbark + gamma*Prootnew[:, i] #mean + scaled root of covariance
            
#         for i in range(L):
#             sigma_matrix_new[:, i+L+1] = Xbark - gamma*Prootnew[:, i] #mean - scaled root of covariance
                
        
#         #step 6: predicted measurement
#         #use the UT to compute
#         upsilon = np.zeros((L, 2*L+1)) #empty matrix
#         for i in range(2*L+1):
#             upsilon[:, i+1] = NRHOmotion(sigma_matrix_new[:, i], Pkminus1, tkminus1, tk)
        
#         Ybarksigma = np.zeros((L, 2*L+1))
        
#         for i in range(2*L+1):
#             if i == 0:
#                 Ybarksigma[:,i] = W0m*upsilon[:, i]
#             else:
#                 Ybarksigma[:,i] = Wim*upsilon[:, i]
            
#         shape = (6,1)
#         Ybark = np.zeros(shape) #initialise
#         Ybark = np.sum(Ybarksigma, axis=1)
        
        
#         #step 7: innovation and cross-correlation
        
#         innovation_op = np.zeros((L, 2*L+1))
#         crosscor_op = np.zeros((L, 2*L+1))
        
#         for i in range(2*L+1):
#             if i == 0: 
#                 innovation_op[:, i] = W0c*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
#                 crosscor_op[:, i] = W0c*np.dot(np.subtract(sigma_matrix[:,i], Xbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
#             else:
#                 innovation_op[:, i] = Wic*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
#                 crosscor_op[:, i] = Wic*np.dot(np.subtract(sigma_matrix[:,i], Xbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            
#             Pyy = Rk + np.sum(innovation_op, axis=1) #innovation
#             Pxy = np.sum(crosscor_op, axis=1) #cross-correlation
        
        
#         #step 8: corrector update
#         Yk = XREF[i+1]
#         Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
#         Xk = Xbark + Kk @ np.subtract(Yk, Ybark) #final state estimation
#         Pk = Pbark - Kk @ Pyy @ np.transpose(Kk) #final covariance
         
#         #repeat for next observation
#         #need to store data in an array such that the filter and the truth data can be plotted against each other
        
#         filter_results[i] = Xk
#         covariance_results[i] = Pk
#         residual_results[i] = XREF[i+1] - Xk #residual
    
#     return filter_results, covariance_results, residual_results
        
