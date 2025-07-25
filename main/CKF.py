#Cubature Kalman Filter

#more complex than EKF, less than UKF since cubature rule used over unscented transform
#more likely to perform well for the more unstable parts of the NRHO
#Cubature Kalman Filter with Improved Numerical Stability

#more complex than EKF, less than UKF since cubature rule used over unscented transform
#more likely to perform well for the more unstable parts of the NRHO
#Cubature Kalman Filter with NRHO-specific improvements

def run_cubature_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, is_it_hybrid, H_criteria, stable):
    
    import numpy as np
    import time
    from scipy import linalg
    from pointprop import point_propagation
    from Hk import compute_Hk
    
    # Set up empty arrays to store filter results
    state_dim = 6  # Should be 6 (position and velocity)
    
    ckf_results = np.zeros_like(Xtruth)
    covariance_results = np.zeros((len(Xtruth), state_dim, state_dim))
    residual_results = np.zeros((len(Xtruth), 2))
    entropy_results = np.zeros(len(Xtruth))
    
    # Initialize first state with reference measurement
    ckf_results[0] = X_initial
    
    if isinstance(initial_covar, np.ndarray) and initial_covar.shape == (6, 6):
        # initial_covar is already a full covariance matrix from previous filter
        covariance_results[0] = initial_covar
    else:
        # Do normal initialization for scalar initial_covar
        P0 = np.eye(6)
        P0[:3, :3] *= initial_covar**2
        P0[3:, 3:] *= initial_covar/1000**2
        covariance_results[0] = P0
    
    # Cubature parameters
    nx = state_dim
    xi = 0.5*np.sqrt(nx)  # Standard scaling factor for cubature points
    num_points = 2 * nx  # Total number of cubature points
    weight = 1.0 / num_points  # Equal weights for all points
    
    print(f"CKF - Starting filter with {num_points} cubature points")
    
    for k in range(len(Xtruth) - 1):
        print(f"CKF - Processing time step {k}/{len(Xtruth)-1}")
        
        if k+1 >= len(DSN_Sim_measurements):
            print(f"ERROR: Would access DSN_Sim_measurements[{k+1}] but length is {len(DSN_Sim_measurements)}")
            break
        if k+1 >= len(ground_station_state):
            print(f"ERROR: Would access ground_station_state[{k+1}] but length is {len(ground_station_state)}")
            break
        
        try:
            # Step 1: initialization (a priori)
            x_prev = ckf_results[k].copy()  # Previous estimated state
            P_prev = covariance_results[k].copy()  # Previous covariance
            
            tkminus1 = tk[k]  # Time at previous state
            tk_next = tk[k+1]  # Next time
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            # Step 2: compute square root of covariance
            try:
                # Add small regularization term for numerical stability
                regularized_P = P_prev + np.eye(nx) * 1e-8
                sqrt_P = linalg.cholesky(regularized_P, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed, using eigendecomposition")
                # Alternative: use eigendecomposition for better stability
                eigvals, eigvecs = linalg.eigh(regularized_P)
                # Ensure all eigenvalues are positive
                eigvals = np.maximum(eigvals, 1e-8)
                sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Step 3: Generate cubature points
            cubature_points = np.zeros((num_points, nx))
            
            for j in range(nx):
                # Positive points: x + xi * column of P_root
                cubature_points[j] = x_prev + xi * sqrt_P[:, j]
                # Negative points: x - xi * column of P_root
                cubature_points[j+nx] = x_prev - xi * sqrt_P[:, j]
            
            
            # Step 4: Time update - propagate cubature points
            prop_cubature_points = np.zeros_like(cubature_points)
            
            for j in range(num_points):
                # Propagate this cubature point
                try:
                    prop_cubature_points[j] = point_propagation(cubature_points[j], tkminus1, tk_next)
                    
                    # Check for non-finite values after propagation
                    if not np.all(np.isfinite(prop_cubature_points[j])):
                        raise ValueError("Propagation resulted in non-finite values")
                        
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    # Use linear propagation as fallback
                    dt = tk_next - tkminus1
                    state = cubature_points[j].copy()
                    state[:3] += state[3:] * dt
                    prop_cubature_points[j] = state
            
            # Step 5: Compute predicted state and covariance
            
            x_pred = np.zeros(6) 
            
            for j in range(num_points):
                x_pred += weight * prop_cubature_points[j]
            
            x_pred_pos = x_pred[0:3] - ground_station_state[k+1, 0:3]
            x_pred_vel = x_pred[3:6] - ground_station_state[k+1, 3:6]
            
            cub_point_measurements = np.zeros((num_points, 2)) 
            
            for j in range(num_points):
                
                position = prop_cubature_points[j, 0:3] - ground_station_state[k+1, 0:3]       
                velocity = prop_cubature_points[j, 3:6] - ground_station_state[k+1, 3:6] 
            
                prop_range = np.linalg.norm(position)
                prop_range_rate = np.dot(position, velocity) / np.linalg.norm(position) 
            
                cub_point_measurements[j] = np.array([prop_range, prop_range_rate])
            
            expected_meas = np.zeros(2)
            position = np.zeros(6)
            velocity = np.zeros(6)
            
            for j in range(num_points):
                expected_meas += weight * cub_point_measurements[j]
                
            y_pred = expected_meas
            y_pred_range, y_pred_range_rate = y_pred
            

            # Predicted covariance
            P_pred = np.zeros((nx, nx))
            for j in range(num_points):
                diff = prop_cubature_points[j] - x_pred
                P_pred += weight * np.outer(diff, diff)
            
            # Add process noise
            P_pred += Qd
            
            # Step 6: Measurement update
            yk = np.array(DSN_Sim_measurements[k+1]) # Current range and range rate measurement
            
            # Predicted measurement
            y_pred = expected_meas
            
            # Innovation
            innovation = yk - y_pred
            
            innovation_norm = np.linalg.norm(innovation)
            if innovation_norm > 5.0:  # threshold in km
                print("Large innovation detected - possible divergence")
            
            # Innovation covariance
            #Pyy = P_pred + Rk
            Pyy = np.zeros((2, 2))
            for j in range(num_points):
                diff_meas = cub_point_measurements[j] - y_pred
                Pyy += weight * np.outer(diff_meas, diff_meas)

            Pyy += Rk
        
        
            # Cross-correlation matrix
            #Pxy = P_pred  
            Pxy = np.zeros((nx, 2))  # 6x2 matrix
            for j in range(num_points):
                diff_state = prop_cubature_points[j] - x_pred
                diff_meas = cub_point_measurements[j] - y_pred
                Pxy += weight * np.outer(diff_state, diff_meas)
            
            # Kalman gain
            Kk = Pxy @ linalg.inv(Pyy)
            
            # Update state
            x_updated = x_pred + Kk @ innovation

            
            # Update covariance  
            P_updated = P_pred - Kk @ Pxy.T #RMSE = 12.15km 
            
            # Ensure covariance remains symmetric
            P_updated = (P_updated + P_updated.T) / 2
            
            #calculate entropy [important for hybrid implementation BUT may be used to compare results]
            d = P_updated.shape[0] #dimension of covariance matrix
            logval = (2*np.pi*np.e) ** d * np.linalg.det(P_updated)
            H = 0.5 * np.log(abs(logval)) 
            
            # Eigenvalue check and correction
            eigvals = linalg.eigvalsh(P_updated)
            if np.any(eigvals < 1e-8):
                print("  Warning: Covariance has negative eigenvalues, applying correction")
                eigvals, eigvecs = linalg.eigh(P_updated)
                eigvals = np.maximum(eigvals, 1e-8)
                P_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Store results
            ckf_results[k+1] = x_updated
            covariance_results[k+1] = P_updated
            residual_results[k+1] = innovation
            entropy_results[k+1] = abs(H)
            
        except Exception as exc:
            import traceback
            print(f"Error at time step {k}:")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error message: {str(exc)}")
            print("  Traceback:")
            traceback.print_exc()
            
            print(f"  Using fallback: copying previous state")
            ckf_results[k+1] = ckf_results[k]
            covariance_results[k+1] = covariance_results[k]
            residual_results[k+1] = np.zeros(2)
            
        if is_it_hybrid == 1 and abs(H) > H_criteria and stable == 1:
            print("entropy > criteria, stable region finished, swapping to unstable filter")
            return (ckf_results[:k+2], covariance_results[:k+2], 
                    residual_results[:k+2], entropy_results[:k+2])
            return ckf_results, covariance_results, residual_results, entropy_results
        
        if is_it_hybrid == 1 and abs(H) < H_criteria and stable == 0:
            print("entropy < criteria, unstable region finished, swapping to stable filter")
            return (ckf_results[:k+2], covariance_results[:k+2], 
                    residual_results[:k+2], entropy_results[:k+2])
            return ckf_results, covariance_results, residual_results, entropy_results
    
    return ckf_results, covariance_results, residual_results, entropy_results

# #Cubature Kalman Filter

# #more complex than EKF, less than UKF since cubature rule used over unscented transform
# #more likely to perform well for the more unstable parts of the NRHO
# def run_cubature_kalman_filter(XREF, tk, Rk, Qd, initial_covar):
        
#     import numpy as np
#     from CR3BP import NRHOmotion
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
#     for i, row in enumerate(XREF):
    
#         #step 1: compute weights (via unscented transform)
#         nx = 6
#         Wi = 1/(2*nx)
        
        
#         #step 2: initialisation (a priori)
#         tkminus1 = tk[i] #time at last observation
#         XREFminus1 = XREF[i] #previous reference state determined from last read
        
#         if len(covariance_results) == 0:
#             covariance_results = np.array([0])
            
#         for i, row in enumerate(covariance_results):
#             if i == 0:
#                 Pkminus1 = np.eye(nx) * initial_covar
#             else:
#                 Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from last read
        
#         tk = tk[i+1]
        
        
#         #step 3: compute cubature point matrix, Proot via Cholesky
#         Proot = np.linalg.cholesky(Pkminus1, lower=True)
#         cub_matrix = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         cub_matrix[:, 0] = XREFminus1
        
#         for i in range(nx):
#             cub_matrix[:, i] = XREFminus1 + np.sqrt(nx)*Proot[:, i]
            
#         for i in range(nx):
#             cub_matrix[:, i+nx] = XREFminus1 - np.sqrt(nx)*Proot[:, i]
        
        
#         #step 5: time update !!
#         #this is different because each sigma point (13) is propagated with the equations of motion
#         xkcub = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         for i in range(2*nx+1):
#             xkcub[:, i+1] = Wi*NRHOmotion(cub_matrix[:, i], Pkminus1, tkminus1, tk)
        

#         #step 6: process noise step
#         #sum weighted propagated points to obtain time updated state estimate
#         shape = (6,1)
#         xk = np.zeros(shape) #initialise
#         xk = np.sum(xkcub, axis=1)
        
#         Pkcub = np.zeros((nx, 2*nx+1))
#         for i in range(2*nx):
#             Pkcub[:, i] = np.dot(np.subtract(xkcub[:, i]/Wi, xk), np.transpose(np.subtract(xkcub[:, i]/Wi, xk))*Wi)
            
#         shape = (6,1)
#         Pk = np.zeros(shape) #initialise
#         Pk = np.sum(Pkcub, axis=1) + Qd
        
#         Prootnew = np.linalg.cholesky(Pk, lower=True)
#         cub_matrix_new = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         cub_matrix_new[:, 0] = xk
        
#         for i in range(nx):
#             cub_matrix_new[:, i] = xk + np.sqrt(nx)*Prootnew[:,i]
            
#         for i in range(nx):
#             cub_matrix_new[:, i+nx] = xk - np.sqrt(nx)*Prootnew[:,i]
        
        
#         #step 7: predicted measurement
#         #use the cubature rule to compute
#         h = np.eye(6) #measurement mapping matrix, identity matrix because we know the state (no need to convert from measurement data)
        
#         upsilon = np.zeros((nx, 2*nx+1)) #empty matrix   
        
#         for i in range(2*nx):
#             upsilon[:,i+1] = h @ NRHOmotion(cub_matrix_new[:, i], Pkminus1, tkminus1, tk)
            
#         shape = (6,1)
#         Ybark = np.zeros(shape) #initialise
#         Ybark = Wi*np.sum(upsilon, axis=1)
        
        
#         #step 8: innovation and cross-correlation
#         for i in range(2*nx):
#             innovation_op = np.dot(np.subtract(upsilon[:, i], Ybark), np.transpose(np.subtract(upsilon[:, i], Ybark))) #innovation
#             crosscor_op = np.dot(np.subtract(upsilon[:, i], xk), np.transpose(np.subtract(upsilon[:, i], Ybark))) #cross-correlation
        
#         Pyy = Rk + Wi*np.sum(innovation_op, axis=1) #innovation
#         Pxy = Wi*np.sum(crosscor_op, axis=1) #cross-correlation
        
        
#         #step 9: corrector update
#         Yk = XREF[i+1]        
        
#         Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
#         Xkfinal = xk + Kk @ np.subtract(Yk, Ybark) #final state estimation
#         Pkfinal = Pk - Kk @ np.transpose(Pxy) #final covariance
         
#         #repeat for next observation
#         #need to store data in an array such that the filter and the truth data can be plotted against each other
        
#         filter_results[i] = Xkfinal
#         covariance_results[i] = Pkfinal
#         residual_results[i] = XREF[i+1] - Xkfinal #residual
        
#     return filter_results, covariance_results, residual_results


