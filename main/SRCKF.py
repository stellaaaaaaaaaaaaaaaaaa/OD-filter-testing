#Square Root Cubature Kalman Filter (SRCKF) with Enhanced Stability

#similar to CKF however it calculates the square root of the covariance matrix over the full matrix
#this improves filter robustness and numerical stability as matrix is guaranteed to be positive definite

# https://www.sciencedirect.com/science/article/pii/S187770581600285X#:~:text=A%20new%20fil%20tering%20algorithm%2C%20adaptive,poor%20convergence%20or%20even%20divergence.
#paper incorporates adaptive fading factor to weigh prediction against measurement
#Square Root Cubature Kalman Filter (SRCKF) with Enhanced Stability

#similar to CKF however it calculates the square root of the covariance matrix over the full matrix
#this improves filter robustness and numerical stability as matrix is guaranteed to be positive definite
def run_square_root_CKF(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, is_it_hybrid, H_criteria, stable):

    import numpy as np
    import time
    from scipy import linalg
    from pointprop import point_propagation
    
    # Set up empty arrays to store filter results
    state_dim = 6  # Should be 6 (position and velocity)
    
    srckf_results = np.zeros_like(Xtruth)
    covariance_results = np.zeros((len(Xtruth), state_dim, state_dim))
    residual_results = np.zeros((len(Xtruth), 2))
    entropy_results = np.zeros(len(Xtruth))
    
    # Initialize first state with reference measurement
    srckf_results[0] = X_initial
    
    # Initialize covariance (store full covariance for compatibility)
    covariance_results[0] = np.eye(state_dim) * initial_covar
    
    # Cubature parameters (same as your CKF)
    nx = state_dim
    xi = np.sqrt(nx)  # Standard scaling factor for cubature points
    num_points = 2 * nx  # Total number of cubature points
    weight = 1.0 / num_points  # Equal weights for all points
    
    print(f"SRCKF - Starting filter with {num_points} cubature points")
    
    for k in range(len(Xtruth) - 1):
        print(f"SRCKF - Processing time step {k}/{len(Xtruth)-2}")
        
        try:
            # Step 1: initialization (a priori) - same as CKF
            x_prev = srckf_results[k].copy()  # Previous estimated state
            P_prev = covariance_results[k].copy()  # Previous covariance
            
            tkminus1 = tk[k]  # Time at previous state
            tk_next = tk[k+1]  # Next time
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            # Step 2: compute square root of covariance - same as CKF but store sqrt
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
            
            # Step 3: Generate cubature points - exactly like CKF
            cubature_points = np.zeros((num_points, nx))
            
            for j in range(nx):
                # Positive points: x + xi * column of P_root
                cubature_points[j] = x_prev + xi * sqrt_P[:, j]
                # Negative points: x - xi * column of P_root
                cubature_points[j+nx] = x_prev - xi * sqrt_P[:, j]
            
            # Step 4: Time update - propagate cubature points (same as CKF)
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
                    # Use linear propagation as fallback (same as CKF)
                    dt = tk_next - tkminus1
                    state = cubature_points[j].copy()
                    state[:3] += state[3:] * dt
                    prop_cubature_points[j] = state
            
            # Step 5: Compute predicted state (same as CKF)
            x_pred = np.zeros(6) 
            for j in range(num_points):
                x_pred += weight * prop_cubature_points[j]
            
            # SRCKF DIFFERENCE: Compute predicted covariance square root using QR
            # Create deviation matrix
            deviation_matrix = np.zeros((nx, num_points))
            for j in range(num_points):
                deviation_matrix[:, j] = (prop_cubature_points[j] - x_pred) / np.sqrt(weight)
            
            # Add process noise square root
            try:
                SQ = linalg.cholesky(Qd, lower=True)
            except linalg.LinAlgError:
                eigvals, eigvecs = linalg.eigh(Qd)
                eigvals = np.maximum(eigvals, 1e-8)
                SQ = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # Combine and use QR decomposition for predicted square root
            combined_matrix = np.hstack([deviation_matrix, SQ])
            Q, R = linalg.qr(combined_matrix.T, mode='economic')
            S_pred = R[:nx, :nx].T  # Predicted square root covariance
            
            # Ensure lower triangular
            if np.sum(np.abs(np.triu(S_pred, 1))) > np.sum(np.abs(np.tril(S_pred, -1))):
                S_pred = S_pred.T
            
            # Step 6: Predicted measurements (same as CKF)
            cub_point_measurements = np.zeros((num_points, 2)) 
            
            for j in range(num_points):
                position = prop_cubature_points[j, 0:3] - ground_station_state[k+1, 0:3]       
                velocity = prop_cubature_points[j, 3:6] - ground_station_state[k+1, 3:6] 
                
                prop_range = np.linalg.norm(position)
                prop_range_rate = np.dot(position, velocity) / np.linalg.norm(position) 
                
                cub_point_measurements[j] = np.array([prop_range, prop_range_rate])
            
            y_pred = np.zeros(2)
            for j in range(num_points):
                y_pred += weight * cub_point_measurements[j]
            
            # Step 7: Measurement update (SRCKF version)
            yk = np.array(DSN_Sim_measurements[k+1])
            innovation = yk - y_pred
            
            # SRCKF: Create measurement deviation matrix
            nz = 2
            meas_deviation_matrix = np.zeros((nz, num_points))
            for j in range(num_points):
                meas_deviation_matrix[:, j] = (cub_point_measurements[j] - y_pred) / np.sqrt(weight)
            
            # Measurement noise square root
            try:
                SR = linalg.cholesky(Rk, lower=True)
            except linalg.LinAlgError:
                eigvals, eigvecs = linalg.eigh(Rk)
                eigvals = np.maximum(eigvals, 1e-8)
                SR = eigvecs @ np.diag(np.sqrt(eigvals))
            
            # QR decomposition for innovation covariance square root
            combined_meas_matrix = np.hstack([meas_deviation_matrix, SR])
            Q_meas, R_meas = linalg.qr(combined_meas_matrix.T, mode='economic')
            S_yy = R_meas[:nz, :nz].T  # Innovation covariance square root
            
            if np.sum(np.abs(np.triu(S_yy, 1))) > np.sum(np.abs(np.tril(S_yy, -1))):
                S_yy = S_yy.T
            
            # Cross-covariance (similar to CKF but in square root form)
            Pxy = np.zeros((nx, nz))
            for j in range(num_points):
                diff_state = prop_cubature_points[j] - x_pred
                diff_meas = cub_point_measurements[j] - y_pred
                Pxy += weight * np.outer(diff_state, diff_meas)
            
            # SRCKF Kalman gain using triangular solve
            Pyy_full = S_yy @ S_yy.T
            Kk = Pxy @ linalg.inv(Pyy_full)
            
            # Update state (same as CKF)
            x_updated = x_pred + Kk @ innovation
            
            # SRCKF: Square root covariance update using QR decomposition
            # Approximate H matrix for Joseph form
            P_pred_full = S_pred @ S_pred.T
            H_approx = Pxy.T @ linalg.inv(P_pred_full)
            IKH = np.eye(nx) - Kk @ H_approx
            
            # Joseph form in square root: [S_pred*(I-KH)' | K*SR]
            matrix1 = S_pred @ IKH.T
            matrix2 = Kk @ SR
            combined_update = np.hstack([matrix1, matrix2])
            
            # QR decomposition for updated square root
            Q_update, R_update = linalg.qr(combined_update.T, mode='economic')
            S_updated = R_update[:nx, :nx].T
            
            if np.sum(np.abs(np.triu(S_updated, 1))) > np.sum(np.abs(np.tril(S_updated, -1))):
                S_updated = S_updated.T
            
            # Convert back to full covariance for storage (compatibility)
            P_updated = S_updated @ S_updated.T
            
            # Entropy calculation using square root
            log_det_P = 2 * np.sum(np.log(np.maximum(np.diag(S_updated), 1e-20)))
            H = 0.5 * (nx * np.log(2*np.pi*np.e) + log_det_P)
            
            # Store results
            srckf_results[k+1] = x_updated
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
            srckf_results[k+1] = srckf_results[k]
            covariance_results[k+1] = covariance_results[k]
            residual_results[k+1] = np.zeros(2)
            entropy_results[k+1] = entropy_results[k] if k > 0 else 1.0
        
        # Hybrid filter logic
        if is_it_hybrid == 1 and H > H_criteria and stable == 1:
            print("entropy > criteria, stable region finished, swapping to unstable filter")
            return (srckf_results[:k+2], covariance_results[:k+2], 
                    residual_results[:k+2], entropy_results[:k+2])
        
        if is_it_hybrid == 1 and H < H_criteria and stable == 0:
            print("entropy < criteria, unstable region finished, swapping to stable filter")
            return (srckf_results[:k+2], covariance_results[:k+2], 
                    residual_results[:k+2], entropy_results[:k+2])
        
    return srckf_results, covariance_results, residual_results, entropy_results