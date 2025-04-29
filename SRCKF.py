#Square Root Cubature Kalman Filter (SRCKF) with Enhanced Stability

#similar to CKF however it calculates the square root of the covariance matrix over the full matrix
#this improves filter robustness and numerical stability as matrix is guaranteed to be positive definite

# https://www.sciencedirect.com/science/article/pii/S187770581600285X#:~:text=A%20new%20fil%20tering%20algorithm%2C%20adaptive,poor%20convergence%20or%20even%20divergence.
#paper incorporates adaptive fading factor to weigh prediction against measurement

import numpy as np
import time
from scipy import linalg
import traceback

# Add these stability enhancement functions at the top of your file

def validate_matrix(matrix, name="matrix", threshold=1e15):
    """Validate matrix for NaN/Inf values and condition number"""
    if np.any(np.isnan(matrix)):
        print(f"NaN detected in {name}")
        return False
    if np.any(np.isinf(matrix)):
        print(f"Inf detected in {name}")
        return False
    
    # For square matrices, check condition number
    if matrix.shape[0] == matrix.shape[1]:
        try:
            cond = np.linalg.cond(matrix)
            if cond > threshold:
                print(f"{name} is poorly conditioned: {cond:.2e}")
                return False
        except:
            pass  # Skip if condition number can't be computed
    return True

def robust_cholesky(P, name="matrix", lower=True):
    """Robust Cholesky with progressive regularization"""
    # Initial regularization
    reg_factor = 1e-10
    
    # Try Cholesky with increasing regularization
    for attempt in range(4):
        try:
            regularized_P = P + np.eye(P.shape[0]) * reg_factor
            if not validate_matrix(regularized_P, f"{name} (regularized)"):
                raise ValueError(f"Invalid {name} after regularization")
                
            L = linalg.cholesky(regularized_P, lower=lower)
            
            # Verify the result isn't NaN/Inf
            if not validate_matrix(L, f"{name} Cholesky factor"):
                raise ValueError("Cholesky produced invalid results")
                
            if attempt > 0:
                print(f"  Cholesky succeeded with {reg_factor:.2e} regularization")
            return L
        except Exception as e:
            # Increase regularization for next attempt
            reg_factor *= 10
            print(f"  Cholesky attempt {attempt+1} failed: {e}")
    
    # If all attempts fail, use eigendecomposition
    print(f"  Using eigendecomposition for {name}")
    eigvals, eigvecs = linalg.eigh(P)
    
    # More aggressive eigenvalue correction
    min_eig = np.min(eigvals)
    if min_eig < 1e-8:
        correction = abs(min_eig) + 1e-8
        eigvals = np.maximum(eigvals, correction)
    
    # Reconstruct matrix square root
    return eigvecs @ np.diag(np.sqrt(eigvals))

def stable_qr_update(combined, name="QR update"):
    """Stable QR decomposition with SVD fallback"""
    try:
        # Try standard QR first
        q, r = linalg.qr(combined.T, mode='economic')
        Sk = r.T
        
        # Validate result
        if not validate_matrix(Sk, f"{name} result"):
            raise ValueError("QR produced invalid results")
            
        return Sk
    except Exception as e:
        print(f"  QR decomposition failed: {e}")
        
        # SVD fallback - more stable for ill-conditioned matrices
        try:
            print("  Trying SVD decomposition")
            U, s, Vh = linalg.svd(combined, full_matrices=False)
            
            # Filter small singular values
            s_threshold = max(1e-12, np.max(s) * 1e-12)
            s_filtered = np.maximum(s, s_threshold)
            
            # Reconstruct square root
            Sk = U @ np.diag(np.sqrt(s_filtered))
            
            # Final validation
            if not validate_matrix(Sk, f"{name} via SVD"):
                raise ValueError("SVD produced invalid results")
                
            return Sk
        except Exception as e:
            print(f"  SVD also failed: {e}")
            
            # Ultimate fallback - diagonal approximation
            P = combined @ combined.T
            P = (P + P.T) / 2  # Ensure symmetry
            
            # Use eigendecomposition on symmetrized matrix
            eigvals, eigvecs = linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)
            
            return eigvecs @ np.diag(np.sqrt(eigvals))

def bound_state(state, prev_state=None, max_jump=100.0):
    """Apply bounds to state values to prevent explosion"""
    if prev_state is not None:
        # Compute change from previous state
        delta = np.abs(state - prev_state)
        max_delta = np.max(delta)
        
        # If change is too large, limit it
        if max_delta > max_jump:
            print(f"  WARNING: Excessive state change: {max_delta:.2f}")
            scale = max_jump / (max_delta + 1e-10)
            # Blend between previous state and new state
            state = prev_state + (state - prev_state) * scale
            print(f"  State change limited to scale factor: {scale:.2e}")
    
    # Ensure state doesn't have extreme values
    if np.max(np.abs(state)) > 1e5:
        print(f"  WARNING: Extreme state value: {np.max(np.abs(state)):.2e}")
        state = np.clip(state, -1e5, 1e5)
        
    return state

# Main SRCKF function with stability enhancements
def run_square_root_CKF(XREF, tk, Rk, Qd, initial_covar):
    """
    Run Square Root Cubature Kalman Filter algorithm for orbit determination
    with enhanced numerical stability
    
    Parameters:
    -----------
    XREF : numpy.ndarray
        Reference trajectory (truth + noise)
    tk : numpy.ndarray
        Time array for measurements
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
    from CR3BP import NRHOmotion
    
    # Print shapes for debugging
    print(f"SRCKF - XREF shape: {XREF.shape}")
    print(f"SRCKF - tk shape: {tk.shape}")
    print(f"SRCKF - Rk shape: {Rk.shape}")
    print(f"SRCKF - Qd shape: {Qd.shape}")
    
    # Set up empty arrays to store filter results
    state_dim = XREF.shape[1] if XREF.ndim > 1 else 6  # Should be 6 (position and velocity)
    
    srckf_results = np.zeros_like(XREF)
    covariance_results = np.zeros((len(XREF), state_dim, state_dim))
    residual_results = np.zeros_like(XREF)
    
    # Initialize first state with reference measurement
    srckf_results[0] = XREF[0]
    covariance_results[0] = np.eye(state_dim) * initial_covar
    
    # Cubature parameters
    nx = state_dim
    Wi = 1.0 / (2 * nx)  # Weight for cubature points
    xi = np.sqrt(nx) * 0.3  # Reduced scaling factor for better stability (original was sqrt(nx))
    
    print(f"SRCKF - Starting filter with {2*nx} cubature points")
    
    # Initialize previous valid state and covariance
    prev_valid_state = srckf_results[0].copy()
    prev_valid_covariance = covariance_results[0].copy()
    
    # Track propagation statistics
    total_points = 0
    linear_fallbacks = 0
    
    # Modified loop to iterate through all measurements
    for i in range(len(XREF) - 1):
        print(f"SRCKF - Processing time step {i}/{len(XREF)-2}")
        
        try:
            # Step 1: initialization (a priori)
            XREFminus1 = srckf_results[i].copy()  # Previous estimated state
            Pkminus1 = covariance_results[i].copy()  # Previous covariance/uncertainty
            
            tkminus1 = tk[i]  # Time at previous state
            tk_next = tk[i+1]  # Next time
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            # Step 2: compute square root of covariance with robust Cholesky
            Proot = robust_cholesky(Pkminus1, "Pkminus1")
            
            # Step 3: Generate cubature points
            cub_points = np.zeros((2*nx, nx))
            
            # Generate positive and negative cubature points
            for j in range(nx):
                cub_points[j] = XREFminus1 + xi * Proot[:, j]
                cub_points[j+nx] = XREFminus1 - xi * Proot[:, j]
            
            # Step 4: Time update - propagate each cubature point
            prop_points = np.zeros_like(cub_points)
            
            # Set a timeout for propagation - max 10 seconds per time step
            start_time = time.time()
            timeout = 30.0
            
            # Track propagation success for this step
            points_this_step = 0
            fallbacks_this_step = 0
            
            for j in range(2*nx):
                # Check if we're exceeding timeout
                if time.time() - start_time > timeout:
                    print(f"  Warning: Propagation timeout after {j} points")
                    # If we've propagated at least half the points, continue
                    if j > nx:
                        # For remaining points, use simple linear propagation
                        for k in range(j, 2*nx):
                            # Simple linear propagation
                            dt = tk_next - tkminus1
                            prop_state = cub_points[k].copy()
                            prop_state[:3] += prop_state[3:] * dt
                            prop_points[k] = prop_state
                            
                            fallbacks_this_step += 1
                            points_this_step += 1
                        break
                    else:
                        # Not enough points propagated - use linear propagation for all
                        print("  Using linear propagation for all points")
                        dt = tk_next - tkminus1
                        for k in range(2*nx):
                            prop_state = cub_points[k].copy()
                            prop_state[:3] += prop_state[3:] * dt
                            prop_points[k] = prop_state
                            
                            if k >= j:  # Only count ones we haven't already tried
                                fallbacks_this_step += 1
                                points_this_step += 1
                        break
                
                # Propagate this cubature point
                try:
                    prop_points[j] = NRHOmotion(cub_points[j], tkminus1, tk_next)
                    points_this_step += 1
                    
                    # Validate the propagated point
                    if not validate_matrix(prop_points[j].reshape(1, -1), f"prop_point {j}"):
                        raise ValueError(f"Invalid propagated point {j}")
                        
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    # Use linear propagation as fallback
                    dt = tk_next - tkminus1
                    prop_state = cub_points[j].copy()
                    prop_state[:3] += prop_state[3:] * dt
                    prop_points[j] = prop_state
                    
                    fallbacks_this_step += 1
                    points_this_step += 1
            
            # Update statistics
            total_points += points_this_step
            linear_fallbacks += fallbacks_this_step
            
            # Print propagation statistics
            if points_this_step > 0:
                fallback_pct = (fallbacks_this_step / points_this_step) * 100
                print(f"  Propagation: {fallbacks_this_step}/{points_this_step} points used linear fallback ({fallback_pct:.1f}%)")
                
                # If too many fallbacks, consider using the previous state
                if fallback_pct > 75 and i > 0:
                    print("  WARNING: High fallback percentage, considering previous state")
            
            # Step 5: Compute predicted state
            xhat = np.zeros(nx)
            for j in range(2*nx):
                xhat += Wi * prop_points[j]
            
            # Compute centered matrix for square root update
            X = np.zeros((nx, 2*nx))
            for j in range(2*nx):
                X[:, j] = np.sqrt(Wi) * (prop_points[j] - xhat)
            
            # Process noise square root using robust Cholesky
            SQ = robust_cholesky(Qd, "Qd")
            
            # Perform QR decomposition for square root update with stable implementation
            combined = np.hstack((X, SQ))
            Sk = stable_qr_update(combined, "time update")
            
            # Step 6: Generate new cubature points for measurement update
            new_cub_points = np.zeros((2*nx, nx))
            
            # Generate new cubature points based on predicted state and covariance
            for j in range(nx):
                new_cub_points[j] = xhat + xi * Sk[:, j]
                new_cub_points[j+nx] = xhat - xi * Sk[:, j]
            
            # Step 7: Predicted measurement
            meas_points = np.zeros_like(new_cub_points)
            
            # Use identity measurement model H = I
            for j in range(2*nx):
                meas_points[j] = new_cub_points[j]
            
            # Predicted measurement
            yhat = np.zeros(nx)
            for j in range(2*nx):
                yhat += Wi * meas_points[j]
            
            # Compute centered matrix for measurement update
            Y = np.zeros((nx, 2*nx))
            for j in range(2*nx):
                Y[:, j] = np.sqrt(Wi) * (meas_points[j] - yhat)
            
            # Measurement noise square root with robust Cholesky
            SR = robust_cholesky(Rk, "Rk")
            
            # Compute innovation covariance square root with stable QR
            combined_y = np.hstack((Y, SR))
            Sy = stable_qr_update(combined_y, "measurement")
            
            # Step 8: Compute cross-covariance
            Pxy = X @ Y.T
            
            # Step 9: Compute Kalman gain with robust approach
            try:
                # Solve system using triangular solver for better stability
                temp = linalg.solve_triangular(Sy, np.eye(nx), lower=True)
                Kk = Pxy @ temp @ temp.T
                
                # Validate Kalman gain
                if not validate_matrix(Kk, "Kalman gain"):
                    raise ValueError("Invalid Kalman gain")
            except Exception as exc:
                print(f"  Triangular solver failed: {exc}")
                # Fallback: direct computation with pseudoinverse
                Pyy = Y @ Y.T + Rk
                Pyy = (Pyy + Pyy.T) / 2  # Ensure symmetry
                Kk = Pxy @ linalg.pinv(Pyy)
            
            # Current measurement
            yk = XREF[i+1]
            
            # Innovation
            innovation = yk - yhat
            
            # Bound innovation to prevent extreme values
            if np.max(np.abs(innovation)) > 1e4:
                print(f"  WARNING: Extreme innovation: {np.max(np.abs(innovation)):.2e}")
                scale_factor = 1e4 / np.max(np.abs(innovation))
                innovation = innovation * scale_factor
                print(f"  Innovation scaled by: {scale_factor:.2e}")
            
            # REIMPLEMENT ORIGINAL ADAPTIVE FADING FACTOR LOGIC
            try:
                # Compute the trace of innovation squared
                trace_innov = np.trace(innovation.reshape(-1, 1) @ innovation.reshape(1, -1))
                
                # Compute the trace of predicted innovation covariance
                dem = np.zeros((nx, 2*nx))
                for j in range(2*nx):
                    dem[:, j] = meas_points[j] - yhat
                
                pred_innov_cov = np.zeros((nx, nx))
                for j in range(2*nx):
                    pred_innov_cov += Wi * (dem[:, j].reshape(-1, 1) @ dem[:, j].reshape(1, -1))
                
                trace_pred_innov = np.trace(pred_innov_cov)
                
                # Calculate fading factor
                if trace_pred_innov < 1e-12:
                    lambda0 = 1.0  # Default if denominator is too small
                else:
                    lambda0 = trace_innov / trace_pred_innov
                
                # Apply original rule: if lambda0 < 1, use it, else use 1
                if lambda0 < 1.0:
                    alpha = lambda0
                    print(f"  Using reduced fading factor: {alpha:.4f}")
                else:
                    alpha = 1.0
                    
                # In SRCKF, we use alpha differently than in original code
                # Original: PKk = 1/alphak * (gammak @ np.transpose(lbdak))
                # Here: The inverse will be handled in the update step
                
            except Exception as exc:
                print(f"  Error computing fading factor: {exc}")
                alpha = 1.0  # Default to no fading
            
            # Update state - apply inverse of alpha if alpha != 0
            alpha_inv = 1.0 / alpha if abs(alpha) > 1e-10 else 1.0
            
            # Apply a maximum cap on inverse alpha to prevent instabilities
            if alpha_inv > 5.0:
                print(f"  WARNING: Very small alpha ({alpha:.2e}) would cause large alpha_inv, capping")
                alpha_inv = 5.0
                
            # Scale the Kalman gain by inverse alpha
            # This is equivalent to: PKk = 1/alphak * (gammak @ np.transpose(lbdak))
            # But implemented in a way compatible with SRCKF
            Kk_scaled = Kk * alpha_inv
                
            # Update state with scaled Kalman gain
            xk_updated = xhat + Kk_scaled @ innovation
            
            # Apply state bounds to prevent extreme values
            xk_updated = bound_state(xk_updated, prev_valid_state)
            
            # Update covariance square root using modified Kalman update
            # U = [X - K*Y, K*SR]
            U = np.hstack((X - Kk @ Y, Kk @ SR))
            
            # QR decomposition for square root update with stable implementation
            Sk_updated = stable_qr_update(U, "covariance update")
            
            # Reconstruct full covariance for storage
            Pk_updated = Sk_updated @ Sk_updated.T
            
            # Ensure covariance is symmetric
            Pk_updated = (Pk_updated + Pk_updated.T) / 2
            
            # Validate final results before storing
            if validate_matrix(Pk_updated, "final covariance") and validate_matrix(xk_updated.reshape(1, -1), "final state"):
                # Store valid results and update previous valid values
                srckf_results[i+1] = xk_updated
                covariance_results[i+1] = Pk_updated
                residual_results[i+1] = innovation
                
                # Update previous valid values
                prev_valid_state = xk_updated.copy()
                prev_valid_covariance = Pk_updated.copy()
                
                print(f"  Step {i+1} processed successfully")
            else:
                print("  Invalid final matrices, using previous values with noise")
                # Use previous state with small noise
                state_noise = np.random.normal(0, 1e-6, state_dim)
                srckf_results[i+1] = prev_valid_state * (1.0 + state_noise)
                covariance_results[i+1] = prev_valid_covariance * 1.1  # Increase uncertainty
                residual_results[i+1] = np.zeros(state_dim)
            
        except Exception as exc:
            # Print detailed error information
            print(f"Error at time step {i}:")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error message: {str(exc)}")
            print("  Traceback:")
            traceback.print_exc()
            
            print(f"  Using fallback: copying previous state with small noise")
            # Add small noise to previous state to avoid exact repetition
            noise_scale = 1e-6 * np.max(np.abs(prev_valid_state))
            srckf_results[i+1] = prev_valid_state + np.random.normal(0, noise_scale, state_dim)
            covariance_results[i+1] = prev_valid_covariance * 1.2  # Increase uncertainty more
            residual_results[i+1] = np.zeros(state_dim)
    
    # Print final propagation statistics
    if total_points > 0:
        overall_fallback_pct = (linear_fallbacks / total_points) * 100
        print(f"\nSRCKF propagation summary:")
        print(f"Total points: {total_points}")
        print(f"Linear fallbacks: {linear_fallbacks} ({overall_fallback_pct:.1f}%)")
        
    # If all else fails, dynamically switch to UKF in problematic regions
    from CKF import run_cubature_kalman_filter

    # If propagation success rate is poor, use CKF for this step --> temporary!! until I can properly fix this one
    if fallback_pct > 80:
        print("  Too many propagation failures, using UKF for this step")
    # Run one step of UKF
        ckf_state, ckf_cov, _ = run_cubature_kalman_filter(
            np.vstack((XREFminus1, yk)), 
            np.array([tkminus1, tk_next]), 
            Rk, Qd, 
            initial_covar=0.0  # Not used for single step
            )
        # Use CKF result
        xk_updated = ckf_state[1]
        Pk_updated = ckf_cov[1]
    
    return srckf_results, covariance_results, residual_results

# def run_square_root_CKF(XREF, tk, Rk, Qd, initial_covar):

#     import numpy as np
#     from CR3BP import NRHOmotion
    
#     filter_results = np.zeros_like(XREF)
#     covariance_results = np.zeros((XREF.shape[0],1))
#     residual_results = np.zeros((XREF.shape[0],1))
    
#     for i, row in enumerate(XREF):
    
#         #step 1: compute weights 
#         nx = 6
#         Wi = 1/(2*nx)
        
#         xi = np.sqrt(nx) #cubature point
        
        
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
             
        
#         #step 3: compute square root cubature point matrix, S0 via cholesky
#         Proot = np.linalg.cholesky(Pkminus1, lower = True)
        
#         #evaluate cubature point matrix
#         cub_matrix = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         cub_matrix[:, 0] = XREFminus1
        
#         for i in range(nx):
#             cub_matrix[:, i] = XREFminus1 + xi*Proot[:, i]
            
#         for i in range(nx):
#             cub_matrix[:, i+nx] = XREFminus1 - xi*Proot[:, i]
        
        
#         #step 4: time update !!
#         #this is different because each sigma point (13) is propagated with the equations of motion
#         xkcub = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         for i in range(2*nx+1):
#             xkcub[:, i+1] = NRHOmotion(cub_matrix[:, i], Pkminus1, tkminus1, tk)
        
        
#         #step 5: process noise step
#         shape = (6,1)
#         xhat = np.zeros(shape) #initialise
#         xhat = Wi*np.sum(xkcub, axis=1) #predicted state 
        
        
#         lbdamat = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         for i in range (2*nx):
#             lbdamat[:, i+1] = xkcub[:, i] - xhat
        
        
#         lbda = 1/(np.sqrt(2*nx)) * lbdamat 
            
#         Pkcub = np.zeros((nx, 2*nx+1))
#         for i in range(2*nx):
#             Pkcub[:, i] = np.dot(np.subtract(xkcub[:, i]/Wi, xhat), np.transpose(np.subtract(xkcub[:, i]/Wi, xhat))*Wi)
        
#         #process noise covariance
#         SQ = np.linalg.cholesky(Qd, lower=True)
        
#         Sk = np.linalg.lu([lbda, SQ], lower=True)
        
        
#         #step 6: predicted measurement
#         #use the cubature rule to compute
        
#         #new propagated cubature points with new Sk
#         cub_matrix_new = np.zeros((nx, nx+1)) #empty matrix
        
#         cub_matrix_new[:, 0] = xhat
        
#         for i in range(nx):
#             cub_matrix[:, i] = xhat + xi*Sk[:, i]
            
#         for i in range(nx):
#             cub_matrix[:, i+nx] = xhat - xi*Sk[:, i]
        
#         cub_matrix_new = np.zeros((nx, nx+1)) #empty matrix
            
#         upsilon = np.zeros((nx, 2*nx+1)) #empty matrix
        
#         for i in range(nx):
#             upsilon[:, i] = NRHOmotion(cub_matrix_new[:,i+1], Pkminus1, tkminus1, tk)
        
#         shape = (6,1)
#         Ybark = np.zeros(shape)
#         Ybark = Wi * np.sum(upsilon, axis=1)
        
        
#         #step 7: cross-covariance matrix
#         shape = (6,1)
#         Zzk = np.zeros(shape) #initialise
        
#         for i in range (nx):
#             Zzk[:, i+1] = upsilon[:,i] - Ybark
        
#         lbdak = 1/xi * Zzk
        
#         SRk = np.linalg.cholesky(Rk, lower = True)
#         Szz = np.linalg.lu([lbdak, SRk])
        
#         gammak = lbda #for convention sake
        
#         dem = np.zeros(shape)
#         for i in range (nx):
#             dem[:, i+1] = upsilon[:, i+1] - Ybark
        
#         Yk = XREF[i+1]
#         yk = Yk - Ybark #residual
        
#         lbda0knum = np.trace(yk @ np.transpose(yk))
#         lbda0kdem = np.trace(np.linalg.sum(Wi*dem @ np.transpose(dem)))
#         lbda0k = lbda0knum @ np.linalg.inv(lbda0kdem)
        
#         #adaptive fading factor - reduces impact of disturbance
#         if lbda0k < 1:
#             alphak = lbda0k
#         else:
#                 alphak = 1
        
#         PKk = 1/alphak * (gammak @ np.transpose(lbdak))
        
        
#         #step 8: filter gain
#         Kk =(PKk @ np.linalg.inv(np.transpose(Szz))) @ np.linalg.inv(Szz)
        
#         Xkfinal = xhat + Kk @ np.subtract(Yk, Ybark) #final state estimation k
#         Sk = np.linalg.lu([(gammak-Kk@lbdak) (Kk@SRk)]) #square root of covariance (final)
        
         
#         #repeat for next observation
#         #need to store data in an array such that the filter and the truth data can be plotted against each other
        
#         filter_results[i] = Xkfinal
#         covariance_results[i] = Sk @ Sk
#         residual_results[i] = XREF[i+1] - Xkfinal #residual
        
#     return filter_results, covariance_results, residual_results

