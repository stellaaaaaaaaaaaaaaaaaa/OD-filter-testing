#Hybridisation

#note that this code will be incorporated into the interface as an option

#set boundary conditions for when filters swap
#i.e. for when the orbit dynamics become unstable
#these will be based on position data for simplicity
#(estimated region of instability)
#use differential entropy to swap between filters?
# https://amostech.com/TechnicalPapers/2022/Poster/Chow.pdf
# H = 0.5 * log|2*pi*e*P|
# P = state covariance at each state

def run_hybrid(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, stable, unstable, H_criteria, c, Pcc, Qd, Rk, initial_covar):
    
    from EKF import run_extended_kalman_filter
    from CKF import run_cubature_kalman_filter
    from SRCKF import run_square_root_CKF
    from USKF import run_unscented_schmidt_KF
    from UKF import run_unscented_kalman_filter
    
    import numpy as np
    
    hybrid_results = []
    covariance_results = []
    residual_results = []
    entropy_results = []
    
    # Start with stable region filter
    current_index = 0
    use_stable = True
    
    # Keep track of current state and covariance for continuation
    current_state = np.array(X_initial).copy()
    current_covar = np.array(initial_covar).copy()
    
    while current_index < len(Xtruth) - 1:
        
        # Slice trajectories from current position
        Xtruth_remaining = Xtruth[current_index:]
        DSN_Sim_measurements_remaining = DSN_Sim_measurements[current_index:]
        tk_remaining = tk[current_index:]
        
        if use_stable:
            if stable == 'EKF':
                results = run_extended_kalman_filter(
                    Xtruth_remaining, current_state, DSN_Sim_measurements_remaining, 
                    ground_station_state, tk_remaining, Rk, Qd, current_covar, 
                    1, H_criteria, 1
                )
            elif stable == 'CKF':
                results = run_cubature_kalman_filter(
                    Xtruth_remaining, current_state, DSN_Sim_measurements_remaining, 
                    ground_station_state, tk_remaining, Rk, Qd, current_covar, 
                    1, H_criteria, 1
                )
        else:
            if unstable == 'CKF':
                results = run_cubature_kalman_filter(
                    Xtruth_remaining, current_state, DSN_Sim_measurements_remaining, 
                    ground_station_state, tk_remaining, Rk, Qd, current_covar, 
                    1, H_criteria, 0
                )
            elif unstable == 'SRCKF':
                results = run_square_root_CKF(
                    Xtruth_remaining, current_state, DSN_Sim_measurements_remaining, 
                    ground_station_state, tk_remaining, Rk, Qd, current_covar, 
                    1, H_criteria, 0
                )
            elif unstable == 'USKF':
                results = run_unscented_schmidt_KF(
                    DSN_Sim_measurements_remaining, tk_remaining, Rk, Qd, 
                    current_covar, 1, H_criteria, 0
                )
            elif unstable == 'UKF':
                results = run_unscented_kalman_filter(
                    Xtruth_remaining, current_state, DSN_Sim_measurements_remaining, 
                    ground_station_state, tk_remaining, Rk, Qd, current_covar, 
                    1, H_criteria, 0
                )
        
        state, covariance, residual, entropy = results
        
        # Append results (skip first point if not the first iteration to avoid duplication)
        if current_index == 0:
            hybrid_results.extend(state.tolist())
            covariance_results.extend(covariance.tolist())
            residual_results.extend(residual.tolist())
            entropy_results.extend(entropy.tolist())
        else:
            hybrid_results.extend(state[1:].tolist())
            covariance_results.extend(covariance[1:].tolist())
            residual_results.extend(residual[1:].tolist())
            entropy_results.extend(entropy[1:].tolist())
        
        # Update current state and covariance for next iteration
        current_state = np.array(state[-1]).copy()  # Last state from this filter run
        current_covar = np.array(covariance[-1]).copy()  # Last covariance from this filter run
        
        # Update index to continue from where this filter left off
        current_index += len(state) - 1
        
        # Switch filter for next iteration
        use_stable = not use_stable
        
        # Safety check to prevent infinite loops
        if current_index >= len(Xtruth) - 1:
            break
    
    # Convert to numpy arrays after the loop is complete
    hybrid_results = np.array(hybrid_results)
    covariance_results = np.array(covariance_results)
    residual_results = np.array(residual_results)
    entropy_results = np.array(entropy_results)
    
    return hybrid_results, covariance_results, residual_results, entropy_results