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

def run_hybrid(XREF, tk, stable, unstable, H_criteria, Rk, Qd, c, Pcc, initial_covar):
    
    from EKF import run_extended_kalman_filter
    from CKF import run_cubature_kalman_filter
    from SRCKF import run_square_root_CKF
    from USKF import run_unscented_schmidt_KF
    from UKF import run_unscented_kalman_filter
    
    import numpy as np
    
    #note that another point of comparison can be the widening of the boundary
    #testing how changing the region increases or decreases accuracy and efficiency of the filter overall
    
    state_dim = 6  # State dimension is 6 (position and velocity)
    n_points = len(XREF)
    
    hybrid_results = np.zeros_like(XREF)
    covariance_results = np.zeros((n_points, state_dim, state_dim))
    residual_results = np.zeros_like(XREF)
    entropy_results = np.zeros(len(XREF))
    
    #start with stable region filter
    current_index = 0
    use_stable = True  
    
    while current_index < len(XREF) -1:
        
        #remaining traj
        XREF_to_go = XREF[current_index:]
        tk_to_go = tk[current_index:]
    
        if use_stable:
            if stable == 'EKF':
                results = run_extended_kalman_filter(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 1)
        
            elif stable == 'CKF':
                results = run_cubature_kalman_filter(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 1)
        
        else:
            if unstable == 'CKF':
                results = run_cubature_kalman_filter(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 0)
                
            elif unstable == 'SRCKF':
                results = run_square_root_CKF(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 0)
                
            elif unstable == 'USKF':
               results = run_unscented_schmidt_KF(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 0)
                
            elif unstable == 'UKF':
                results = run_unscented_kalman_filter(XREF_to_go, tk_to_go, Rk, Qd, initial_covar, 1, H_criteria, 0)
    
        state, covariance, residual, entropy = results
        
        if current_index == 0:
            hybrid_results.extend(state)
            covariance_results.extend(covariance)
            residual_results.extend(residual)
            entropy_results.extend(entropy)
        else:
            hybrid_results.extend(state[1:])
            covariance_results.extend(covariance[1:])
            residual_results.extend(residual[1:])
            entropy_results.extend(entropy[1:])
        
        # Update index and switch filter
        current_index += len(state) - 1
        use_stable = not use_stable  # swap filters
    
    return hybrid_results, covariance_results, residual_results, entropy_results 

