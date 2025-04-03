#Hybridisation

#note that this code will be incorporated into the interface as an option

#set boundary conditions for when filters swap
#i.e. for when the orbit dynamics become unstable
#these will be based on position data for simplicity
#(estimated region of instability)

def run_hybrid(XREF, tk, stable, unstable, bc1, bc2, Rk, Qd, c, Pcc, Qd):
    
    import numpy as np
    from EKF import run_extended_kalman_filter
    from CKF import run_cubature_kalman_filter
    from SRCKF import run_square_root_CKF
    from USKF import run_unscented_schmidt_KF
    from UKF import run_unscented_kalman_filter
    
    #note that another point of comparison can be the widening of the boundary
    #testing how changing the region increases or decreases accuracy and efficiency of the filter overall
    
    #sort state and time into stable and unstable
    
    stablepoints = []
    unstablepoints = []
    
    stabletimes = []
    unstabletimes = []
    
    for i, state in enumerate(XREF):
    # Example stability check (replace with your actual criterion)
    if (XREF[i] > bc1) & (XREF[i] < bc2):
        unstablepoints.append(i)
        unstabletimes.append(tk[i])
    else:
        stablepoints.append(i)
        stabletimes.append(tk[i])
    
    
    #set up which filters are run from interface input
    
    #stable
    stable_filter_results = {}
    
    if stable = 'EKF':
        stable_filter_results['EKF'] = run_extended_kalman_filter(stablepoints, stabletimes, Rk)
        
    if stable = 'CKF':
        stable_filter_results['CKF'] = run_cubature_kalman_filter(stablepoints, stabletimes, Rk, Qd)
        
    #unstable
    unstable_filter_results = {}
    
    if unstable = 'CKF':
        unstable_filter_results['CKF'] = run_cubature_kalman_filter(unstablepoints, unstabletimes, Rk, Qd)
        
    if unstable = 'SRCKF':
        unstable_filter_results['SRCKF'] = run_square_root_CKF(unstablepoints, unstabletimes, Rk, Qd)
        
    if unstable = 'USKF':
        unstable_filter_results['USKF'] = run_unscented_schmidt_KF(unstablepoints, unstabletimes, c, Pcc, Rk, Qd)
        
    if unstable = 'UKF':
        unstable_filter_results['UKF'] = run_unscented_kalman_filter(unstablepoints, unstabletimes, Rk, Qd)
    
    #plot
    
    

