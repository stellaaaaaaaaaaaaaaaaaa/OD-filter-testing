#Extended Kalman Filter (EKF) Algorithm

#least computationally complex
#unlikely to be adequate for NRHO OD, however, may be useful for hybridised filters

def run_extended_kalman_filter(XREF, tk, Rk, Qd):

    import numpy as np
    from CR3BP import NRHOmotion, STM
    
    #set up empty array to store filter results - 'estimated state'
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    residual_results = np.zeros((XREF.shape[0],1))
    
    for i, row in enumerate(XREF):
    
        #step 1: initialisation (a priori)
        XREFminus1 = XREF[i] #previous reference state determined from latest measurement
        
        if len(covariance_results) == 0:
            covariance_results = np.array([0])
            
        for i, row in enumerate(covariance_results):
            if i == 0:
                Pkminus1 = 0
            else:
                Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from measurement
        
        tkminus1 = tk[i] #time at previous reference state/measurement
        tk = tk[i+1] #next time read
        
        
        #step 2: read next observation 
        #use CR3BP function to obtain XREFk (current reference state) and STM(tk, tk-1)
        NRHOmotion(XREFminus1, Pkminus1, tkminus1, tk)
        STM(XREFk, tk)
        
        
        #step 3: time update
        xbark = STM @ XREFminus1 #compute state
        Pbark = STM @ Pkminus1 @ np.transpose(STM) + Qd #compute covariance (this is where process noise is added)
        
        
        #step 4: measurement update
        
        Hk = np.eye(6) #observation mapping matrix
        Kk = Pbark @ np.transpose(Hk) @ np.linalg.inv(Hk@Pbark@np.transpose(Hk)+Rk) #compute gain
        xxk = xbark + Kk@(yk-Hk@xbark) #compute state deviation
        Xk = XREFk + xxk #compute state
        Pk = (np.identity(3)-Kk@Hk)@Pbark #compute covariance
        
        
        #step 5: EKF update
        #"if estimate has converged let XREFk = Xk and xk = 0"
        if XREFk == Xk:
            xxk = 0
            
        filter_results[i] = Xk
        covariance_results[i] = Pk
        residual_results[i] = XREF[i+1] - Hk@xbark #residual
    
    return filter_results, covariance_results, residual_results

