#Cubature Kalman Filter

#more complex than EKF, less than UKF since cubature rule used over unscented transform
#more likely to perform well for the more unstable parts of the NRHO
def run_cubature_kalman_filter(XREF, tk, Rk, Qd):
        
    import numpy as np
    from CR3BP import NRHOmotion
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    
    for i, row in enumerate(XREF):
    
        #step 1: compute weights (via unscented transform)
        nx = 6
        Wi = 1/(2*nx)
        
        #step 2: initialisation (a priori)
        tkminus1 = tk[i] #time at last observation
        XREFminus1 = XREF[i] #previous reference state determined from last read
        
        if len(covariance_results) == 0:
            covariance_results = np.array([0])
            
        for i, row in enumerate(covariance_results):
            if i == 0:
                Pkminus1 = 0
            else:
                Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from last read
        
        XREF0 == XREFkminus1 #current reference state
        P0 == Pkminus1 #covariance set by you
        tk = tk[i+1]
        
        #step 3: compute sigma point matrix, Proot via Cholesky
        Proot = np.linalg.cholesky(Pkminus1, lower=True)
        sigma_matrix = np.zeros((nx, nx+1)) #empty matrix
        
        for i in range(nx):
            sigma_matrix[:, i] = XREFminus1 + np.sqrt(nx)*Proot[:, i]
            
        for i in range(nx):
            sigma_matrix[:, i+nx] = XREFminus1 - np.sqrt(nx)*Proot[:, i]
        
        
        #step 4: read next observation 
        #use CR3BP function to obtain XREFk (current reference state) and STM(tk, tk-1)
        NRHOmotion(XREF0, P0, tkminus1, tk, Rk)
        
        
        #step 5: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xksigma = np.zeros((nx, 2*nx+1)) #empty matrix
        for i in range(2*nx):
            xksigma[:, i+1] = Wi*NRHOmotion(sigma_matrix[:, i], P0, tkminus1, tk, Yk, Rk)
        
        shape = (6,1)
        xk = np.zeros(shape) #initialise
        xk = np.sum(xksigma, axis=1)
        
        #step 6: process noise step
        
        Pksigma = np.zeros((L, 2*L+1))
        for i in range(2*nx):
            Pksigma[:, i] = xksigma[:, i]/Wi - np.dot(xk, np.transpose(np.subtract(NRHOmotion(sigma_matrix[:, i], P0, tkminus1, tk, Yk, Rk), xk)))*Wi + Qd
            
        shape = (6,1)
        Pk = np.zeros(shape) #initialise
        Pk = np.sum(Pksigma, axis=1)
        
        Prootnew = np.linalg.cholesky(Pk, lower=True)
        sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
        for i in range(nx):
            sigma_matrix_new[:, i] = xk + np.sqrt(nx)*Prootnew[:, i]
            
        for i in range(nx):
            sigma_matrix_new[:, i+nx] = xk - np.sqrt(nx)*Prootnew[:, i]
        
        
        #step 7: predicted measurement
        #use the cubature rule to compute
        h = np.eye(6) #measurement mapping matrix, identity matrix because we know the state (no need to convert from measurement data)
        
        for i in range(2*nx):
            Ybarksigma = Wi* h @ sigma_matrix_new[:, i]
            
        shape = (6,1)
        Ybark = np.zeros(shape) #initialise
        Ybark = np.sum(Ybarksigma, axis=1)
        
        #step 8: innovation and cross-correlation
        for i in range(2*nx):
            Pyy = Rk + np.sum(Wi*np.dot(np.subtract(h @ sigma_matrix_new[:, i], Ybark[:,i]), np.transpose(np.subtract(h @ sigma_matrix_new[:, i], Ybark[:,i])))) #innovation
            Pxy = np.sum(Wi*np.dot(np.subtract(sigma_matrix_new[:, i], xk[:,i]), np.transpose(np.subtract(h @ sigma_matrix_new[:, i], Ybark[:,i])))) #cross-correlation
        
        #step 9: corrector update
        Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
        Xhatk = xk + Kk @ np.subtract(Yk, Ybark) #final state estimation
        Pk = Pbark - Kk @ np.transpose(Pxy) #final covariance
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xhatk
        covariance_results[i] = Pk
        
    return filter_results, covariance_results