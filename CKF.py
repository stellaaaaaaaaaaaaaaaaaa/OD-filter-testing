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
        
        tk = tk[i+1]
        
        
        #step 3: compute cubature point matrix, Proot via Cholesky
        Proot = np.linalg.cholesky(Pkminus1, lower=True)
        cub_matrix = np.zeros((nx, 2*nx+1)) #empty matrix
        
        cub_matrix[:, 0] = XREFminus1
        
        for i in range(nx):
            cub_matrix[:, i] = XREFminus1 + np.sqrt(nx)*Proot[:, i]
            
        for i in range(nx):
            cub_matrix[:, i+nx] = XREFminus1 - np.sqrt(nx)*Proot[:, i]
        
        
        #step 5: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xkcub = np.zeros((nx, 2*nx+1)) #empty matrix
        
        for i in range(2*nx):
            xkcub[:, i+1] = Wi*NRHOmotion(cub_matrix[:, i], Pkminus1, tkminus1, tk)
        

        #step 6: process noise step
        #sum weighted propagated points to obtain time updated state estimate
        shape = (6,1)
        xk = np.zeros(shape) #initialise
        xk = np.sum(xkcub, axis=1)
        
        Pksigma = np.zeros((nx, 2*nx+1))
        for i in range(2*nx):
            Pksigma[:, i] = np.dot(np.subtract(xkcub[:, i]/Wi, xk), np.transpose(np.subtract(xkcub[:, i]/Wi, xk))*Wi)
            
        shape = (6,1)
        Pk = np.zeros(shape) #initialise
        Pk = np.sum(Pksigma, axis=1) + Qd
        
        Prootnew = np.linalg.cholesky(Pk, lower=True)
        cub_matrix_new = np.zeros((nx, 2*nx+1)) #empty matrix
        
        cub_matrix_new[:, 0] = xk
        
        for i in range(nx):
            cub_matrix_new[:, i] = xk + np.sqrt(nx)*Prootnew
            
        for i in range(nx):
            cub_matrix_new[:, i+nx] = xk - np.sqrt(nx)*Prootnew
        
        
        #step 7: predicted measurement
        #use the cubature rule to compute
        h = np.eye(6) #measurement mapping matrix, identity matrix because we know the state (no need to convert from measurement data)
        
        upsilon = np.zeros((nx, 2*nx+1)) #empty matrix   
        
        for i in range(2*nx):
            upsilon[:,i+1] = h @ NRHOmotion(cub_matrix_new[:, i], Pkminus1, tkminus1, tk)
            
        shape = (6,1)
        Ybark = np.zeros(shape) #initialise
        Ybark = Wi*np.sum(upsilon, axis=1)
        
        
        #step 8: innovation and cross-correlation
        for i in range(2*nx):
            innovation_op = np.dot(np.subtract(upsilon[:, i], Ybark), np.transpose(np.subtract(upsilon[:, i], Ybark))) #innovation
            crosscor_op = np.dot(np.subtract(upsilon[:, i], xk), np.transpose(np.subtract(upsilon[:, i], Ybark))) #cross-correlation
        
        Pyy = Rk + Wi*np.sum(innovation_op, axis=1) #innovation
        Pxy = Wi*np.sum(crosscor_op, axis=1) #cross-correlation
        
        
        #step 9: corrector update
        Yk = XREF[i+1]        
        
        Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
        Xkfinal = xk + Kk @ np.subtract(Yk, Ybark) #final state estimation
        Pkfinal = Pk - Kk @ np.transpose(Pxy) #final covariance
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xkfinal
        covariance_results[i] = Pkfinal
        
    return filter_results, covariance_results


