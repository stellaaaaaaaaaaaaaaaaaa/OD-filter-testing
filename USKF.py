#Unscented Schmidt Kalman Filter (USKF) Algorithm

#more complex
#more likely to perform well for the more unstable parts of the NRHO
#incorporates consider parameters - for orbit determination this may include:
    #3BP perturbations, gravitational uncertainties, SRP, ground station biases

def run_unscented_schmidt_KF(XREF, tk, c, Pcc, Rk, Qd):
        
    import numpy as np
    from CR3BP import NRHOmotion
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    
    for i, row in enumerate(XREF):
    
        #step 1: compute weights (via unscented transform)
        nx = 6
        nc = 4 #assuming using all consider parameters aforementioned
        L = (nx+nc)
        gamma = np.sqrt(L)
        Wi = 1/(2*(nx+nc))
        
        
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
        
        Pxx = Pkminus1
        Pxc = np.zeros(6) #check this lol
        Pcx = np.transpose(Pxc)
        
        PxxPxc = np.stack((Pxx, Pxc), axis = 1).shape
        PcxPcc = np.stack((Pcx, Pcc), axis = 1).shape
        Pzz = np.stack((PxxPxc, PcxPcc)).shape
        
        
        #step 3: combine XREFk and consider parameter vector to get 'z'
        z = np.vstack((XREFminus1, c))
        
        #measurement mapping matrices
        #hx, hc, assume no change, identity matrices
       
        
        #step 4: compute sigma point matrix, Szz via Cholesky
        Szz = np.linalg.cholesky(Pzz, lower=True)
        sigma_matrix = np.zeros((L, 2*L+1)) #empty matrix
        
        sigma_matrix[:, 0] = z #first sigma point
        
        for i in range(L):
            sigma_matrix[:, i+1] = z + gamma*Szz[:, i]
            
        for i in range(L):
            sigma_matrix[:, i+L+1] = z - gamma*Szz[:, i]
        
        
        #step 5: time update !! 
        #this is different because each sigma point (13) is propagated with the equations of motion
        Zk = np.zeros((L, 2*L+1)) #empty matrix
        for i in range(L):
            Zk[:, i+1] = NRHOmotion(sigma_matrix[:, i], Pkminus1, tkminus1, tk, Rk)
            
        
        #step 6: process noise step
        #compute a priori state xk and covariance Pk for tk
        zksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            zksigma = Wi*Zk[:,i]
        
        shape = (6,1)
        Zbark = np.zeros(shape) #initialise
        Zbark = np.sum(zksigma, axis=1)
        
        #process noise covariance
        
        Pzzksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            Pzzksigma[:, i] = Wi*np.dot(np.subtract(sigma_matrix[:, i], Zbark), np.transpose(np.subtract(sigma_matrix[:, i], Zbark)))
            
        shape = (6,1)
        Pzzk = np.zeros(shape) #initialise
        Pzzk = np.sum(Pzzksigma, axis=1) + Qd
        
        Szznew = np.linalg.cholesky(Pzzk, lower=True)
        sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
        sigma_matrix_new[:, 0] = Zbark       
        
        for i in range(L):
            sigma_matrix_new[:, i+1] = Zbark + gamma*Szznew[:, i]
            
        for i in range(L):
            sigma_matrix_new[:, i+L+1] = Zbark - gamma*Szznew[:, i]
        
        
        #step 7: predicted measurement
        #use the UT to compute
        
        upsilon = np.zeros((L, 2*L+1)) #empty matrix
        for i in range(L):
            upsilon[:, i+1] = NRHOmotion(sigma_matrix_new[:, i], Pkminus1, tkminus1, tk)
        
        Ybarksigma = np.zeros((L, 2*L+1))
        
        for i in range(2*L):
            Ybarksigma[:,i+1] = Wi*upsilon[:, i]
            
        shape = (6,1)
        Ybark = np.zeros(shape) #initialise
        Ybark = np.sum(Ybarksigma, axis=1)
        
        
        #step 8: innovation and cross-correlation
        
        innovation_op = np.zeros((L, 2*L+1))
        crosscor_op = np.zeros((L, 2*L+1))
        
        for i in range(2*L):
            innovation_op[:, i] = Wi*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            crosscor_op[:, i] = Wi*np.dot(np.subtract(sigma_matrix[:,i], Zbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            
            Pyy = Rk + np.sum(innovation_op, axis=1) #innovation
            Pzy = np.sum(crosscor_op, axis=1) #cross-correlation
    
        
        #step 9: corrector update
        Kz = Pzy @ np.linalg.inv(Pyy) #Kalman gain
        
        Kx = np.array([[Kz[0,0]],
                       [Kz[0,1]],
                       [Kz[0,2]],
                       [Kz[0,3]],
                       [Kz[0,4]],
                       [Kz[0,5]]])
        
        Kc = np.array([[Kz[0,6]],
                       [Kz[0,7]],
                       [Kz[0,8]],
                       [Kz[0,9]]])
        
        KX = np.array([[Kx],
                       [0]])
        
        Yk = XREF[i+1]
        Zkfinal = Zbark + KX @ np.subtract(Yk, Ybark) #final state estimation
        
        Kc = Kz[0,1]
        uncert = np.array([[Kx @ Pyy @ np.transpose(Kx), Kx @ Pyy @ np.transpose(Kc)],
                          [Kc @ Pyy @ np.transpose(Kx), 0]])
        
        Pzzk = Pzz - uncert #final covariance
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Zkfinal
        covariance_results[i] = Pzzk
        
        
    return filter_results, covariance_results
        
