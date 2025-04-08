#Unscented Kalman Filter (UKF) Algorithm

#more complex
#more likely to perform well for the more unstable parts of the NRHO

def run_unscented_kalman_filter(XREF, tk, Rk, Qd):

    import numpy as np
    from CR3BP import NRHOmotion
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    
    
    for i, row in enumerate(XREF):
    
        #step 1: compute weights (via unscented transform)
        L = 6 #number of states, 3 position, 3 velocity
        
        #tuning parameters for Gaussian initial PDF
        alpha = 1 #determines spread of sigma points
        beta = 2
        kappa = 3 - L
        
        lbda = (alpha**2)*(L+kappa) - L
        gamma = np.sqrt(L + lbda)
        
        W0m = lbda/(L+lbda)
        W0c = lbda/(L+lbda) + 1 - alpha**2 + beta
        
        Wim = 1/(2*(L+lbda))
        Wic = Wim
        
        
        #step 2: initialisation (a priori)
        XREFminus1 = XREF[i] #previous reference state determined from last read
        
        if len(covariance_results) == 0:
            covariance_results = np.array([0])
            
        for i, row in enumerate(covariance_results):
            if i == 0:
                Pkminus1 = 0
            else:
                Pkminus1 = covariance_results[i-1] #previous covariance/uncertainty determined from last read
        
        tkminus1 = tk[i]
        tk = tk[i+1]
 
        
        #step 3: compute sigma point matrix, Proot via Cholesky
        Proot = np.linalg.cholesky(Pkminus1, lower=True)
        sigma_matrix = np.zeros((L, 2*L+1)) #empty matrix
        
        sigma_matrix[:, 0] = XREFminus1 #first sigma point
        
        for i in range(L):
            sigma_matrix[:, i+1] = XREFminus1 + gamma*Proot[:, i] #mean + scaled root of covariance
            
        for i in range(L):
            sigma_matrix[:, i+L+1] = XREFminus1 - gamma*Proot[:, i] #mean - scaled root of covariance
        
        
        #step 4: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xbark = np.zeros((L, 2*L+1)) #empty matrix
        for i in range(L):
            xbark[:, i+1] = NRHOmotion(sigma_matrix[:, i], Pkminus1, tkminus1, tk)
            
        
        #step 5: process noise step
        #compute a state xk and covariance Pk for tk
        Xksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            if i == 0:
                Xksigma[:,i] = W0m*xbark[:,i]
            else:
                Xksigma[:,i] = Wim*xbark[:,i]
        
        shape = (6,1)
        Xbark = np.zeros(shape) #initialise
        Xbark = np.sum(Xksigma, axis=1)
        
        #process noise covariance
        
        Pksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            if i == 0:
                Pksigma[:, i] = W0c*np.dot(np.subtract(xbark[:,i], Xbark), np.transpose(np.subtract(xbark[:,i], Xbark)))
            else:
                Pksigma[:, i] = Wic*np.dot(np.subtract(xbark[:,i], Xbark), np.transpose(np.subtract(xbark[:,i], Xbark)))
            
        shape = (6,1)
        Pbark = np.zeros(shape) #initialise
        Pbark = np.sum(Pksigma, axis=1) + Qd
        
        Prootnew = np.linalg.cholesky(Pbark, lower=True)
        sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
        sigma_matrix_new[:, 0] = Xbark #first sigma point
        
        for i in range(L):
            sigma_matrix_new[:, i+1] = Xbark + gamma*Prootnew[:, i] #mean + scaled root of covariance
            
        for i in range(L):
            sigma_matrix_new[:, i+L+1] = Xbark - gamma*Prootnew[:, i] #mean - scaled root of covariance
                
        
        #step 6: predicted measurement
        #use the UT to compute
        upsilon = np.zeros((L, 2*L+1)) #empty matrix
        for i in range(L):
            upsilon[:, i+1] = NRHOmotion(sigma_matrix_new[:, i], Pkminus1, tkminus1, tk)
        
        Ybarksigma = np.zeros((L, 2*L+1))
        
        for i in range(2*L):
            if i == 0:
                Ybarksigma[:,i] = W0m*upsilon[:, i]
            else:
                Ybarksigma[:,i] = Wim*upsilon[:, i]
            
        shape = (6,1)
        Ybark = np.zeros(shape) #initialise
        Ybark = np.sum(Ybarksigma, axis=1)
        
        
        #step 7: innovation and cross-correlation
        
        innovation_op = np.zeros((L, 2*L+1))
        crosscor_op = np.zeros((L, 2*L+1))
        
        for i in range(2*L):
            if i == 0: 
                innovation_op[:, i] = W0c*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
                crosscor_op[:, i] = W0c*np.dot(np.subtract(sigma_matrix[:,i], Xbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            else:
                innovation_op[:, i] = Wic*np.dot(np.subtract(upsilon[:,i], Ybark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
                crosscor_op[:, i] = Wic*np.dot(np.subtract(sigma_matrix[:,i], Xbark), np.transpose(np.subtract(upsilon[:,i], Ybark)))
            
            Pyy = Rk + np.sum(innovation_op, axis=1) #innovation
            Pxy = np.sum(crosscor_op, axis=1) #cross-correlation
        
        
        #step 8: corrector update
        Yk = XREF[i+1]
        Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
        Xk = Xbark + Kk @ np.subtract(Yk, Ybark) #final state estimation
        Pk = Pbark - Kk @ Pyy @ np.transpose(Kk) #final covariance
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xk
        covariance_results[i] = Pk
    
    return filter_results, covariance_results
        
