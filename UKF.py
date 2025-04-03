#Unscented Kalman Filter (UKF) Algorithm

#more complex
#more likely to perform well for the more unstable parts of the NRHO

def run_unscented_kalman_filter(XREF, tk, Rk, Qd):

    import numpy as np
    from CR3BP import NRHOmotion
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    state_dev_results = np.zeros_like(XREF)
    
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
        sigma_matrix = np.zeros((L, 2*L+1)) #empty matrix
        
        for i in range(L):
            sigma_matrix[:, i+1] = XREFminus1 + gamma*Proot[:, i]
            
        for i in range(L):
            sigma_matrix[:, i+L+1] = XREFminus1 - gamma*Proot[:, i]
        
        
        #step 4: read next observation 
        #use CR3BP function to obtain XREFk (current reference state) and STM(tk, tk-1)
        NRHOmotion(XREF0, P0, tkminus1, tk, Rk)
        
        
        #step 5: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xk = np.zeros((L, 2*L+1)) #empty matrix
        for i in range(L):
            xk[:, i+1] = NRHOmotion(sigma_matrix[:, i], P0, tkminus1, tk, Rk)
            
        
        #step 6: process noise step
        #compute a priori state xk and covariance Pk for tk
        Xksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            Xksigma = Wim*xk[:,i]
        
        shape = (6,1)
        Xk = np.zeros(shape) #initialise
        Xk = np.sum(Xksigma, axis=1)
        
        #process noise covariance
        
        Pksigma = np.zeros((L, 2*L+1))
        for i in range(2*L):
            Pksigma[:, i] = Wic*np.dot(self, np.subtract(sigma_matrix, Xk), np.subtract(sigma_matrix, Xk)) + Qd
            
        shape = (6,1)
        Pk = np.zeros(shape) #initialise
        Pk = np.sum(Pksigma, axis=1)
        
        Prootnew = np.linalg.cholesky(Pk, lower=True)
        sigma_matrix_new = np.zeros((L, 2*L+1)) #empty matrix
        
        for i in range(L):
            sigma_matrix_new[:, i+1] = Xk + gamma*Prootnew[:, i]
            
        for i in range(L):
            sigma_matrix_new[:, i+L+1] = Xk - gamma*Prootnew[:, i]
        
        
        #step 7: predicted measurement
        #use the UT to compute
        G == sigma_matrix_new #assuming full state is observed
        upsilon == G #just maintaining equation conventions
        H = np.eye(6) #observation mapping matrix
        
        for i in range(2*L):
            Ybarksigma = Wim*upsilon[:, i]
            
        shape = (6,1)
        Ybark = np.zeros(shape) #initialise
        Ybark = np.sum(Ybarksigma, axis=1)
        
        #step 8: innovation and cross-correlation
        for i in range(2*L):
            Pyy = Rk + np.sum(Wic*np.dot(np.subtract(upsilon, Ybark), np.transpose(np.subtract(upsilon, Ybark))), axis=1) #innovation
            Pxy = np.sum(Wic*np.dot(np.subtract(sigma_matrix, Xk), np.transpose(np.subtract(upsilon, Ybark))), axis=1) #cross-correlation
        
        #step 9: corrector update
        Kk = Pxy @ np.linalg.inv(Pyy) #Kalman gain
        Xhatk = Xk + Kk @ np.subtract(Yk, Ybark) #final state estimation
        Pk = Pbark - Kk @ Pyy @ np.transpose(Kk) #final covariance
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xhatk
        covariance_results[i] = Pk
    
    return filter_results, covariance_results
        
