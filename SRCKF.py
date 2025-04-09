#Square Root Cubature Kalman Filter (SRCKF)

#similar to CKF however it calculates the square root of the covariance matrix over the full matrix
#this improves filter robustness and numerical stability as matrix is guaranteed to be positive definite

# https://www.sciencedirect.com/science/article/pii/S187770581600285X#:~:text=A%20new%20fil%20tering%20algorithm%2C%20adaptive,poor%20convergence%20or%20even%20divergence.
#paper incorporates adaptive fading factor to weigh prediction against measurement

def run_square_root_CKF(XREF, tk, Rk, Qd):

    import numpy as np
    from CR3BP import NRHOmotion
    
    filter_results = np.zeros_like(XREF)
    covariance_results = np.zeros((XREF.shape[0],1))
    
    for i, row in enumerate(XREF):
    
        #step 1: compute weights 
        nx = 6
        Wi = 1/(2*nx)
        
        xi = np.sqrt(nx) #cubature point
        
        
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
             
        
        #step 3: compute square root cubature point matrix, S0 via cholesky
        Proot = np.linalg.cholesky(Pkminus1, lower = True)
        
        #evaluate cubature point matrix
        cub_matrix = np.zeros((nx, 2*nx+1)) #empty matrix
        
        cub_matrix[:, 0] = XREFminus1
        
        for i in range(nx):
            cub_matrix[:, i] = XREFminus1 + xi*Proot[:, i]
            
        for i in range(nx):
            cub_matrix[:, i+nx] = XREFminus1 - xi*Proot[:, i]
        
        
        #step 4: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xkcub = np.zeros((nx, 2*nx+1)) #empty matrix
        
        for i in range(nx):
            xkcub[:, i+1] = NRHOmotion(cub_matrix[:, i], Pkminus1, tkminus1, tk)
        
        
        #step 5: process noise step
        shape = (6,1)
        xhat = np.zeros(shape) #initialise
        xhat = Wi*np.sum(xkcub, axis=1) #predicted state 
        
        
        lbdamat = np.zeros((nx, 2*nx+1)) #empty matrix
        
        for i in range (2*nx):
            lbdamat[:, i+1] = xkcub[:, i] - xhat
        
        
        lbda = 1/(np.sqrt(2*nx)) * lbdamat 
            
        Pkcub = np.zeros((nx, 2*nx+1))
        for i in range(2*nx):
            Pkcub[:, i] = np.dot(np.subtract(xkcub[:, i]/Wi, xhat), np.transpose(np.subtract(xkcub[:, i]/Wi, xhat))*Wi)
        
        #process noise covariance
        SQ = np.linalg.cholesky(Qd, lower=True)
        
        Sk = np.linalg.lu([lbda, SQ], lower=True)
        
        
        #step 6: predicted measurement
        #use the cubature rule to compute
        
        #new propagated cubature points with new Sk
        cub_matrix_new = np.zeros((nx, nx+1)) #empty matrix
        
        cub_matrix_new[:, 0] = xhat
        
        for i in range(nx):
            cub_matrix[:, i] = xhat + xi*Sk[:, i]
            
        for i in range(nx):
            cub_matrix[:, i+nx] = xhat - xi*Sk[:, i]
        
        cub_matrix_new = np.zeros((nx, nx+1)) #empty matrix
            
        upsilon = np.zeros((nx, 2*nx+1)) #empty matrix
        
        for i in range(nx):
            upsilon[:, i] = NRHOmotion(cub_matrix_new[:,i+1], Pkminus1, tkminus1, tk)
        
        shape = (6,1)
        Ybark = np.zeros(shape)
        Ybark = Wi * np.sum(upsilon, axis=1)
        
        
        #step 7: cross-covariance matrix
        shape = (6,1)
        Zzk = np.zeros(shape) #initialise
        
        for i in range (nx):
            Zzk[:, i+1] = upsilon[:,i] - Ybark
        
        lbdak = 1/xi * Zzk
        
        SRk = np.linalg.cholesky(Rk, lower = True)
        Szz = np.linalg.lu([lbdak, SRk])
        
        gammak = lbda #for convention sake
        
        dem = np.zeros(shape)
        for i in range (nx):
            dem[:, i+1] = upsilon[:, i+1] - Ybark
        
        Yk = XREF[i+1]
        yk = Yk - Ybark #residual
        
        lbda0knum = np.trace(yk @ np.transpose(yk))
        lbda0kdem = np.trace(np.linalg.sum(Wi*dem @ np.transpose(dem)))
        lbda0k = lbda0knum @ np.linalg.inv(lbda0kdem)
        
        #adaptive fading factor - reduces impact of disturbance
        if lbda0k < 1:
            alphak = lbda0k
        else:
                alphak = 1
        
        PKk = 1/alphak * (gammak @ np.transpose(lbdak))
        
        
        #step 8: filter gain
        Kk =(PKk @ np.linalg.inv(np.transpose(Szz))) @ np.linalg.inv(Szz)
        
        Xkfinal = xhat + Kk @ np.subtract(Yk, Ybark) #final state estimation k
        Sk = np.linalg.lu([(gammak-Kk@lbdak) (Kk@SRk)]) #square root of covariance (final)
        
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xkfinal
        covariance_results[i] = Sk @ Sk
        
    return filter_results, covariance_results

