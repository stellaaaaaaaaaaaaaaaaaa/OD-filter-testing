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
    
        #step 1: compute weights (via unscented transform)
        nx = 6
        Wi = 1/(2*nx)
        
        xi = sqrt(nx) #cubature point
        
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
        
        
        #step 3: read next observation 
        #use CR3BP function to obtain XREFk (current reference state)
        NRHOmotion(XREF0, P0, tkminus1, tk, Rk)
        xhat == XREF0 #state expected value (mean) - I think this is just the current reference state
        
        #step 4: compute square root cubature point matrix, S0 via cholesky
        stateP = np.subtract(x0, xhat) @ np.transpose(np.subtract(x0, xhat))
        S0 = np.linalg.cholesky(stateP, lower=True)
        
        #evaluate cubature point matrix
        cub_matrix = np.zeros((nx, nx+1)) #empty matrix
        
        for i in range(nx):
            cub_matrix[:, i] = xhat + xi*S0[:, i]
            
        for i in range(nx):
            cub_matrix[:, i+nx] = xhat - xi*S0[:, i]
        
        #step 5: time update !!
        #this is different because each sigma point (13) is propagated with the equations of motion
        xkcub = np.zeros((nx, nx+1)) #empty matrix
        for i in range(nx):
            xkcub[:, i+1] = Wi*NRHOmotion(cub_matrix[:, i], P0, tkminus1, tk, Rk)
        
        shape = (6,1)
        xk = np.zeros(shape) #initialise
        xk = np.sum(xkcub, axis=1)
        
        for i in range (nx):
            xk[:, i+1] = xkcub[:, i] - xhat
            
        lbda = 1/xi * xk
            
        shape = (6,1)
        Pk = np.zeros(shape) #initialise
        Pk = np.sum(Pkcub, axis=1)
        
        Snew = np.linalg.cholesky(Pk, lower=True)
        
        #step 6: process noise step
        #process noise covariance
        SQ = np.linalg.cholesky(Qd, lower=True)
        
        Sk = np.linalg.lu([lbda SQ])
        
        #step 7: predicted measurement
        #use the cubature rule to compute
        
        #new propagated cubature points with new Sk
        cub_matrix_new = np.zeros((nx, nx+1)) #empty matrix
        
        for i in range (nx):
            cub_matrix_new = xi*Sk + xk[:, i+1]
        
        h = np.eye(6) #measurement mapping matrix, identity matrix because we know the state (no need to convert from measurement data)
        
        Zk = h @ cub_matrix_new
        
        #predicted measurement
        shape = (6,1)
        zk = np.zeros(shape)
        zk = Wi * np.sum(Zk, axis=1)
        
        #step 8: cross-covariance matrix
        shape = (6,1)
        Zzk = np.zeros(shape) #initialise
        
        for i in range (nx):
            Zzk[:, i+1] = Zk[:,i] - zk
        
        lbdak = 1/xi * Zzk
        
        SRk = np.linalg.cholesky(Rk, lower=true)
        Szz = np.linalg.lu([lbdak SRk])
        
        gammak == lbda #for convention sake
        
        yk = np.linalg.subtract(zk, xk) #residual
        
        dem = np.zeros(shape)
        for i in range (nx):
            dem[:, i+1] = Zk[:, i+1] - zk[:, i+1]
        
        lbda0knum = np.trace(yk @ np.transpose(yk))
        lbda0kdem = np.trace(np.linalg.sum(Wi*dem @ np.transpose(dem)))
        lbda0k = lbda0knum @ np.linalg.inv(lbda0kdem)
        
        #adaptive fading factor - reduces impact of disturbance
        if lbda0k < 1:
            alphak = lbda0k
            else:
                alphak = 1
        
        PKk = 1/alphak * (gammak @ np.transpose(lbdak))
        
        #step 9: filter gain
        Kk =(PKk @ np.linalg.inv(np.transpose(Szz))) @ np.linalg.inv(Szz)
        
        Xhatk = xk + Kk @ np.subtract(Yk, zk) #final state estimation k
        Sk = np.linalg.lu([(gammak-Kk@lbdak) (Kk@SRk)]) #square root of covariance (final)
        
         
        #repeat for next observation
        #need to store data in an array such that the filter and the truth data can be plotted against each other
        
        filter_results[i] = Xhatk
        covariance_results[i] = Sk @ Sk
        
    return filter_results, covariance_results