#NRHO Equations of Motion for Filtering Algorithms

#position data from HALO truth data will be propagated via filters that use
#these equations of motion

#for NRHO: circular restricted three body problem (CR3BP)
#note that additional perturbations will be accounted for as process noise

#input measured time, position, measurement noise
#propagate via equations of motion
#output reference state and state transition matrix (STM)
#note Newtonian coordinates, relative to Earth

def NRHOmotion(XREF0, x0, P0, tkminus1, tk, Rk):
    
    #a priori state
    #XREF0 = reference state (last known) == Yk
    #x0 = known state deviation (x0 = 0 initially, updated over time)
    #P0 = state covariance matrix/uncertainty
    
    #observation data
    #tk = current time data
    #Rk = measurement noise
    
    import numpy as np
    from scipy.integrate import solve_ivp
    
    Yk = XREF0
    
    #Earth-Moon system
    m1 = 5.9722e24 #Earth mass, kg
    m2 = 7.3477e22 #Moon mass, kg
    mu = m2/(m1+m2) #mass ratio
    
    #solving CR3BP non-dimensional coordinates
    x, y, z = Yk[:3]
    xdot, ydot, zdot = Yk[3:]
    
    #derivative vector
    Ydot = np.zeros_like(Yk)
    Ydot[:3] = Yk[3:]
    
    
    sigma = np.sqrt(np.sum(np.square([x + mu, y, z])))
    psi = np.sqrt(np.sum(np.square([x - 1 + mu, y, z])))
    Ydot[3] = (
    2 * ydot
    + x
    - (1 - mu) * (x + mu) / sigma**3
    - mu * (x - 1 + mu) / psi**3
    )
    Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    Ydot[5] = -(1 - mu) / sigma**3 * z - mu / psi**3 * z
    return Ydot
    
    t_points = np.linspace(tkminus1, tk, 1)
    sol = solve_ivp(nondim_cr3bp, [t_0, t_f], Y_0, t_eval=t_points)

    #XREFk output
    XREFk = sol.y.T
    return XREFk
    

def STM(XREFk, tk):
    #STM - this is fairly computationally heavy, hence should be its own function as only EKF uses it
    import numpy as np

    x, y, z = XREFk[:3]
    xdot, ydot, zdot = XREFk[3:]
    
    #Earth-Moon system
    m1 = 5.9722e24 #Earth mass, kg
    m2 = 7.3477e22 #Moon mass, kg
    mu = m2/(m1+m2) #mass ratio
     
    # Compute distances to Earth and Moon
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)# Distance to Earth
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2) #Distance to the Moon

    # Compute second derivatives of the potential U
    Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
    Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
    Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
    Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
    Uyx = Uxy
    Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
    Uzx = Uxz
    Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
    Uzy = Uyz

    # Construct A matrix
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)  # Identity matrix
    A[3, :3] = [Uxx, Uxy, Uxz]
    A[4, :3] = [Uyx, Uyy, Uyz]
    A[5, :3] = [Uzx, Uzy, Uzz]
        #2*omega matrix, represent Coriolis terms
    A[3, 3:] = [0, 2, 0]
    A[4, 3:] = [-2, 0, 0]
    A[5, 3:] = [0, 0, 0]
    
    #determine state transition matrix (STM)
    # https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Introduction_to_Control_Systems_(Iqbal)/08%3A_State_Variable_Models/8.02%3A_State-Transition_Matrix_and_Asymptotic_Stability#:~:text=with%20Complex%20Roots-,The%20State%2DTransition%20Matrix,vector%2C%20x(t).
    STM  = np.expm(A * tk) # Compute the state transition matrix
    return STM
    



#sources:
#    https://orbital-mechanics.space/the-n-body-problem/Equations-of-Motion-CR3BP.html
#    https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
#   Dr. Shane Ross orbital mechanics lecture
# https://people.unipi.it/tommei/wp-content/uploads/sites/124/2021/08/3body.pdf - page 197