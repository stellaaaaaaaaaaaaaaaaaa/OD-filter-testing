#single point propagation
#via HALO high-fidelity modelling --> i.e. take current state k and propagate to state at k+1
#note that NRHO input data may have to change as the current 'truth' trajectory was generated via this propagational model
#should be an NRHO data file (excel spreadsheet)


from scipy.integrate import solve_ivp
from input.LoadModel import LoadModel
from prop.prophpop import prophpop
from scipy.linalg import expm
import time
import numpy as np
import spiceypy as spice

def point_propagation(XREF0, tkminus1, tk):
    
    #XREF0 includes 6 state vectors, tkminus1 and tk already in ET seconds atm
    
    spice.kclear()
    spice.furnsh('metakernel.tm')
    
    orb = {}
    orb = LoadModel("Fast", orb) #can also change to 'fast' for debugging, default "Ref"
    
    #note below is to ensure the "Fast" setting works
    if orb['prop']['harmonics']['degreeE'] == 0:
        orb['prop']['harmonics']['degreeE'] = 1
        orb['prop']['harmonics']['orderE'] = 1
        orb['prop']['harmonics']['ECnm'] = np.zeros((2, 2))
        orb['prop']['harmonics']['ESnm'] = np.zeros((2, 2))
        orb['prop']['harmonics']['ECnm'][0, 0] = 1
    
    sol = solve_ivp(
        prophpop,
        (tkminus1, tk),
        XREF0,
        method='DOP853',
        args = (orb,),
        rtol=1e-7, atol=1e-14
    )
    
    if not sol.success:
        raise RuntimeError(f"Integration failed")
    
    if sol.success:
        XREFk = sol.y[:, -1] 
    
    return XREFk

    
def STM(XREFk, dt):
    
    # Start timing the operation
    start_time = time.time()
    max_time = 10  # Maximum time for STM calculation
    
    x, y, z = XREFk[0], XREFk[1], XREFk[2]
    
    #Earth-Moon system
    m1 = 5.9722e24  # Earth mass, kg
    m2 = 7.3477e22  # Moon mass, kg
    mu = m2/(m1+m2)  # mass ratio
     
    # Compute distances to Earth and Moon
    r1 = np.sqrt((x+mu)**2 + y**2 + z**2)  # Distance to Earth
    r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)  # Distance to Moon
        
    # Compute second derivatives of the potential U
    try:
        Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
        Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
        Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
        Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
        Uyx = Uxy
        Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
        Uzx = Uxz
        Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
        Uzy = Uyz
    
    except Exception as exc:
        print(f"  Error in potential derivatives: {exc}")
        # Return simplified STM on error
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        return np.eye(6) + A * dt
    
    # Check for timeout
    if time.time() - start_time > max_time:
        print("  STM calculation taking too long, using approximation")
        # Return simplified STM on timeout
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        return np.eye(6) + A * dt

    # Construct A matrix (variational equation matrix)
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)  # Identity matrix for position derivatives
    
    # Potential derivatives
    A[3, :3] = [Uxx, Uxy, Uxz]
    A[4, :3] = [Uyx, Uyy, Uyz]
    A[5, :3] = [Uzx, Uzy, Uzz]
    
    # Coriolis terms
    A[3, 4] = 2    # 2Ω
    A[4, 3] = -2   # -2Ω
    
    # Compute state transition matrix using two approaches
    try:
        # First try matrix exponential
        STM = expm(A * dt)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(STM)) or np.any(np.isinf(STM)):
            raise ValueError("STM contains NaN or Inf values")
            
        return STM
    except Exception as exc:
        print(f"  Matrix exponential failed: {exc}")
        # Fallback to Taylor series approximation
        I = np.eye(6)
        return I + A*dt + 0.5*A@A*dt**2
    