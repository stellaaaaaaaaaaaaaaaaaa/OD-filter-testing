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
    
    #old STM for CR3BP
    
    start_time = time.time()
    max_time = 5.0  # Maximum time for STM calculation
    
    """STM for CR3BP in dimensional units (km, km/s)"""
    import numpy as np
    from scipy.linalg import expm
    
    # Constants
    mu_earth = 398600.435507  # km³/s² 
    mu_moon = 4902.800118     # km³/s²
    L_EM = 384400             # Earth-Moon distance in km
    
    # Convert to normalized units for calculation
    x_norm = XREFk[0] / L_EM
    y_norm = XREFk[1] / L_EM
    z_norm = XREFk[2] / L_EM
    
    # Mass ratio
    mu = mu_moon / (mu_earth + mu_moon)
    
    # Angular velocity of Earth-Moon system
    n = np.sqrt((mu_earth + mu_moon) / L_EM**3)  # rad/s
    
    # Distances in normalized units
    r1 = np.sqrt((x_norm + mu)**2 + y_norm**2 + z_norm**2)
    r2 = np.sqrt((x_norm - 1 + mu)**2 + y_norm**2 + z_norm**2)
    
    # Safety check
    if r1 < 0.001 or r2 < 0.001:  # About 384 km minimum distance
        print(f"Warning: Too close to primary! r1={r1*L_EM:.1f} km, r2={r2*L_EM:.1f} km")
        r1 = max(r1, 0.001)
        r2 = max(r2, 0.001)
    
    # Calculate normalized derivatives
    Uxx_norm = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x_norm+mu)**2/(r1**5) + 3*mu*(x_norm-1+mu)**2/(r2**5)
    Uyy_norm = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y_norm**2/(r1**5) + 3*mu*y_norm**2/(r2**5)
    Uzz_norm = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z_norm**2/(r1**5) + 3*mu*z_norm**2/(r2**5)
    Uxy_norm = 3*((1-mu)*(x_norm+mu)/(r1**5) + mu*(x_norm-1+mu)/(r2**5))*y_norm
    Uxz_norm = 3*((1-mu)*(x_norm+mu)/(r1**5) + mu*(x_norm-1+mu)/(r2**5))*z_norm
    Uyz_norm = 3*((1-mu)/(r1**5) + mu/(r2**5))*y_norm*z_norm
    
    # Convert to dimensional units
    # U has units of 1/s², so multiply by n²
    Uxx = Uxx_norm * n**2
    Uyy = Uyy_norm * n**2
    Uzz = Uzz_norm * n**2
    Uxy = Uxy_norm * n**2
    Uxz = Uxz_norm * n**2
    Uyz = Uyz_norm * n**2
    
    # Build A matrix in dimensional units
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, :3] = [Uxx, Uxy, Uxz]
    A[4, :3] = [Uxy, Uyy, Uyz]
    A[5, :3] = [Uxz, Uyz, Uzz]
    A[3, 4] = 2 * n    # Coriolis
    A[4, 3] = -2 * n   # Coriolis
    
    # Calculate STM
    return expm(A * dt)
  

def STM_from_states(X0, Xf, t0, tf, orb=None):

    if orb is None:
        orb = {}
        orb = LoadModel("Fast", orb)
        
        if orb['prop']['harmonics']['degreeE'] == 0:
            orb['prop']['harmonics']['degreeE'] = 1
            orb['prop']['harmonics']['orderE'] = 1
            orb['prop']['harmonics']['ECnm'] = np.zeros((2, 2))
            orb['prop']['harmonics']['ESnm'] = np.zeros((2, 2))
            orb['prop']['harmonics']['ECnm'][0, 0] = 1
    
    # Constants
    mu_earth = 398600.435507  # km³/s²
    mu_moon = 4902.800118     # km³/s²
    
  
    dt = tf - t0
    
    #Earth-Moon vector at midpoint
    t_mid = (t0 + tf) / 2
    
    if spice.ktotal('ALL') == 0:
        spice.furnsh('metakernel.tm')
        
    r_em, _ = spice.spkpos('EARTH', t_mid, 'J2000', 'NONE', 'MOON')
    r_em = np.array(r_em)

    
    # Use initial position for Jacobian
    r = np.array(X0[:3])
    
    # Compute combined Jacobian
    r_norm = np.linalg.norm(r)
    r_e_vec = r + r_em
    r_e_norm = np.linalg.norm(r_e_vec)
    
    # Moon's gravity Jacobian
    J_moon = -mu_moon / r_norm**3 * (np.eye(3) - 3 * np.outer(r, r) / r_norm**2)
    
    # Earth's gravity Jacobian
    J_earth = -mu_earth / r_e_norm**3 * (np.eye(3) - 3 * np.outer(r_e_vec, r_e_vec) / r_e_norm**2)
    
    J2 = 0.00108263  # Earth J2
    R_earth = 6378.137  # km
    r_earth = r + r_em
    r_earth_norm = np.linalg.norm(r_earth)
    
    J_harmonics = np.zeros((3, 3))
    if r_earth_norm > R_earth:
        x, y, z = r_earth
        factor = 3 * mu_earth * J2 * R_earth**2 / (2 * r_earth_norm**5)
        
        J_harmonics[0, 0] = factor * (1 - 5*(z/r_earth_norm)**2 + 15*(x/r_earth_norm)**2)
        J_harmonics[1, 1] = factor * (1 - 5*(z/r_earth_norm)**2 + 15*(y/r_earth_norm)**2)
        J_harmonics[2, 2] = factor * (3 - 5*(z/r_earth_norm)**2)
        J_harmonics[0, 1] = J_harmonics[1, 0] = factor * 15 * x * y / r_earth_norm**2
        J_harmonics[0, 2] = J_harmonics[2, 0] = factor * 15 * x * z / r_earth_norm**2
        J_harmonics[1, 2] = J_harmonics[2, 1] = factor * 15 * y * z / r_earth_norm**2
    
    # Jacobian
    J = J_moon + J_earth + J_harmonics
    
    # 
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = J
    
    # Compute STM using matrix exponential
    STM = expm(A * dt)
    
    return STM
