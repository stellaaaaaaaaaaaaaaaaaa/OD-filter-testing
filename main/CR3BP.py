#NRHO Equations of Motion for Filtering Algorithms

#position data from HALO truth data will be propagated via filters that use
#these equations of motion

#for NRHO: circular restricted three body problem (CR3BP)
#note that additional perturbations will be accounted for as process noise

# Improved CR3BP implementation with timeout protection

# CR3BP implementation with proper normalization

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import time

# Constants for Earth-Moon system (UNCHANGED)
EARTH_MASS = 5.9722e24  # kg
MOON_MASS = 7.3477e22   # kg
EARTH_MOON_DISTANCE = 384400.0  # km
EARTH_MOON_PERIOD = 27.3 * 24 * 3600  # seconds (27.3 days)

# constants (UNCHANGED)
MU = MOON_MASS / (EARTH_MASS + MOON_MASS)
# Angular velocity of the rotating frame (UNCHANGED)
OMEGA = 2 * np.pi / EARTH_MOON_PERIOD  # rad/s

# ADD: Gravitational parameters
MU_EARTH = 3.986004418e5  # km³/s² 
MU_MOON = 4.9048695e3     # km³/s²

# Conversion functions between physical and normalised units (UNCHANGED)
def metric_to_normalised_pos(pos_km):
    return pos_km / EARTH_MOON_DISTANCE

def metric_to_normalised_vel(vel_km_s):
    return vel_km_s / (EARTH_MOON_DISTANCE * OMEGA)

def metric_to_normalised_time(time_s):
    return time_s * OMEGA

def normalised_to_metric_pos(pos_norm):
    return pos_norm * EARTH_MOON_DISTANCE

def normalised_to_metric_vel(vel_norm):
    return vel_norm * (EARTH_MOON_DISTANCE * OMEGA)

def normalised_to_metric_time(time_norm):
    return time_norm / OMEGA

# Counter for linear propagation fallbacks (UNCHANGED)
linear_prop_count = 0

# ADD: Helper functions for inertial frame
def get_earth_moon_positions_inertial(t):
    """Get Earth and Moon positions in barycentric inertial frame"""
    theta = OMEGA * t
    
    r_earth = np.array([
        -MU * EARTH_MOON_DISTANCE * np.cos(theta),
        -MU * EARTH_MOON_DISTANCE * np.sin(theta),
        0.0
    ])
    
    r_moon = np.array([
        (1 - MU) * EARTH_MOON_DISTANCE * np.cos(theta),
        (1 - MU) * EARTH_MOON_DISTANCE * np.sin(theta),
        0.0
    ])
    
    return r_earth, r_moon

def transform_rotating_to_inertial(state_rotating_norm, t):
    """Transform from rotating normalized to inertial physical coordinates"""
    theta = OMEGA * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotation matrix
    R = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta,  cos_theta, 0],
        [0,         0,          1]
    ])
    
    R_dot = OMEGA * np.array([
        [-sin_theta, -cos_theta, 0],
        [cos_theta,  -sin_theta, 0],
        [0,          0,          0]
    ])
    
    # Convert to physical units
    pos_phys = state_rotating_norm[:3] * EARTH_MOON_DISTANCE
    vel_phys = state_rotating_norm[3:] * (EARTH_MOON_DISTANCE * OMEGA)
    
    # Transform to inertial
    pos_inertial = R @ pos_phys
    vel_inertial = R @ vel_phys + R_dot @ pos_phys
    
    return np.concatenate([pos_inertial, vel_inertial])

def transform_inertial_to_rotating(state_inertial, t):
    """Transform from inertial physical to rotating normalized coordinates"""
    theta = OMEGA * t
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Inverse rotation matrix
    R_inv = np.array([
        [cos_theta,  sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0,          0,         1]
    ])
    
    R_dot_inv = OMEGA * np.array([
        [-sin_theta, cos_theta, 0],
        [-cos_theta, -sin_theta, 0],
        [0,          0,          0]
    ])
    
    pos_inertial = state_inertial[:3]
    vel_inertial = state_inertial[3:]
    
    # Transform to rotating
    pos_rotating = R_inv @ pos_inertial
    vel_rotating = R_inv @ vel_inertial - R_dot_inv @ pos_inertial
    
    # Convert to normalized units
    pos_norm = pos_rotating / EARTH_MOON_DISTANCE
    vel_norm = vel_rotating / (EARTH_MOON_DISTANCE * OMEGA)
    
    return np.concatenate([pos_norm, vel_norm])

def NRHOmotion(XREF0, tkminus1, tk):
    """
    MODIFIED: Now returns state in barycentric inertial frame
    Uses existing rotating frame integration, then transforms output
    """

    global linear_prop_count
    
    XREF0 = np.array(XREF0).flatten()
    
    # NEW: Transform input from inertial to rotating normalized
    x_norm = transform_inertial_to_rotating(XREF0, tkminus1)
    
    # Convert times to normalised units (UNCHANGED)
    t0_norm = metric_to_normalised_time(tkminus1)
    tf_norm = metric_to_normalised_time(tk)
    dt_norm = tf_norm - t0_norm
    
    # For very small time steps, just return the initial state (UNCHANGED)
    if abs(dt_norm) < 1e-10:
        return XREF0.copy()
    
    # CR3BP equations of motion (UNCHANGED)
    def cr3bp_eqn(t, Y):
        """CR3BP equations of motion in normalized units"""
        x, y, z, xdot, ydot, zdot = Y
    
        # Derivative vector
        Ydot = np.zeros_like(Y)
        Ydot[:3] = Y[3:]  # Position derivatives = velocity
    
        # Distances to primaries in normalized units
        r1_sqd = (x + MU)**2 + y**2 + z**2      # Distance to Earth
        r2_sqd = (x - 1 + MU)**2 + y**2 + z**2  # Distance to Moon
        
        # Safety check - prevent division by very small numbers
        min_dist = 0.01
        
        if r1_sqd < min_dist**2 or r2_sqd < min_dist**2:
            # Instead of raising an error, use clamped distances
            r1 = max(np.sqrt(r1_sqd), min_dist)
            r2 = max(np.sqrt(r2_sqd), min_dist)
        else:
            r1 = np.sqrt(r1_sqd)
            r2 = np.sqrt(r2_sqd)
        
        # Acceleration terms in normalised units
        Ydot[3] = (
            2 * ydot 
            + x
            - (1 - MU) * (x + MU) / r1**3
            - MU * (x - 1 + MU) / r2**3
            )
        Ydot[4] = (
            -2 * xdot 
            + y 
            - (1 - MU) * y / r1**3 
            - MU * y / r2**3
            )
        Ydot[5] = (
            -(1 - MU) * z / r1**3 
            - MU * z / r2**3
            )
        
        return Ydot

    try:
        # Integration in rotating frame (UNCHANGED)
        sol = solve_ivp(
            cr3bp_eqn,
            [0, dt_norm],
            x_norm,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
            dense_output=True
        )

        if sol.success:
            result_norm = sol.y[:, -1]
            # NEW: Transform result from rotating normalized to inertial physical
            result_inertial = transform_rotating_to_inertial(result_norm, tk)
            print("Non-linear propagation successful")
            return result_inertial

    except Exception as e:
        print(f"Error with RK45: {str(e)}")
    
    # If all integration methods failed, use linear propagation (MOSTLY UNCHANGED)
    print("All integration methods failed, revert to fallback")
    
    # Linear propagation
    n_small_steps = 20
    tiny_dt = dt_norm / n_small_steps
    
    # Take many steps with the full dynamics
    current_state = x_norm.copy()
    
    for _ in range(n_small_steps):
        # Evaluate derivatives
        derivatives = cr3bp_eqn(0, current_state)
        
        # Euler step in normalised units
        current_state += derivatives * tiny_dt
    
    # NEW: Transform fallback result to inertial frame
    result_inertial = transform_rotating_to_inertial(current_state, tk)
    
    # Count this as a linear fallback (UNCHANGED)
    linear_prop_count += 1
    
    # Only print occasionally to reduce output spam (UNCHANGED)
    if linear_prop_count % 10 == 0:
        print(f"fallback count: {linear_prop_count}")
    
    return result_inertial  # CHANGED: was result_metric


def STM(XREFk, dt):

    x_norm = np.zeros_like(XREFk)
    x_norm[:3] = metric_to_normalised_pos(XREFk[:3])
    x_norm[3:] = metric_to_normalised_vel(XREFk[3:])
    
    dt_norm = metric_to_normalised_time(dt)
    
    x, y, z = x_norm[0], x_norm[1], x_norm[2]
    
    r1_sq = (x+MU)**2 + y**2 + z**2
    r2_sq = (x-1+MU)**2 + y**2 + z**2
    
    # Safety check - prevent division by very small numbers
    min_safe_dist_sq = 0.01**2
    if r1_sq < min_safe_dist_sq or r2_sq < min_safe_dist_sq:
        print("Warning: Very close to primary bodies in STM calculation")
        # Use a simplified approach (linear) for very close approaches
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)  # Position derivatives = velocity
        return np.eye(6) + A * dt_norm
    
    r1 = np.sqrt(r1_sq)  # Distance to Earth
    r2 = np.sqrt(r2_sq)  # Distance to Moon
        
    r1_cubed = r1**3
    r2_cubed = r2**3
    r1_fifth = r1**5
    r2_fifth = r2**5
    
    # Compute second derivatives of the potential U in normalised units
    try:
        # Main diagonal terms
        Uxx = 1 - (1-MU)/r1_cubed - MU/r2_cubed + 3*(1-MU)*(x+MU)**2/r1_fifth + 3*MU*(x-1+MU)**2/r2_fifth
        Uyy = 1 - (1-MU)/r1_cubed - MU/r2_cubed + 3*(1-MU)*y**2/r1_fifth + 3*MU*y**2/r2_fifth
        Uzz = -(1-MU)/r1_cubed - MU/r2_cubed + 3*(1-MU)*z**2/r1_fifth + 3*MU*z**2/r2_fifth
        
        # Cross terms
        Uxy = 3*((1-MU)*(x+MU)/r1_fifth + MU*(x-1+MU)/r2_fifth)*y
        Uyx = Uxy  # Symmetric
        Uxz = 3*((1-MU)*(x+MU)/r1_fifth + MU*(x-1+MU)/r2_fifth)*z
        Uzx = Uxz  # Symmetric
        Uyz = 3*((1-MU)/r1_fifth + MU/r2_fifth)*y*z
        Uzy = Uyz  # Symmetric
    
    except Exception as exc:
        print(f"  Error in potential derivatives: {exc}")
        # Return simplified STM on error
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        return np.eye(6) + A * dt_norm
    
    # Construct A matrix
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)  # Identity matrix for position derivatives
    
    # Potential derivatives
    A[3, :3] = [Uxx, Uxy, Uxz]
    A[4, :3] = [Uyx, Uyy, Uyz]
    A[5, :3] = [Uzx, Uzy, Uzz]
    
    # Coriolis terms
    A[3, 4] = 2    # 2Ω
    A[4, 3] = -2   # -2Ω
    
    # Compute state transition matrix in normalised units
    try:
        # Matrix exponential approach
        STM_norm = expm(A * dt_norm)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(STM_norm)) or np.any(np.isinf(STM_norm)):
            raise ValueError("STM contains NaN or Inf values")
            
        # STM is dimensionless and doesn't need unit conversion
        return STM_norm
    except Exception as exc:
        print(f"  Matrix exponential failed: {exc}")
        # Fall back to Taylor series approx
        I = np.eye(6)
        return I + A*dt_norm + 0.5*A@A*dt_norm**2
#sources:
#    https://orbital-mechanics.space/the-n-body-problem/Equations-of-Motion-CR3BP.html
#    https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
#   Dr. Shane Ross orbital mechanics lecture
# https://people.unipi.it/tommei/wp-content/uploads/sites/124/2021/08/3body.pdf - page 197
