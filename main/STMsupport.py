#HALO propagator STM required functions

#courtesy of Yang

#the variational_equtions doesn't consider about the point mass of Earth while the cr3bp_jacobian_moon_inertial only considers about the point mass of the Moon. 
#so you need to combine them together for your Jacobian matrix and STM accordingly. 

from prop.accelharmonic import accelharmonic
import numpy as np

def G_AccelHarmonic_fd(r, E, n_max, m_max, Cnm, Snm, GM, R_ref, eps=1.0):
    G = np.zeros((3, 3))
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        da = accelharmonic(
            r + dr, E, n_max, m_max, Cnm, Snm, GM, R_ref
        ) - accelharmonic(r - dr, E, n_max, m_max, Cnm, Snm, GM, R_ref)
        G[:, i] = da / (2 * eps)
    return G
 

def variational_equations(t, yPhi, E, n_max, m_max, Cnm, Snm, GM, R_ref):
    r = yPhi[0:3]
    v = yPhi[3:6]
    Phi = yPhi[6:].reshape((6, 6))
    a = accelharmonic(r, E, n_max, m_max, Cnm, Snm, GM, R_ref)
    G = G_AccelHarmonic_fd(r, E, n_max, m_max, Cnm, Snm, GM, R_ref)
    dfdy = np.zeros((6, 6))
    dfdy[0:3, 3:6] = np.eye(3)
    dfdy[3:6, 0:3] = G
    Phip = dfdy @ Phi
    yPhip = np.zeros(42)
    yPhip[0:3] = v
    yPhip[3:6] = a
    yPhip[6:] = Phip.flatten()
    return yPhip

 
def cr3bp_jacobian_moon_inertial(r, r_em, mu_moon, mu_earth):
    """
    Compute the Jacobian of acceleration w.r.t position in the Moon-centered inertial frame.
    """
    r_norm = np.linalg.norm(r)
    r_e_vec = r + r_em
    r_e_norm = np.linalg.norm(r_e_vec)
 
    # Moon's gravity contribution
    term_moon = (
        -mu_moon / r_norm**3 * (np.eye(3) - 3 * np.outer(r, r) / r_norm**2)
    )
    # Earth's gravity contribution
    term_earth = (
        -mu_earth
        / r_e_norm**3
        * (np.eye(3) - 3 * np.outer(r_e_vec, r_e_vec) / r_e_norm**2)
    )
    return term_moon + term_earth
 
 
def cr3bp_variational_moon_inertial(t, yPhi, mu_moon, mu_earth, r_em):
    """
    Combined state + STM propagation in the Moon-centered inertial frame using CR3BP.
    yPhi: [r(3), v(3), Phi(36)]
    """
    r = yPhi[:3]
    v = yPhi[3:6]
    Phi = yPhi[6:].reshape((6, 6))
    r_norm = np.linalg.norm(r)
    r_e_vec = r + r_em
    r_e_norm = np.linalg.norm(r_e_vec)
 
    # Acceleration: Moon + Earth + indirect (tidal)
    a = (
        -mu_moon * r / r_norm**3
        - mu_earth * r_e_vec / r_e_norm**3
        + mu_earth * r_em / np.linalg.norm(r_em) ** 3
    )
 
    # Jacobian of acceleration
    jac = cr3bp_jacobian_moon_inertial(r, r_em, mu_moon, mu_earth)
 
    # Assemble A matrix
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = jac
 
    # STM propagation
    Phip = A @ Phi
 
    # Pack derivatives
    dydt = np.zeros_like(yPhi)
    dydt[:3] = v
    dydt[3:6] = a
    dydt[6:] = Phip.flatten()
    return dydt