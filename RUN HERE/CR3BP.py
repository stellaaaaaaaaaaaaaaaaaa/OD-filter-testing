#NRHO Equations of Motion for Filtering Algorithms

#position data from HALO truth data will be propagated via filters that use
#these equations of motion

#for NRHO: circular restricted three body problem (CR3BP)
#note that additional perturbations will be accounted for as process noise

#NRHO Equations of Motion for Filtering Algorithms

#position data from HALO truth data will be propagated via filters that use
#these equations of motion

#for NRHO: circular restricted three body problem (CR3BP)
#note that additional perturbations will be accounted for as process noise

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import time

# Custom exception for timeout handling - this will keep the processing time down
class TimeoutException(Exception):
    pass

# Counter for linear propagation fallbacks
linear_propagation_count = 0

def NRHOmotion(XREF0, tkminus1, tk):

    global linear_propagation_count
    
    # Ensure XREF0 is a 1D array
    XREF0 = np.array(XREF0).flatten()
    
    # Calculate time difference - use relative time instead of absolute
    dt = tk - tkminus1
    
    # CR3BP equations with relative time
    def cr3bp_eqn(t, Y):
        """CR3BP equations of motion"""
        x, y, z, xdot, ydot, zdot = Y
    
        #Earth-Moon system
        m1 = 5.9722e24  # Earth mass, kg
        m2 = 7.3477e22  # Moon mass, kg
        mu = m2/(m1+m2)  # mass ratio
    
        #derivative vector
        Ydot = np.zeros_like(Y)
        Ydot[:3] = Y[3:]  # Position derivatives = velocity
    
        # Distances to primaries
        sigma = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to Earth
        psi = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to Moon
        
        # Indicates proximity to Earth and Moon
        if sigma < 0.001 or psi < 0.001:
            raise ValueError("Too close to primary bodies")
        
        # Acceleration terms
        Ydot[3] = (
            2 * ydot
            + x
            - (1 - mu) * (x + mu) / sigma**3
            - mu * (x - 1 + mu) / psi**3
            )
        Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
        Ydot[5] = -(1 - mu) * z / sigma**3 - mu * z / psi**3
        
        return Ydot
    
    # Create a wrapper function with timeout
    max_steps = 1000  # Maximum number of function evaluations
    step_count = 0
    start_time = time.time()
    max_time = 20  # Maximum seconds for integration
    
    def cr3bp_eqn_with_timeout(t, Y):
        nonlocal step_count
        
        # Check step count
        step_count += 1
        if step_count > max_steps:
            raise TimeoutException("Too many integration steps")
        
        # Check elapsed time
        if time.time() - start_time > max_time:
            raise TimeoutException("Integration timed out")
        
        return cr3bp_eqn(t, Y)
    
    # Initial integration parameters - use less strict settings
    rtol = 1e-4
    atol = 1e-4
    max_step = dt / 5  # Ensure at least 5 steps
    
    # Adaptive integration with multiple fallback options
    integration_success = False
    max_attempts = 3  # Maximum number of tolerance relaxation attempts
    attempt = 0
    
    while not integration_success and attempt < max_attempts:
        try:
            print(f"Attempt {attempt+1}: Integrating from {tkminus1} to {tk} with rtol={rtol}, atol={atol}")
            
            # Reset timeout counters for each attempt
            step_count = 0
            start_time = time.time()
            
            # Attempt integration with current tolerances
            sol = solve_ivp(
                cr3bp_eqn_with_timeout, 
                [0, dt],  # Use relative time (0 to dt)
                XREF0, 
                method='RK45', 
                rtol=rtol, 
                atol=atol,
                max_step=max_step,
                dense_output=False  # Disable dense output for speed
            )
            
            # Check if integration was successful
            if sol.success:
                integration_success = True
                # If we had to relax tolerances, print a message
                if attempt > 0:
                    print(f"  Integration succeeded with relaxed tolerances: rtol={rtol}, atol={atol}")
            else:
                # Integration failed due to error in ODE solver
                print(f"  Warning: Integration failed with message: {sol.message}")
                # Relax tolerances for next attempt
                attempt += 1
                rtol *= 10
                atol *= 10
                max_step *= 2
                print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
                
        except TimeoutException as exc:
            elapsed = time.time() - start_time
            print(f"  Integration timed out after {elapsed:.1f} seconds ({step_count} steps)")
            # Relax tolerances for next attempt
            attempt += 1
            rtol *= 10
            atol *= 10
            max_step *= 2
            print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
            
        except KeyboardInterrupt:
            # Allow user to interrupt
            print("  KeyboardInterrupt detected. Stopping integration.")
            raise
            
        except Exception as exc:
            print(f"  Error in integration attempt {attempt+1}: {str(exc)}")
            # Relax tolerances for next attempt
            attempt += 1
            rtol *= 10
            atol *= 10
            max_step *= 2
            print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
    
    # If integration succeeded, return the result
    if integration_success:
        return sol.y[:, -1]  # Return the final state
    else:
        # If all attempts failed, use a simpler approach
        print("  All integration attempts failed. Using linear propagation as fallback.")
        # Simple linear propagation as last resort
        linear_state = np.copy(XREF0)
        # Update positions based on velocities
        linear_state[:3] += linear_state[3:] * dt
        linear_propagation_count += 1
        print(f"Linear propagation fallback count: {linear_propagation_count}")
        
        return linear_state


def STM(XREFk, dt):
    
    # Start timing the operation
    start_time = time.time()
    max_time = 5.0  # Maximum time for STM calculation
    
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

# #NRHO Equations of Motion for Filtering Algorithms

# #position data from HALO truth data will be propagated via filters that use
# #these equations of motion

# #for NRHO: circular restricted three body problem (CR3BP)
# #note that additional perturbations will be accounted for as process noise

# import numpy as np
# from scipy.integrate import solve_ivp
# from scipy.linalg import expm
# import time

# # Custom exception for timeout handling - this will keep the processing time down
# class TimeoutException(Exception):
#     pass

# # Counter for linear propagation fallbacks
# linear_propagation_count = 0

# def NRHOmotion(XREF0, tkminus1, tk):

#     global linear_propagation_count
    
#     # Ensure XREF0 is a 1D array
#     XREF0 = np.array(XREF0).flatten()
    
#     # Calculate time difference - use relative time instead of absolute
#     dt = tk - tkminus1
    
#     # CR3BP equations with relative time
#     def cr3bp_eqn(t, Y):
#         """CR3BP equations of motion"""
#         x, y, z, xdot, ydot, zdot = Y
    
#         #Earth-Moon system
#         m1 = 5.9722e24  # Earth mass, kg
#         m2 = 7.3477e22  # Moon mass, kg
#         mu = m2/(m1+m2)  # mass ratio
    
#         #derivative vector
#         Ydot = np.zeros_like(Y)
#         Ydot[:3] = Y[3:]  # Position derivatives = velocity
    
#         # Distances to primaries
#         sigma = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to Earth
#         psi = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to Moon
        
#         # Indicates proximity to Earth and Moon
#         if sigma < 0.001 or psi < 0.001:
#             raise ValueError("Too close to primary bodies")
        
#         # Acceleration terms
#         Ydot[3] = (
#             2 * ydot
#             + x
#             - (1 - mu) * (x + mu) / sigma**3
#             - mu * (x - 1 + mu) / psi**3
#             )
#         Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
#         Ydot[5] = -(1 - mu) * z / sigma**3 - mu * z / psi**3
        
#         return Ydot
    
#     # Create a wrapper function with timeout
#     max_steps = 1000  # Maximum number of function evaluations
#     step_count = 0
#     start_time = time.time()
#     max_time = 2.0  # Maximum seconds for integration
    
#     def cr3bp_eqn_with_timeout(t, Y):
#         nonlocal step_count
        
#         # Check step count
#         step_count += 1
#         if step_count > max_steps:
#             raise TimeoutException("Too many integration steps")
        
#         # Check elapsed time
#         if time.time() - start_time > max_time:
#             raise TimeoutException("Integration timed out")
        
#         return cr3bp_eqn(t, Y)
    
#     # Initial integration parameters - use less strict settings
#     rtol = 1e-4
#     atol = 1e-4
#     max_step = dt / 5  # Ensure at least 5 steps
    
#     # Adaptive integration with multiple fallback options
#     integration_success = False
#     max_attempts = 3  # Maximum number of tolerance relaxation attempts
#     attempt = 0
    
#     while not integration_success and attempt < max_attempts:
#         try:
#             print(f"Attempt {attempt+1}: Integrating from {tkminus1} to {tk} with rtol={rtol}, atol={atol}")
            
#             # Reset timeout counters for each attempt
#             step_count = 0
#             start_time = time.time()
            
#             # Attempt integration with current tolerances
#             sol = solve_ivp(
#                 cr3bp_eqn_with_timeout, 
#                 [0, dt],  # Use relative time (0 to dt)
#                 XREF0, 
#                 method='RK45', 
#                 rtol=rtol, 
#                 atol=atol,
#                 max_step=max_step,
#                 dense_output=False  # Disable dense output for speed
#             )
            
#             # Check if integration was successful
#             if sol.success:
#                 integration_success = True
#                 # If we had to relax tolerances, print a message
#                 if attempt > 0:
#                     print(f"  Integration succeeded with relaxed tolerances: rtol={rtol}, atol={atol}")
#             else:
#                 # Integration failed due to error in ODE solver
#                 print(f"  Warning: Integration failed with message: {sol.message}")
#                 # Relax tolerances for next attempt
#                 attempt += 1
#                 rtol *= 10
#                 atol *= 10
#                 max_step *= 2
#                 print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
                
#         except TimeoutException as exc:
#             elapsed = time.time() - start_time
#             print(f"  Integration timed out after {elapsed:.1f} seconds ({step_count} steps)")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             max_step *= 2
#             print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
            
#         except KeyboardInterrupt:
#             # Allow user to interrupt
#             print("  KeyboardInterrupt detected. Stopping integration.")
#             raise
            
#         except Exception as exc:
#             print(f"  Error in integration attempt {attempt+1}: {str(exc)}")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             max_step *= 2
#             print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
    
#     # If integration succeeded, return the result
#     if integration_success:
#         return sol.y[:, -1]  # Return the final state
#     else:
#         # If all attempts failed, use a simpler approach
#         print("  All integration attempts failed. Using linear propagation as fallback.")
#         # Simple linear propagation as last resort
#         linear_state = np.copy(XREF0)
#         # Update positions based on velocities
#         linear_state[:3] += linear_state[3:] * dt
#         linear_propagation_count += 1
#         print(f"Linear propagation fallback count: {linear_propagation_count}")
        
#         return linear_state


# def STM(XREFk, dt):
    
#     # Start timing the operation
#     start_time = time.time()
#     max_time = 5.0  # Maximum time for STM calculation
    
#     x, y, z = XREFk[0], XREFk[1], XREFk[2]
    
#     #Earth-Moon system
#     m1 = 5.9722e24  # Earth mass, kg
#     m2 = 7.3477e22  # Moon mass, kg
#     mu = m2/(m1+m2)  # mass ratio
     
#     # Compute distances to Earth and Moon
#     r1 = np.sqrt((x+mu)**2 + y**2 + z**2)  # Distance to Earth
#     r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)  # Distance to Moon
        
#     # Compute second derivatives of the potential U
#     try:
#         Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
#         Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
#         Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
#         Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
#         Uyx = Uxy
#         Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
#         Uzx = Uxz
#         Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
#         Uzy = Uyz
    
#     except Exception as exc:
#         print(f"  Error in potential derivatives: {exc}")
#         # Return simplified STM on error
#         A = np.zeros((6, 6))
#         A[:3, 3:] = np.eye(3)
#         return np.eye(6) + A * dt
    
#     # Check for timeout
#     if time.time() - start_time > max_time:
#         print("  STM calculation taking too long, using approximation")
#         # Return simplified STM on timeout
#         A = np.zeros((6, 6))
#         A[:3, 3:] = np.eye(3)
#         return np.eye(6) + A * dt

#     # Construct A matrix (variational equation matrix)
#     A = np.zeros((6, 6))
#     A[:3, 3:] = np.eye(3)  # Identity matrix for position derivatives
    
#     # Potential derivatives
#     A[3, :3] = [Uxx, Uxy, Uxz]
#     A[4, :3] = [Uyx, Uyy, Uyz]
#     A[5, :3] = [Uzx, Uzy, Uzz]
    
#     # Coriolis terms
#     A[3, 4] = 2    # 2Ω
#     A[4, 3] = -2   # -2Ω
    
#     # Compute state transition matrix using two approaches
#     try:
#         # First try matrix exponential
#         STM = expm(A * dt)
        
#         # Check for NaN or Inf values
#         if np.any(np.isnan(STM)) or np.any(np.isinf(STM)):
#             raise ValueError("STM contains NaN or Inf values")
            
#         return STM
#     except Exception as exc:
#         print(f"  Matrix exponential failed: {exc}")
#         # Fallback to Taylor series approximation
#         I = np.eye(6)
#         return I + A*dt + 0.5*A@A*dt**2


# sources:
#    https://orbital-mechanics.space/the-n-body-problem/Equations-of-Motion-CR3BP.html
#    https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
#   Dr. Shane Ross orbital mechanics lecture
# https://people.unipi.it/tommei/wp-content/uploads/sites/124/2021/08/3body.pdf - page 197


# #NRHO Equations of Motion for Filtering Algorithms

# #position data from HALO truth data will be propagated via filters that use
# #these equations of motion

# #for NRHO: circular restricted three body problem (CR3BP)
# #note that additional perturbations will be accounted for as process noise

# import numpy as np
# from scipy.integrate import solve_ivp
# from scipy.linalg import expm
# import time

# #to count how much linear propagation is fallen back on
# linear_propagation_count = 0


# # Custom exception for timeout handling - this will keep the processing time down
# class TimeoutException(Exception):
#     pass

# def NRHOmotion(XREF0, tkminus1, tk):

#     global linear_propagation_count    

#     # Ensure XREF0 is a 1D array
#     XREF0 = np.array(XREF0).flatten()
    
#     # Calculate time difference - use relative time instead of absolute
#     dt = tk - tkminus1
    
#     # CR3BP equations with relative time
#     def cr3bp_eqn(t, Y):
#         """CR3BP equations of motion"""
#         x, y, z, xdot, ydot, zdot = Y
    
#         #Earth-Moon system
#         m1 = 5.9722e24  # Earth mass, kg
#         m2 = 7.3477e22  # Moon mass, kg
#         mu = m2/(m1+m2)  # mass ratio
    
#         #derivative vector
#         Ydot = np.zeros_like(Y)
#         Ydot[:3] = Y[3:]  # Position derivatives = velocity
    
#         # Distances to primaries
#         sigma = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to Earth
#         psi = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to Moon
        
#         # Indicates proximity to Earth and Moon
#         if sigma < 0.001 or psi < 0.001:
#             raise ValueError("Too close to primary bodies")
        
#         # Acceleration terms
#         Ydot[3] = (
#             2 * ydot
#             + x
#             - (1 - mu) * (x + mu) / sigma**3
#             - mu * (x - 1 + mu) / psi**3
#             )
#         Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
#         Ydot[5] = -(1 - mu) * z / sigma**3 - mu * z / psi**3
        
#         return Ydot
    
#     # # Create a wrapper function with timeout  --> uncomment this is debugging interface
#     # max_steps = 1000  # Maximum number of function evaluations
#     # step_count = 0
#     # start_time = time.time()
#     # max_time = 2.0  # Maximum seconds for integration
    
#     # def cr3bp_eqn_with_timeout(t, Y):
#     #     nonlocal step_count
        
#     #     # Check step count
#     #     step_count += 1
#     #     if step_count > max_steps:
#     #         raise TimeoutException("Too many integration steps")
        
#     #     # Check elapsed time
#     #     if time.time() - start_time > max_time:
#     #         raise TimeoutException("Integration timed out")
    
#         return cr3bp_eqn(t, Y)
    
#     # Initial integration parameters - use less strict settings
#     rtol = 1e-3
#     atol = 1e-3
#     max_step = dt / 5  # Ensure at least 5 steps
    
#     # Integration options in case first attempt fails
#     integration_success = False
#     max_attempts = 3  # Maximum number of tolerance relaxation attempts
#     attempt = 0
    
#     while not integration_success and attempt < max_attempts:
#         try:
#             print(f"Attempt {attempt+1}: Integrating from {tkminus1} to {tk} with rtol={rtol}, atol={atol}")
            
#             # Reset timeout counters for each attempt
#             step_count = 0
#             start_time = time.time()
            
#             # Attempt integration with current tolerances
#             sol = solve_ivp(
#                 cr3bp_eqn, 
#                 [0, dt],  # Use relative time (0 to dt)
#                 XREF0, 
#                 method='RK45', 
#                 rtol=rtol, 
#                 atol=atol,
#                 max_step=max_step,
#                 dense_output=True  # Disable dense output for speed
#             )
            
#             # Check if integration was successful
#             if sol.success:
#                 integration_success = True
#                 # If we had to relax tolerances, print a message
#                 if attempt > 0:
#                     print(f"  Integration succeeded with relaxed tolerances: rtol={rtol}, atol={atol}")
#             else:
#                 # Integration failed due to error in ODE solver
#                 print(f"  Warning: Integration failed with message: {sol.message}")
#                 # Relax tolerances for next attempt
#                 attempt += 1
#                 rtol *= 10
#                 atol *= 10
#                 max_step *= 2
#                 print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
                
#         except TimeoutException as exc:
#             elapsed = time.time() - start_time
#             print(f"  Integration timed out after {elapsed:.1f} seconds ({step_count} steps)")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             max_step *= 2
#             print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
            
            
#         except Exception as exc:
#             print(f"  Error in integration attempt {attempt+1}: {str(exc)}")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             max_step *= 2
#             print(f"  Relaxing tolerances for attempt {attempt+1}: rtol={rtol}, atol={atol}")
    
#     # If integration succeeded, return the result
#     if integration_success:
#         return sol.y[:, -1]  # Return the final state
#     else:
#         # If all attempts failed, approximate state via linear propagation
#         print("  All integration attempts failed. Using linear propagation as fallback.")
#         # Simple linear propagation as last resort
#         linear_state = np.copy(XREF0)
#         # Update positions based on velocities
#         linear_state[:3] += linear_state[3:] * dt
#         linear_propagation_count += 1
#         print(f"Linear propagation fallback count: {linear_propagation_count}")
#         return linear_state


# def STM(XREFk, dt):
    
#     x, y, z = XREFk[0], XREFk[1], XREFk[2]
    
#     #Earth-Moon system
#     m1 = 5.9722e24  # Earth mass, kg
#     m2 = 7.3477e22  # Moon mass, kg
#     mu = m2/(m1+m2)  # mass ratio
     
#     # Compute distances to Earth and Moon
#     r1 = np.sqrt((x+mu)**2 + y**2 + z**2)  # Distance to Earth
#     r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)  # Distance to Moon
        
#     # Compute second derivatives of the potential U
#     Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
#     Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
#     Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
#     Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
#     Uyx = Uxy
#     Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
#     Uzx = Uxz
#     Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
#     Uzy = Uyz
    
#     # Construct A matrix (variational equation matrix)
#     A = np.zeros((6, 6))
#     A[:3, 3:] = np.eye(3)  # Identity matrix for position derivatives
    
#     # Potential derivatives
#     A[3, :3] = [Uxx, Uxy, Uxz]
#     A[4, :3] = [Uyx, Uyy, Uyz]
#     A[5, :3] = [Uzx, Uzy, Uzz]
    
#     # Coriolis terms
#     A[3, 4] = 2    # 2Ω
#     A[4, 3] = -2   # -2Ω
    
#     # Compute state transition matrix using two approaches
#     try:
#         # First try matrix exponential
#         STM = expm(A * dt)
        
#         # Check for NaN or Inf values - indicates something wrong with STM for debugging
#         if np.any(np.isnan(STM)) or np.any(np.isinf(STM)):
#             raise ValueError("STM contains NaN or Inf values")
            
#         return STM
    
#     except Exception as exc:
#         print(f"  Matrix exponential failed: {exc}")
#         # If invalid return Taylor series approximation of STM
#         I = np.eye(6)
#         return I + A*dt + 0.5*A@A*dt**2


#sources:
#    https://orbital-mechanics.space/the-n-body-problem/Equations-of-Motion-CR3BP.html
#    https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
#   Dr. Shane Ross orbital mechanics lecture
# https://people.unipi.it/tommei/wp-content/uploads/sites/124/2021/08/3body.pdf - page 197

# #NRHO Equations of Motion for Filtering Algorithms

# #position data from HALO truth data will be propagated via filters that use
# #these equations of motion

# #for NRHO: circular restricted three body problem (CR3BP)
# #note that additional perturbations will be accounted for as process noise

# #input measured time, position, measurement noise
# #propagate via equations of motion
# #output reference state and state transition matrix (STM)
# #note Newtonian coordinates, relative to Earth

# def NRHOmotion(XREF0, tkminus1, tk):
#     """
#     Propagate state using CR3BP equations
    
#     Parameters:
#     -----------
#     XREF0 : numpy.ndarray
#         Initial state vector [x, y, z, vx, vy, vz]
#     tkminus1 : float
#         Initial time
#     tk : float
#         Final time
        
#     Returns:
#     --------
#     XREFk : numpy.ndarray
#         Propagated state vector at time tk
#     """
#     import numpy as np
#     from scipy.integrate import solve_ivp
#     import time
    
#     # Ensure XREF0 is a 1D array
#     XREF0 = np.array(XREF0).flatten()
    
#     def cr3bp_eqn(t, Y):
#         """CR3BP equations of motion"""
#         x, y, z, xdot, ydot, zdot = Y
    
#         #Earth-Moon system
#         m1 = 5.9722e24  # Earth mass, kg
#         m2 = 7.3477e22  # Moon mass, kg
#         mu = m2/(m1+m2)  # mass ratio
    
#         #derivative vector
#         Ydot = np.zeros_like(Y)
#         Ydot[:3] = Y[3:]  # Position derivatives = velocity
    
#         # Distances to primaries
#         sigma = np.sqrt((x + mu)**2 + y**2 + z**2)  # Distance to Earth
#         psi = np.sqrt((x - 1 + mu)**2 + y**2 + z**2)  # Distance to Moon
        
#         # Acceleration terms
#         Ydot[3] = (
#             2 * ydot
#             + x
#             - (1 - mu) * (x + mu) / sigma**3
#             - mu * (x - 1 + mu) / psi**3
#             )
#         Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
#         Ydot[5] = -(1 - mu) * z / sigma**3 - mu * z / psi**3
        
#         return Ydot
    
#     # Initial integration parameters
#     rtol = 1e-6
#     atol = 1e-6
#     max_step = (tk - tkminus1) / 10  # Ensure at least 10 steps
    
#     # Adaptive integration with timeout mechanism
#     integration_success = False
#     max_attempts = 4  # Maximum number of tolerance relaxation attempts
#     attempt = 0
    
#     while not integration_success and attempt < max_attempts:
#         try:
#             # Set timeout for this integration attempt
#             start_time = time.time()
#             timeout = 5.0  # 5 seconds max per integration attempt
            
#             # Attempt integration with current tolerances
#             sol = solve_ivp(
#                 cr3bp_eqn, 
#                 [tkminus1, tk], 
#                 XREF0, 
#                 method='RK45', 
#                 rtol=rtol, 
#                 atol=atol,
#                 max_step=max_step,
#                 first_step=(tk-tkminus1)/20
#             )
            
#             # Check if integration was successful
#             if sol.success:
#                 integration_success = True
#                 # If we had to relax tolerances, print a message
#                 if attempt > 0:
#                     print(f"Integration succeeded with relaxed tolerances: rtol={rtol}, atol={atol}")
#             else:
#                 # Integration failed due to error in ODE solver
#                 print(f"Warning: Integration failed with message: {sol.message}")
#                 # Relax tolerances for next attempt
#                 attempt += 1
#                 rtol *= 10
#                 atol *= 10
#                 print(f"Relaxing tolerances for attempt {attempt}: rtol={rtol}, atol={atol}")
                
#         except KeyboardInterrupt:
#             # Allow user to interrupt
#             raise
#         except Exception as e:
#             print(f"Error in integration attempt {attempt+1}: {str(e)}")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             print(f"Relaxing tolerances for attempt {attempt}: rtol={rtol}, atol={atol}")
            
#         # Check if timeout occurred
#         if time.time() - start_time > timeout and not integration_success:
#             print(f"Integration timed out after {timeout:.1f} seconds")
#             # Relax tolerances for next attempt
#             attempt += 1
#             rtol *= 10
#             atol *= 10
#             print(f"Relaxing tolerances for attempt {attempt}: rtol={rtol}, atol={atol}")
    
#     # If integration succeeded, return the result
#     if integration_success:
#         return sol.y[:, -1]  # Return the final state
#     else:
#         # If all attempts failed, use a simpler approach
#         print("All integration attempts failed. Using linear propagation as fallback.")
#         # Simple linear propagation as last resort
#         dt = tk - tkminus1
#         linear_state = np.copy(XREF0)
#         # Update positions based on velocities
#         linear_state[:3] += linear_state[3:] * dt
#         return linear_state

# def STM(XREFk, dt):
#     """
#     Compute State Transition Matrix for CR3BP
    
#     Parameters:
#     -----------
#     XREFk : numpy.ndarray
#         Current state vector [x, y, z, vx, vy, vz]
#     dt : float
#         Time difference for the STM calculation
        
#     Returns:
#     --------
#     STM : numpy.ndarray
#         State Transition Matrix
#     """
#     import numpy as np
#     from scipy.linalg import expm

#     # Ensure XREFk is properly dimensioned
#     if np.isscalar(XREFk) or XREFk.ndim == 0:
#         raise ValueError("XREFk must be an array with at least one dimension")
    
#     # Handle both 1D and 2D arrays
#     if XREFk.ndim == 1:
#         # It's a 1D array, extract components directly
#         x, y, z = XREFk[0], XREFk[1], XREFk[2]
#     else:
#         # It might be a 2D array, so take the first row if there are multiple
#         x, y, z = XREFk[0, 0], XREFk[0, 1], XREFk[0, 2]
    
#     #Earth-Moon system
#     m1 = 5.9722e24  # Earth mass, kg
#     m2 = 7.3477e22  # Moon mass, kg
#     mu = m2/(m1+m2)  # mass ratio
     
#     # Compute distances to Earth and Moon
#     r1 = np.sqrt((x+mu)**2 + y**2 + z**2)  # Distance to Earth
#     r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2)  # Distance to Moon

#     # Compute second derivatives of the potential U
#     Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
#     Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
#     Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
#     Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
#     Uyx = Uxy
#     Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
#     Uzx = Uxz
#     Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
#     Uzy = Uyz

#     # Construct A matrix (variational equation matrix)
#     A = np.zeros((6, 6))
#     A[:3, 3:] = np.eye(3)  # Identity matrix for position derivatives
    
#     # Potential derivatives
#     A[3, :3] = [Uxx, Uxy, Uxz]
#     A[4, :3] = [Uyx, Uyy, Uyz]
#     A[5, :3] = [Uzx, Uzy, Uzz]
    
#     # Coriolis terms
#     A[3, 4] = 2    # 2Ω
#     A[4, 3] = -2   # -2Ω
    
#     # Compute state transition matrix using matrix exponential
#     STM = expm(A * dt)
#     return STM
    

# # def STM(XREFk, tk):
# #     #STM - this is fairly computationally heavy, hence should be its own function as only EKF uses it
# #     import numpy as np
# #     from scipy.linalg import expm

# #     x, y, z = XREFk[:3]
# #     xdot, ydot, zdot = XREFk[3:]
    
# #     #Earth-Moon system
# #     m1 = 5.9722e24 #Earth mass, kg
# #     m2 = 7.3477e22 #Moon mass, kg
# #     mu = m2/(m1+m2) #mass ratio
     
# #     # Compute distances to Earth and Moon
# #     r1 = np.sqrt((x+mu)**2 + y**2 + z**2)# Distance to Earth
# #     r2 = np.sqrt((x-1+mu)**2 + y**2 + z**2) #Distance to the Moon

# #     # Compute second derivatives of the potential U
# #     Uxx = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*(x+mu)**2/(r1**5) + 3*mu*(x-1+mu)**2/(r2**5)
# #     Uyy = 1 - (1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*y**2/(r1**5) + 3*mu*y**2/(r2**5)
# #     Uzz = -(1-mu)/(r1**3) - mu/(r2**3) + 3*(1-mu)*z**2/(r1**5) + 3*mu*z**2/(r2**5)
# #     Uxy = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*y
# #     Uyx = Uxy
# #     Uxz = 3*((1-mu)*(x+mu)/(r1**5) + mu*(x-1+mu)/(r2**5))*z
# #     Uzx = Uxz
# #     Uyz = 3*((1-mu)/(r1**5) + mu/(r2**5))*y*z
# #     Uzy = Uyz

# #     # Construct A matrix
# #     A = np.zeros((6, 6))
# #     A[:3, 3:] = np.eye(3)  # Identity matrix
# #     A[3, :3] = [Uxx, Uxy, Uxz]
# #     A[4, :3] = [Uyx, Uyy, Uyz]
# #     A[5, :3] = [Uzx, Uzy, Uzz]
# #         #2*omega matrix, represent Coriolis terms
# #     A[3, 3:] = [0, 2, 0]
# #     A[4, 3:] = [-2, 0, 0]
# #     A[5, 3:] = [0, 0, 0]
    
# #     #determine state transition matrix (STM)
# #     # https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Introduction_to_Control_Systems_(Iqbal)/08%3A_State_Variable_Models/8.02%3A_State-Transition_Matrix_and_Asymptotic_Stability#:~:text=with%20Complex%20Roots-,The%20State%2DTransition%20Matrix,vector%2C%20x(t).
# #     STM  = expm(A * tk) # Compute the state transition matrix
# #     return STM
    



# #sources:
# #    https://orbital-mechanics.space/the-n-body-problem/Equations-of-Motion-CR3BP.html
# #    https://orbital-mechanics.space/the-n-body-problem/circular-restricted-three-body-problem.html
# #   Dr. Shane Ross orbital mechanics lecture
# # https://people.unipi.it/tommei/wp-content/uploads/sites/124/2021/08/3body.pdf - page 197


