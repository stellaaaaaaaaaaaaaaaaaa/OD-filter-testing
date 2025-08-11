#Cubature Kalman Filter

#more complex than EKF, less than UKF since cubature rule used over unscented transform
#more likely to perform well for the more unstable parts of the NRHO
#Cubature Kalman Filter with Improved Numerical Stability

#ref https://kalman-filter.com/cubature-kalman-filter/

def run_cubature_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, is_it_hybrid, H_criteria, stable):
    
    import numpy as np
    import time
    from scipy import linalg
    from pointprop import point_propagation
    from Hk import compute_Hk
    
    #initialise arrays
    state_dim = 6  #should be 6 (position and velocity)
    
    ckf_results = np.zeros_like(Xtruth)
    covariance_results = np.zeros((len(Xtruth), state_dim, state_dim))
    residual_results = np.zeros((len(Xtruth), 2))
    entropy_results = np.zeros(len(Xtruth))
    
    #first state is initial
    ckf_results[0] = X_initial
    
    if isinstance(initial_covar, np.ndarray) and initial_covar.shape == (6, 6):
        #initial_covar is already a full covariance matrix from previous filter
        #allows more complex initial_covar from the interface rathe than just scalar
        covariance_results[0] = initial_covar
    else:
        #initialisation for scalar initial_covar
        P0 = np.eye(6)
        P0[:3, :3] *= initial_covar**2
        P0[3:, 3:] *= initial_covar/1000**2
        covariance_results[0] = P0
    
    #CKF parameters (spread)
    nx = state_dim
    xi = np.sqrt(nx)  #standard scaling factor for cubature points
    num_points = 2 * nx  #total number of cubature points
    weight = 1.0 / num_points  #equal weights for all points
    
    print(f"CKF - Starting filter with {num_points} cubature points")
    
    for k in range(len(Xtruth) - 1):
        print(f"CKF - Processing time step {k}/{len(Xtruth)-1}")
        
        if k+1 >= len(DSN_Sim_measurements):
            print(f"ERROR:DSN measurement array alignment mismatched")
            break
        if k+1 >= len(ground_station_state):
            print(f"ERROR:ground station state mismatched")
            break
        
        try:
            #initialisation (a priori) t = k -1
            x_prev = ckf_results[k].copy()  #previous estimated state
            P_prev = covariance_results[k].copy()  #previous covariance
            
            tkminus1 = tk[k]  
            tk_next = tk[k+1]
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            #compute root of covariance for cubature point generation
            try:
                #regulaisation to ensure not instable
                regularized_P = P_prev + np.eye(nx) * 1e-8
                sqrt_P = linalg.cholesky(regularized_P, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed, using eigendecomposition")
                #eigendecomposition was an option, but every time this seemed to fall back on this, the filter was totally destabilised
                eigvals, eigvecs = linalg.eigh(regularized_P)
                eigvals = np.maximum(eigvals, 1e-8)
                sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))
            
            #generate cubature points for propagation
            cubature_points = np.zeros((num_points, nx))
            
            for j in range(nx):
                # Positive points: x + xi * column of P_root
                cubature_points[j] = x_prev + xi * sqrt_P[:, j]
                # Negative points: x - xi * column of P_root
                cubature_points[j+nx] = x_prev - xi * sqrt_P[:, j]
            
            
            #time update - propagate cubature points
            prop_cubature_points = np.zeros_like(cubature_points)
            
            for j in range(num_points):
                try:
                    prop_cubature_points[j] = point_propagation(cubature_points[j], tkminus1, tk_next)
                    
                    #debugging to ensure factors not broken
                    if not np.all(np.isfinite(prop_cubature_points[j])):
                        raise ValueError("Propagation resulted in non-finite values")
                        
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    #use linear propagation as fallback, again destabilises though
                    dt = tk_next - tkminus1
                    state = cubature_points[j].copy()
                    state[:3] += state[3:] * dt
                    prop_cubature_points[j] = state
            
            #time update measurement prediction and covariance prediction
            
            x_pred = np.zeros(6) 
            
            for j in range(num_points):
                x_pred += weight * prop_cubature_points[j]
            
            x_pred_pos = x_pred[0:3] - ground_station_state[k+1, 0:3]
            x_pred_vel = x_pred[3:6] - ground_station_state[k+1, 3:6]
            
            cub_point_measurements = np.zeros((num_points, 2)) 
            
            for j in range(num_points):
                
                position = prop_cubature_points[j, 0:3] - ground_station_state[k+1, 0:3]       
                velocity = prop_cubature_points[j, 3:6] - ground_station_state[k+1, 3:6] 
            
                prop_range = np.linalg.norm(position)
                prop_range_rate = np.dot(position, velocity) / np.linalg.norm(position) 
            
                cub_point_measurements[j] = np.array([prop_range, prop_range_rate])
            
            expected_meas = np.zeros(2)
            position = np.zeros(6)
            velocity = np.zeros(6)
            
            for j in range(num_points):
                expected_meas += weight * cub_point_measurements[j]
                
            y_pred = expected_meas
            y_pred_range, y_pred_range_rate = y_pred
            

            #covariance prediction
            P_pred = np.zeros((nx, nx))
            for j in range(num_points):
                diff = prop_cubature_points[j] - x_pred
                P_pred += weight * np.outer(diff, diff)
            
            #process noise
            P_pred += Qd
            
            #measurement update t = k
            yk = np.array(DSN_Sim_measurements[k+1]) #range and range rate measurement received
            
            #predicted measurement
            y_pred = expected_meas
            
            #innovation
            innovation = yk - y_pred
            
            innovation_norm = np.linalg.norm(innovation)
            if innovation_norm > 5.0:  #threshold in km
                print("Large innovation detected - possible divergence")
            
            #innovation covariance
            #Pyy = P_pred + Rk
            Pyy = np.zeros((2, 2))
            for j in range(num_points):
                diff_meas = cub_point_measurements[j] - y_pred
                Pyy += weight * np.outer(diff_meas, diff_meas)

            Pyy += Rk
        
        
            #cross-correlation matrix
            #Pxy = P_pred  
            Pxy = np.zeros((nx, 2))  # 6x2 matrix
            for j in range(num_points):
                diff_state = prop_cubature_points[j] - x_pred
                diff_meas = cub_point_measurements[j] - y_pred
                Pxy += weight * np.outer(diff_state, diff_meas)
            
            #Kalman gain
            Kk = Pxy @ linalg.inv(Pyy)
            
            #update state
            x_updated = x_pred + Kk @ innovation

            
            #update covariance  
            #P_updated = P_pred - Kk @ Pxy.T #RMSE = 12.15km 
            
            P_updated = P_pred - Kk @ Pyy @ Kk.T
            
            #ensure symmetry 
            P_updated = (P_updated + P_updated.T) / 2
            
            #calculate entropy [important for hybrid implementation but may be used to compare results]
            d = P_updated.shape[0] #dimension of covariance matrix
            logval = (2*np.pi*np.e) ** d * np.linalg.det(P_updated)
            H = 0.5 * np.log(abs(logval)) 
            
            #eigenvalue
            eigvals = linalg.eigvalsh(P_updated)
            if np.any(eigvals < 1e-8):
                print("  Warning: Covariance has negative eigenvalues, applying correction")
                eigvals, eigvecs = linalg.eigh(P_updated)
                eigvals = np.maximum(eigvals, 1e-8)
                P_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            #store results
            ckf_results[k+1] = x_updated
            covariance_results[k+1] = P_updated
            residual_results[k+1] = innovation
            entropy_results[k+1] = abs(H)
            
        except Exception as exc:
            import traceback
            print(f"Error at time step {k}:")
            print(f"  Error type: {type(exc).__name__}")
            print(f"  Error message: {str(exc)}")
            print("  Traceback:")
            traceback.print_exc()
            
            print(f"  Using fallback: copying previous state")
            ckf_results[k+1] = ckf_results[k]
            covariance_results[k+1] = covariance_results[k]
            residual_results[k+1] = np.zeros(2)
            
        # if is_it_hybrid == 1 and abs(H) > H_criteria and stable == 1 and k > 0:
        #     print("entropy > criteria, stable region finished, swapping to unstable filter")
        #     return (ckf_results[:k+2], covariance_results[:k+2], 
        #             residual_results[:k+2], entropy_results[:k+2])
        #     return ckf_results, covariance_results, residual_results, entropy_results
        
        # if is_it_hybrid == 1 and abs(H) < H_criteria and stable == 0:
        #     print("entropy < criteria, unstable region finished, swapping to stable filter")
        #     return (ckf_results[:k+2], covariance_results[:k+2], 
        #             residual_results[:k+2], entropy_results[:k+2])
        #     return ckf_results, covariance_results, residual_results, entropy_results
    
    return ckf_results, covariance_results, residual_results, entropy_results




