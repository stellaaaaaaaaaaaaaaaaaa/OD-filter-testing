#Unscented Kalman Filter (UKF) Algorithm 

#ref

def run_unscented_kalman_filter(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, Rk, Qd, initial_covar, is_it_hybrid, H_criteria, stable):
  
    import numpy as np
    import time
    from scipy import linalg
    from pointprop import point_propagation
    from Hk import compute_Hk
    

    #initialise arrays
    state_dim = 6  #should be 6 (position and velocity)
    
    ukf_results = np.zeros_like(Xtruth)
    covariance_results = np.zeros((len(Xtruth), state_dim, state_dim))
    residual_results = np.zeros((len(Xtruth), 2))
    entropy_results = np.zeros(len(Xtruth))
    
    ukf_results[0] = X_initial
    
    if isinstance(initial_covar, np.ndarray) and initial_covar.shape == (6, 6):
        #initial_covar is already a full covariance matrix from previous filter
        #allows more complex initial_covar from the interface rathe than just scalar
        covariance_results[0] = initial_covar
    else:
    #initialise with scalar
        P0 = np.eye(6)
        P0[:3, :3] *= initial_covar**2
        P0[3:, 3:] *= initial_covar/1000**2
        covariance_results[0] = P0

    
    #unscented transform parameters
    L = state_dim
    alpha = 0.1  #small positive value (determines spread of sigma points)
    beta = 2.0   #optimal for Gaussian distributions
    kappa = 3-L  #secondary scaling parameter
    
    lambda_param = (alpha**2) * (L + kappa) - L
    gamma = np.sqrt(L + lambda_param)  #scaling factor for sigma points
    
    #weights for propagated points
    W0m = lambda_param / (L + lambda_param)
    W0c = W0m + (1 - alpha**2 + beta)
    
    Wim = 1.0 / (2.0 * (L + lambda_param))
    Wic = Wim
    
    Wm = np.ones(2*L + 1) * Wim
    Wm[0] = W0m
    
    Wc = np.ones(2*L + 1) * Wic
    Wc[0] = W0c
    
    for k in range(len(Xtruth) - 1):
        print(f"UKF - Processing time step {k}/{len(Xtruth)-2}")
        
        try:
            #initialisation (t = k-1)
            x_prev = ukf_results[k].copy()  #state
            P_prev = covariance_results[k].copy()  #covariance
            
            tkminus1 = tk[k]  
            tk_next = tk[k+1]  
            
            print(f"  Times: {tkminus1} -> {tk_next}")
            
            #compute square root of covariance via Cholesky decomposition
            try:
                ##regulaisation to ensure not instable
                regularised_P = P_prev + np.eye(L) * 1e-8
                sqrt_P = linalg.cholesky(regularised_P, lower=True)
            except linalg.LinAlgError:
                print("  Warning: Cholesky decomposition failed, using eigendecomposition")
                #eigendecomposition was an option, but every time this seemed to fall back on this, the filter was totally destabilised
                eigvals, eigvecs = linalg.eigh(regularised_P)
                eigvals = np.maximum(eigvals, 1e-8) 
                sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))
            
            #generate sigma points
            sigma_points = np.zeros((2*L + 1, L))
            
            sigma_points[0] = x_prev
            for j in range(L):
                sigma_points[j+1] = x_prev + gamma * sqrt_P[:, j]
                sigma_points[j+L+1] = x_prev - gamma * sqrt_P[:, j]
            
            #time update - propagate sigma points
            prop_sigma_points = np.zeros_like(sigma_points)
            
            for j in range(2*L + 1):
                try:
                    prop_sigma_points[j] = point_propagation(sigma_points[j], tkminus1, tk_next)
                    
                    if not np.all(np.isfinite(prop_sigma_points[j])):
                        print(f"  Point {j}: Non-finite after propagation")
                        print(f"    Input: {sigma_points[j]}")
                        print(f"    Output: {prop_sigma_points[j]}")
                        
                except Exception as exc:
                    print(f"  Point {j} propagation failed: {exc}")
                    print(f"    Input state: {sigma_points[j]}")
                    raise exc  
                        
            
                        
                except Exception as exc:
                    print(f"  Error propagating point {j}: {exc}")
                    #linear propagation as fallback -- this was more for debugging, not ever adequate for propagation
                    dt = tk_next - tkminus1
                    state = sigma_points[j].copy()
                    state[:3] += state[3:] * dt
                    prop_sigma_points[j] = state
            
            #time update measurement and covariance
            #determine predicted measurement from weighted prop state
            x_pred = np.zeros(L)
            for j in range(2*L + 1):
                x_pred += Wm[j] * prop_sigma_points[j]
            
            x_pred_pos = x_pred[0:3] - ground_station_state[k+1, 0:3]
            x_pred_vel = x_pred[3:6] - ground_station_state[k+1, 3:6]
            
            #predicted covar
            P_pred = np.zeros((L, L))
            for j in range(2*L + 1):
                diff = prop_sigma_points[j] - x_pred
                P_pred += Wc[j] * np.outer(diff, diff)
            
            #process noise
            P_pred += Qd
            
            #predicted measurement mean 
            sigma_point_measurements = np.zeros((2*L+1, 2)) 
            
            for j in range(2*L+1):
                position = prop_sigma_points[j, 0:3] - ground_station_state[k+1, 0:3]       
                velocity = prop_sigma_points[j, 3:6] - ground_station_state[k+1, 3:6] 
            
                prop_range = np.linalg.norm(position)
                prop_range_rate = np.dot(position, velocity) / np.linalg.norm(position) 
            
                sigma_point_measurements[j] = np.array([prop_range, prop_range_rate])
            
            expected_meas = np.zeros(2)
            
            for j in range(2*L+1):
                expected_meas += Wm[j] * sigma_point_measurements[j]
                
            y_pred = expected_meas
            y_pred_range, y_pred_range_rate = y_pred
            
            #innovation covariance and cross-correlation
            Pyy = np.zeros((2, 2))
            Pxy = np.zeros((L, 2))
            
            for j in range(2*L + 1):
                diff_y = sigma_point_measurements[j] - y_pred
                diff_x = prop_sigma_points[j] - x_pred
                
                Pyy += Wc[j] * np.outer(diff_y, diff_y)
                Pxy += Wc[j] * np.outer(diff_x, diff_y)
            
            #add measurement noise
            Pyy += Rk
            
            #Kalman gain and state update
            Kk = Pxy @ linalg.inv(Pyy)

            #current measurement
            yk = np.array(DSN_Sim_measurements[k+1])
            
            #innovation
            innovation = yk - y_pred  
            
            #final state
            x_updated = x_pred + Kk @ innovation
                
            # 
            #P_updated = P_pred - Kk @ Pxy.T
            
            P_updated = P_pred - Kk @ Pyy @ Kk.T
            
            #I = np.eye(L)
            #Hk = compute_Hk(x_pred_pos, x_pred_vel, y_pred_range, y_pred_range_rate)
            #P_updated = (I - Kk @ Hk) @ P_pred @ (I - Kk @ Hk).T + Kk @ Rk @ Kk.T
            
            #ensure covariance symmetric
            P_updated = (P_updated + P_updated.T) / 2
            
            #eigenvalue check for debugging -- if negative clear divergence
            eigvals = linalg.eigvalsh(P_updated)
            if np.any(eigvals < 1e-8):  
                print("  Warning: Covariance has negative eigenvalues, applying correction")
                eigvals, eigvecs = linalg.eigh(P_updated)
                eigvals = np.maximum(eigvals, 1e-8)  
                P_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            
            #calculate entropy [important for hybrid implementation BUT may be used to compare results]
            d = P_updated.shape[0] #dimension of covariance matrix
            det_P = np.linalg.det(P_updated)
            H = 0.5 * (d * np.log(2 * np.pi * np.e) + np.log(abs(det_P)))
            
            ukf_results[k+1] = x_updated
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
            ukf_results[k+1] = ukf_results[k]
            covariance_results[k+1] = covariance_results[k]
            residual_results[k+1] = np.zeros(2)  
            
            #Old delete later 
        # if is_it_hybrid == 1 and H > H_criteria and stable == 1:
        #     print("entropy > criteria, stable region finished, swapping to unstable filter")
        #     return (ukf_results[:k+2], covariance_results[:k+2], 
        #             residual_results[:k+2], entropy_results[:k+2])
        
        # if is_it_hybrid == 1 and H < H_criteria and stable == 0:
        #     print("entropy < criteria, unstable region finished, swapping to stable filter")
        #     return (ukf_results[:k+2], covariance_results[:k+2], 
        #             residual_results[:k+2], entropy_results[:k+2])
    
    return ukf_results, covariance_results, residual_results, entropy_results


