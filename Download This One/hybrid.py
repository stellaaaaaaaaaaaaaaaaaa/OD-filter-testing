#Hybridisation

#note that this code will be incorporated into the interface as an option

#set boundary conditions for when filters swap
#i.e. for when the orbit dynamics become unstable
#these will be based on position data for simplicity
#(estimated region of instability)
#use differential entropy to swap between filters?
# https://amostech.com/TechnicalPapers/2022/Poster/Chow.pdf
# H = 0.5 * log|2*pi*e*P|
# P = state covariance at each state

def run_hybrid(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, stable, unstable, 
               H_criteria_stable, H_criteria_unstable, c, Pcc, Qd, Rk, initial_covar):
  
    #this is the entropy function 
    
    from EKF import run_extended_kalman_filter
    from CKF import run_cubature_kalman_filter
    from UKF import run_unscented_kalman_filter
    
    import numpy as np
    
    #convert inputs to arrays for indexing
    Xtruth = np.asarray(Xtruth)
    X_initial = np.asarray(X_initial)
    DSN_Sim_measurements = np.asarray(DSN_Sim_measurements)
    tk = np.asarray(tk)
    initial_covar = np.asarray(initial_covar)
    
    #initialise
    hybrid_results = []
    covariance_results = []
    residual_results = []
    entropy_results = []
    
    #for indexing
    current_state = np.array(X_initial, dtype=float).copy()
    current_covar = np.array(initial_covar, dtype=float).copy()
    
    #burn in
    BURN_IN_STEPS = 4000
    
    #first filter state
    current_filter = 'stable'  #start with stable filter
    current_index = 0
    total_steps = len(tk)
    filter_switch_count = 0
    
    #function processes in 'chunks'
    #so at the end of each chunk the entropies are checked
    CHUNK_SIZE = 1000
    
    #count when the user-set threshold is exceeded
    threshold_exceed_count = 0
    consecutive_steps = 250
    
    
    while current_index < total_steps:
        
        # is filter segment the first filter segment
        if current_index == 0 or threshold_exceed_count >= consecutive_steps:
            print(f"Starting new segment at index {current_index}/{total_steps}")
            print(f"Current filter: {current_filter}")
            
            #reset threshold count when filter switched
            threshold_exceed_count = 0
        
        #determine which filter to use and threshold
        if current_filter == 'stable':
            filter_name = stable
            entropy_threshold = H_criteria_stable
        else:
            filter_name = unstable
            entropy_threshold = H_criteria_unstable
        
        #determine chunk boundaries
        chunk_start = current_index
        chunk_end = min(current_index + CHUNK_SIZE, total_steps)
        
        #chunks after the first, we need to include one overlapping point to ensure continuity in the filter
        if current_index > 0 and len(hybrid_results) > 0:
            #go back one step to overlap
            chunk_start = current_index - 1
            use_overlap = True
        else:
            use_overlap = False
        
        #split inputs into truth data
        Xtruth_chunk = Xtruth[chunk_start:chunk_end, :]
        DSN_Sim_measurements_chunk = DSN_Sim_measurements[chunk_start:chunk_end, :]
        tk_chunk = tk[chunk_start:chunk_end]
        
        if ground_station_state is not None:
            ground_station_state_arr = np.asarray(ground_station_state)
            if ground_station_state_arr.ndim > 1 and ground_station_state_arr.shape[0] == len(tk):
                ground_station_state_chunk = ground_station_state_arr[chunk_start:chunk_end, :]
            else:
                ground_station_state_chunk = ground_station_state_arr
        else:
            ground_station_state_chunk = None
        
        print(f"\nProcessing chunk: indices {chunk_start} to {chunk_end}")
        print(f"Filter: {filter_name}, Overlap: {use_overlap}")
        
        #run filter
        try:
            if filter_name == 'EKF':
                results = run_extended_kalman_filter(
                    Xtruth_chunk, current_state, DSN_Sim_measurements_chunk, 
                    ground_station_state_chunk, tk_chunk, Rk, Qd, current_covar, 
                    1, entropy_threshold, 1
                )
            elif filter_name == 'CKF':
                results = run_cubature_kalman_filter(
                    Xtruth_chunk, current_state, DSN_Sim_measurements_chunk, 
                    ground_station_state_chunk, tk_chunk, Rk, Qd, current_covar, 
                    1, entropy_threshold, 1
                )
            elif filter_name == 'UKF':
                results = run_unscented_kalman_filter(
                    Xtruth_chunk, current_state, DSN_Sim_measurements_chunk, 
                    ground_station_state_chunk, tk_chunk, Rk, Qd, current_covar, 
                    1, entropy_threshold, 1
                )
            else:
                raise ValueError(f"Unknown filter type: {filter_name}")
            
            state, covariance, residual, entropy = results
            
            #skip first results so no overlap
            if use_overlap:
                state = state[1:]
                covariance = covariance[1:]
                residual = residual[1:]
                entropy = entropy[1:]
                
            
            #check for invalid entropy values
            zero_entropy_mask = (entropy == 0.0) | np.isnan(entropy) | np.isinf(entropy)
            if np.any(zero_entropy_mask):
                print(f"WARNING: Found {np.sum(zero_entropy_mask)} invalid entropy values")
                #replace if occurrence
                for i in range(len(entropy)):
                    if zero_entropy_mask[i]:
                        if len(entropy_results) > 0:
                            entropy[i] = entropy_results[-1]
                        elif i > 0:
                            entropy[i] = entropy[i-1]
                        else:
                            #estimate via covariance
                            entropy[i] = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * covariance[i]))
                        print(f"  Replaced entropy at local index {i} with {entropy[i]:.4f}")
            
            #switch or no switch
            should_switch = False
            switch_index = None
            
            #store results and check if threshold has been met
            for i in range(len(state)):
                hybrid_results.append(state[i])
                covariance_results.append(covariance[i])
                residual_results.append(residual[i])
                entropy_results.append(entropy[i])
                
                global_idx = current_index + i
                if use_overlap:
                    global_idx -= 1
                
                print(f"Step {global_idx}: Entropy = {entropy[i]:.4f}, Threshold = {entropy_threshold:.4f}")
                
                #check if condition met
                if current_filter == 'stable':
                    threshold_condition = entropy[i] > entropy_threshold
                    condition_text = "exceeded"
                else:
                    threshold_condition = entropy[i] < entropy_threshold
                    condition_text = "dropped below"
                
                if threshold_condition:
                    threshold_exceed_count += 1
                    print(f"  → Entropy {condition_text} threshold ({threshold_exceed_count}/{consecutive_steps})")
                    
                    if threshold_exceed_count >= consecutive_steps and switch_index is None:
                        print(f"\n*** FILTER SWITCH TRIGGERED ***")
                        should_switch = True
                        switch_index = i
                        #finish processing chunk totally then swap
                else:
                    if threshold_exceed_count > 0:
                        print(f"  → Resetting counter (was {threshold_exceed_count}/{consecutive_steps})")
                    threshold_exceed_count = 0
            
            #update state and covariance so the filter continues to perform as usual despite chunk type processing
            if len(state) > 0:
                current_state = state[-1].copy()
                current_covar = covariance[-1].copy()
                print(f"Updated state for next chunk: {current_state[:3]} (position)")
            
            #keep track of index
            current_index = chunk_end
            
            #do we need filters to switch
            if should_switch and current_index < total_steps:
                filter_switch_count += 1
                
                #switch
                old_filter = current_filter
                current_filter = 'unstable' if current_filter == 'stable' else 'stable'
                print(f"\nSwitching from {old_filter} to {current_filter} filter")
                
                #burn-in
                burn_in_start_idx = max(0, len(hybrid_results) - BURN_IN_STEPS)
                burn_in_steps = len(hybrid_results) - burn_in_start_idx
                
                if burn_in_steps > 0:
                    print(f"Performing burn-in: using last {burn_in_steps} steps to warm up new filter")
                    
                    #start
                    if burn_in_start_idx > 0:
                        burn_in_state = hybrid_results[burn_in_start_idx-1].copy()
                    else:
                        burn_in_state = X_initial.copy()

                    #reset covariance for transition between filters
                    reset_covar = np.eye(6)
                    transition_noise_factor = 20000
                    if initial_covar.ndim > 1:
                        reset_covar[:3, :3] *= initial_covar[0,0]**2
                        reset_covar[3:, 3:] *= (initial_covar[0,0]/1000)**2
                    else:
                        reset_covar[:3, :3] *= initial_covar**2
                        reset_covar[3:, 3:] *= (initial_covar/1000)**2
                    burn_in_covar = reset_covar * transition_noise_factor
                    
                    #run burn-in with new filter
                    burn_in_end_idx = burn_in_start_idx + burn_in_steps
                    
                    Xtruth_burnin = Xtruth[burn_in_start_idx:burn_in_end_idx, :]
                    DSN_Sim_measurements_burnin = DSN_Sim_measurements[burn_in_start_idx:burn_in_end_idx, :]
                    tk_burnin = tk[burn_in_start_idx:burn_in_end_idx]
                    
                    if ground_station_state is not None and ground_station_state_arr.ndim > 1:
                        ground_station_state_burnin = ground_station_state_arr[burn_in_start_idx:burn_in_end_idx, :]
                    else:
                        ground_station_state_burnin = ground_station_state
                    
                    #start burn in with new filter
                    new_filter_name = stable if current_filter == 'stable' else unstable
                    new_entropy_threshold = H_criteria_stable if current_filter == 'stable' else H_criteria_unstable
                    
                    print(f"Running {new_filter_name} burn-in from index {burn_in_start_idx} to {burn_in_end_idx}")
                    
                    if new_filter_name == 'EKF':
                        burn_results = run_extended_kalman_filter(
                            Xtruth_burnin, burn_in_state, DSN_Sim_measurements_burnin, 
                            ground_station_state_burnin, tk_burnin, Rk, Qd, burn_in_covar, 
                            1, new_entropy_threshold, 1
                        )
                    elif new_filter_name == 'CKF':
                        burn_results = run_cubature_kalman_filter(
                            Xtruth_burnin, burn_in_state, DSN_Sim_measurements_burnin, 
                            ground_station_state_burnin, tk_burnin, Rk, Qd, burn_in_covar, 
                            1, new_entropy_threshold, 1
                        )
                    elif new_filter_name == 'UKF':
                        burn_results = run_unscented_kalman_filter(
                            Xtruth_burnin, burn_in_state, DSN_Sim_measurements_burnin, 
                            ground_station_state_burnin, tk_burnin, Rk, Qd, burn_in_covar, 
                            1, new_entropy_threshold, 1
                        )
                    
                    #use burn results as last state, rather than direct state and covariance from last filter
                   #this is because covariance is not translatable, causes filter divergence
                    burn_state, burn_covar, burn_residual, burn_entropy = burn_results
                    
                    #burn-in results are just for warming up the filter
                    #final state and covariance from the burn-in to continue
                    if len(burn_state) > 0:
                        current_state = burn_state[-1].copy()
                        current_covar = burn_covar[-1].copy()
                    
                    print(f"Burn-in complete, discarded {len(burn_state)} burn-in results")
                    
                    #no burn in results are stored in the final results!!
                    
                #reset count for switch
                threshold_exceed_count = 0
                
        except Exception as e:
            print(f"Error running filter: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    #ensure results are arrays
    hybrid_results = np.array(hybrid_results)
    covariance_results = np.array(covariance_results)
    residual_results = np.array(residual_results)
    entropy_results = np.array(entropy_results)
    
    print(f"\n{'='*60}")
    print(f"Hybrid filter completed successfully!")
    print(f"Total filter switches: {filter_switch_count}")
    
    return hybrid_results, covariance_results, residual_results, entropy_results




def run_hybrid_time_based(Xtruth, X_initial, DSN_Sim_measurements, ground_station_state, tk, 
                         stable, unstable, switch_times, Pcc, Qd, Rk, initial_covar):
    
    from EKF import run_extended_kalman_filter
    from CKF import run_cubature_kalman_filter
    from UKF import run_unscented_kalman_filter
    
    import numpy as np
    
    #convert inputs to arrays for indexing
    Xtruth = np.asarray(Xtruth)
    X_initial = np.asarray(X_initial)
    DSN_Sim_measurements = np.asarray(DSN_Sim_measurements)
    tk = np.asarray(tk)
    initial_covar = np.asarray(initial_covar)
    
    #initialise
    hybrid_results = []
    covariance_results = []
    residual_results = []
    entropy_results = []
    
    #sort times to be in order
    switch_times = sorted(switch_times)
    
    #form intervals from switch times
    time_intervals = []
    if len(switch_times) > 0:
        #first switch
        time_intervals.append((float(tk[0]), float(switch_times[0])))
        
        #middle switches
        for i in range(len(switch_times) - 1):
            time_intervals.append((float(switch_times[i]), float(switch_times[i + 1])))
        
        #last switch
        time_intervals.append((float(switch_times[-1]), float(tk[-1])))
    else:
        #no switches, run normally
        time_intervals.append((float(tk[0]), float(tk[-1])))
    
    current_state = np.array(X_initial, dtype=float).copy()
    current_covar = np.array(initial_covar, dtype=float).copy()
    
    #burn in
    BURN_IN_STEPS = 20
    
    #run filter for each time interval
    for interval_idx, (t_start, t_end) in enumerate(time_intervals):
        
        #set which filter to use first
        use_stable = (interval_idx % 2 == 0)
        
        #interval indices
        if interval_idx == 0:
            #first
            interval_mask = (tk >= t_start) & (tk <= t_end)
        else:
            #next intervals remove start time so no duplicate/competing data stored
            interval_mask = (tk > t_start) & (tk <= t_end)
        
        interval_indices = np.where(interval_mask)[0]
        
        #same as entropy, burn in overlaps with last filter so new filter can converge
        burn_in_indices = interval_indices
        burn_in_state = current_state
        burn_in_covar = current_covar
        
        if interval_idx > 0:  
            global_start_idx = interval_indices[0]
            
            #start burn in index
            burn_in_start_idx = max(0, global_start_idx - BURN_IN_STEPS)
            
            #extend data set for iteration
            burn_in_indices = np.arange(burn_in_start_idx, interval_indices[-1] + 1)
            
            print(f"Burn-in period: including {global_start_idx - burn_in_start_idx} extra steps")
            
            #pass previous state
            if len(hybrid_results) >= BURN_IN_STEPS:
                burn_in_state = np.array(hybrid_results[-BURN_IN_STEPS], dtype=float).copy()
                burn_in_covar = np.array(covariance_results[-BURN_IN_STEPS], dtype=float).copy()
                print(f"Using burn-in state from {BURN_IN_STEPS} steps back")
        
        print(f"Total points including burn-in: {len(burn_in_indices)}")
        
        #exclude burn in data
        try:
            Xtruth_interval = Xtruth[burn_in_indices, :]
            DSN_Sim_measurements_interval = DSN_Sim_measurements[burn_in_indices, :]
            tk_interval = tk[burn_in_indices]
        except IndexError as e:
            print(f"IndexError when extracting interval data: {e}")
            print(f"Xtruth shape: {Xtruth.shape}")
            print(f"DSN_Sim_measurements shape: {DSN_Sim_measurements.shape}")
            print(f"tk shape: {tk.shape}")
            print(f"burn_in_indices shape: {burn_in_indices.shape}")
            print(f"burn_in_indices range: {burn_in_indices[0]} to {burn_in_indices[-1]}")
            raise
        
        if ground_station_state is not None:
            ground_station_state = np.asarray(ground_station_state)
            if ground_station_state.ndim > 1 and ground_station_state.shape[0] == len(tk):
                ground_station_state_interval = ground_station_state[burn_in_indices, :]
            else:
                ground_station_state_interval = ground_station_state
        else:
            ground_station_state_interval = None
        
        #run filters
        try:
            if use_stable:
                print(f" EXECUTING {stable} filter (stable) from t={tk_interval[0]:.2e} to t={tk_interval[-1]:.2e}")
                
                if stable == 'EKF':
                    results = run_extended_kalman_filter(
                        Xtruth_interval, burn_in_state, DSN_Sim_measurements_interval, 
                        ground_station_state_interval, tk_interval, Rk, Qd, burn_in_covar, 
                        0, 1, 1  #using default H_criteria values
                    )
                elif stable == 'CKF':
                    results = run_cubature_kalman_filter(
                        Xtruth_interval, burn_in_state, DSN_Sim_measurements_interval, 
                        ground_station_state_interval, tk_interval, Rk, Qd, burn_in_covar, 
                        0, 1, 1  #using default H_criteria values
                    )
                else:
                    raise ValueError(f"Unknown stable filter type: {stable}")
            else:
                print(f" EXECUTING {unstable} filter (unstable) from t={tk_interval[0]:.2e} to t={tk_interval[-1]:.2e}")
                
                if unstable == 'CKF':
                    results = run_cubature_kalman_filter(
                        Xtruth_interval, burn_in_state, DSN_Sim_measurements_interval, 
                        ground_station_state_interval, tk_interval, Rk, Qd, burn_in_covar, 
                        0, 0, 0  #using default H_criteria values
                    )
                elif unstable == 'UKF':
                    results = run_unscented_kalman_filter(
                        Xtruth_interval, burn_in_state, DSN_Sim_measurements_interval, 
                        ground_station_state_interval, tk_interval, Rk, Qd, burn_in_covar, 
                        0, 0, 0  #using default H_criteria values
                    )
                else:
                    raise ValueError(f"Unknown unstable filter type: {unstable}")
            
        except Exception as e:
            print(f"Error running filter in interval {interval_idx}: {e}")
            print(f"Filter type: {'stable' if use_stable else 'unstable'}")
            print(f"Interval data shapes:")
            print(f"  Xtruth_interval: {Xtruth_interval.shape}")
            print(f"  DSN_Sim_measurements_interval: {DSN_Sim_measurements_interval.shape}")
            print(f"  tk_interval: {tk_interval.shape}")
            print(f"  burn_in_state: {burn_in_state.shape}")
            print(f"  burn_in_covar: {burn_in_covar.shape}")
            raise
        
        state, covariance, residual, entropy = results
        
        #results are arrays
        state = np.asarray(state)
        covariance = np.asarray(covariance)
        residual = np.asarray(residual)
        entropy = np.asarray(entropy)
        
        print(f"Filter results shapes: state={state.shape}, covariance={covariance.shape}")
        
        #delete burn in results so that they're not included in the final results
        if interval_idx > 0 and len(burn_in_indices) > len(interval_indices):
            burn_in_count = len(burn_in_indices) - len(interval_indices)
            state = state[burn_in_count:]
            covariance = covariance[burn_in_count:]
            residual = residual[burn_in_count:]
            entropy = entropy[burn_in_count:]
            print(f"Discarded {burn_in_count} burn-in results")
    
        
        #store results
        hybrid_results.extend(state.tolist())
        covariance_results.extend(covariance.tolist())
        residual_results.extend(residual.tolist())
        entropy_results.extend(entropy.tolist())
        
        #update state for the next interval (nothing else passed on)
        current_state = np.array(state[-1], dtype=float).copy()
        
        #reset covariance for next interval
        if interval_idx < len(time_intervals) - 1:  #not the last interval
            reset_covar = np.eye(6)
            transition_noise_factor = 10000
            reset_covar[:3, :3] *= initial_covar**2
            reset_covar[3:, 3:] *= initial_covar/1000**2
            current_covar = reset_covar*transition_noise_factor
        else:
            current_covar = np.array(covariance[-1], dtype=float).copy()
    
    #convert total results to arrays
    hybrid_results = np.array(hybrid_results)
    covariance_results = np.array(covariance_results)
    residual_results = np.array(residual_results)
    entropy_results = np.array(entropy_results)
    
    print(f"\nHybrid filter completed successfully!")
    
    return hybrid_results, covariance_results, residual_results, entropy_results

