#Deep Space Network (DSN) Simulation Function

#Takes truth dataset and converts to DSN measurements
#Then converts back to measurements for use in CR3BP.py

# Convert DSN station coordinates from geodetic (latitude, longitude, altitude) to Earth Centred, Earth Fixed (ECEF) Frame -->
# --> Convert these coordinates from ECEF to Earth Centred Inertial (ECI) Frame

#'Truth' data from HALO (xyzuvw, __) -->
# --> Convert truth data from barycentric rotating frame (BRF) to ECI -->
# --> Generate range and range rate data by subtracting the position of the station from the data --> ***
# --> Apply noise to the measurements to generate 'reference data' --->
# --> Convert measurements back to BRF

def generate_DSNSim_from_truth(truth_data, tk):
    
    import numpy as np
    
    def geodetic_to_ECEF(coordinates):
        #https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf - equations source
        
        # Convert latitude and longitude from degrees to radians
        phi = np.radians(coordinates[0])
        Lambda = np.radians(coordinates[1])
        h = coordinates[2]
        
        a = 6378 #equatorial radius, km
        f = 1/298.25 #flattening factor of Earth
        e = 2*f - f**2 #Earth's first eccentricity
        
        RN = a / np.sqrt(1- e**2 * np.sin(phi)**2) #radius of curvature in the prime vertical
        
        #convert to xyz
        x = (RN + h) * np.cos(phi) * np.cos(Lambda)
        y = (RN + h) * np.cos(phi) * np.sin(Lambda)
        z = ((1 - e**2) * RN + h) * np.sin(phi)
        
        ECEF_coordinates = [x, y, z]
        
        return ECEF_coordinates
    
    
    def ECEF_to_ECI(coordinates, time):
        #equations source https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf
        
        all_rotated_coordinates = np.zeros_like(coordinates) #initialised empty array
        all_rotated_velocities = np.zeros_like(coordinates) 
        
        for i in range(len(time)):
            
            J0 = 2451545.0 + tk[i] / 60 / 60 / 24 #convert sec to days

            T0 = (J0 - 2451545) / 36525
            GMST = 280.46061837 + 360.98564736629 * (J0 - 2451545.0) + 0.000387933 * T0**2 - T0**3 / 38710000.0
            GMST = np.radians(GMST % 360) #normalise to be between 0 and 360 degrees and convert to radians
            
            # UT1 = #Universal Time 1
            # UTC = #Universal Time Coordinated
            #theta = omega * (GMST - (UT1 - UTC)) #angular rotation occurred since Epoch, note that UT1 == UTC therefore:
            omega = 0.26179939 / 60 / 60 #angular rotational velocity of Earth, rad/hr to sec
            theta = omega * GMST
            
            #form RotationMatrix
            RotationMatrix = np.zeros((3, 3))
            RotationMatrix[0, :3] = [np.cos(theta), -1*np.sin(theta), 0]
            RotationMatrix[1, :3] = [np.sin(theta), np.cos(theta), 0]
            RotationMatrix[2, :3] = [0, 0, 1]
            
            rotated_coordinates = RotationMatrix.T @ coordinates[i, :3]
            all_rotated_coordinates[i, :3] = rotated_coordinates
            
            omega_radpersec = omega/60/60
            
            station_velocity = np.cross(np.array([0, 0, omega_radpersec]), rotated_coordinates)
            all_rotated_velocities[i, :3] = station_velocity  
            
        return all_rotated_coordinates, all_rotated_velocities
            
            
    #Station Coordinates (latitude (deg), longitude (deg), height (km))
    #for NRHO L2 Southern orbit, can assume no lunar blockages (reason why orbit is so advantageous is constant LoS with Earth)
    #therefore only have to consider Earth blockages for the ground stations
    
    #Canberra, ACT, Australia
    #https://www.cdscc.nasa.gov/Pages/opening_hours.html#:~:text=Location,58'%2053%22%20.574%20East
    Canberra_coord = [-35.40, 149.0, 0.550] 
    
    Canberra_ECEF_coord = geodetic_to_ECEF(Canberra_coord)
    Canberra_ECEF_coord_array = np.full((len(tk), 3), Canberra_ECEF_coord)  
    Canberra_ECI_coord, Canberra_ECI_velocity = ECEF_to_ECI(Canberra_ECEF_coord_array, tk)
    
    Canberra_ECI_state = np.zeros_like(truth_data)
    Canberra_ECI_state[:, 0:3] = Canberra_ECI_coord  
    Canberra_ECI_state[:, 3:6] = Canberra_ECI_velocity
    
    #Madrid, Spain
    #https://www.mdscc.nasa.gov/index.php/en/training-and-visitors-center/visitors-center/horarios-e-informacion-general/#:~:text=Directions,that%20are%20developed%20in%20MDSCC.
    Madrid_coord = [40.43, -4.249, 0.720] 
    
    Madrid_ECEF_coord = geodetic_to_ECEF(Madrid_coord)
    Madrid_ECEF_coord_array = np.full((len(tk), 3), Madrid_ECEF_coord)  
    Madrid_ECI_coord, Madrid_ECI_velocity = ECEF_to_ECI(Madrid_ECEF_coord_array, tk)
    
    Madrid_ECI_state = np.zeros_like(truth_data)
    Madrid_ECI_state[:, 0:3] = Madrid_ECI_coord  
    Madrid_ECI_state[:, 3:6] = Madrid_ECI_velocity  
    
    #Goldstone, California, USA
    #https://pds.nasa.gov/ds-view/pds/viewContext.jsp?identifier=urn%3Anasa%3Apds%3Acontext%3Atelescope%3Agoldstone.dss14_70m&version=1.0
    #elevation: https://ipnpr.jpl.nasa.gov/progress_report/42-196/196A.pdf
    Goldstone_coord = [35.43, -116.9, 1.001] 
    
    Goldstone_ECEF_coord = geodetic_to_ECEF(Goldstone_coord)
    Goldstone_ECEF_coord_array = np.full((len(tk), 3), Goldstone_ECEF_coord)  
    Goldstone_ECI_coord, Goldstone_ECI_velocity = ECEF_to_ECI(Goldstone_ECEF_coord_array, tk)
    
    Goldstone_ECI_state = np.zeros_like(truth_data)
    Goldstone_ECI_state[:, 0:3] = Goldstone_ECI_coord  
    Goldstone_ECI_state[:, 3:6] = Goldstone_ECI_velocity  
    
    #Convert truth data MCF (moon centred inertial frame) --> ECI
    
    def MCF_to_ECI(coordinates, time):
        
        #note time is already in ephemeris time (sec)
        
        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  # Leap seconds kernel
        spice.furnsh("de430.bsp")     # Earth-Moon ephemeris kernel
        
        converted_coord = np.zeros_like(coordinates) #initialise empty array
        Earth_State_Data = np.zeros((len(time), 6))  
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('301', et, 'J2000', 'NONE', '399')[0]
            Earth_State_Data[i] = Earth_State
            
            EarthPositionData = Earth_State_Data[i, :3]
            EarthVelocityData = Earth_State_Data[i, 3:6]
            
            converted_coord[i, 0:3] = coordinates[i, 0:3] + EarthPositionData
            converted_coord[i, 3:6] = coordinates[i, 3:6] + EarthVelocityData
        
        return converted_coord
    
    ECI_truth = MCF_to_ECI(truth_data, tk)
    
    DSN_sim_coord = []
    visible_indices = []
    
    for i in range(len(ECI_truth)):
        
        #check equations, attach sources
        Earth_radius = 6378 #km
        
        #check line of sight for each station
        #https://www.satnow.com/calculators/coverage-angle-of-a-satellite
        
        #determine satellite position relative to each ground station
        #i.e. ground station-to-satellite vector
        
        satellite_Earth_vector = ECI_truth[:, 0:3]
        
        Canberra_Satellite_Vector = ECI_truth[i, 0:3] - Canberra_ECI_state[i, 0:3]
        Madrid_Satellite_Vector = ECI_truth[i, 0:3] - Madrid_ECI_state[i, 0:3]
        Goldstone_Satellite_Vector = ECI_truth[i, 0:3] - Goldstone_ECI_state[i, 0:3]
        
        #note Earth-to-ground-station vectors are = ECI_states
        Earth_Canberra_Vector = Canberra_ECI_state[i, 0:3]
        Earth_Madrid_Vector = Madrid_ECI_state[i, 0:3]
        Earth_Goldstone_Vector = Goldstone_ECI_state[i, 0:3]
        
        Canberra_Earth_mag = np.linalg.norm(Canberra_ECI_state[i, 0:3])
        Madrid_Earth_mag = np.linalg.norm(Madrid_ECI_state[i, 0:3])
        Goldstone_Earth_mag = np.linalg.norm(Goldstone_ECI_state[i, 0:3])
        
        #line of sight- is the Earth blocking the vector between the ground station and the satellite?
        #determine projection vectors (Earth-to-station onto station-to-satellite)
        
        proj_C_parameter = -np.dot(Earth_Canberra_Vector, Canberra_Satellite_Vector)/np.dot(Canberra_Satellite_Vector, Canberra_Satellite_Vector)
        proj_M_parameter = -np.dot(Earth_Madrid_Vector, Madrid_Satellite_Vector)/np.dot(Madrid_Satellite_Vector, Madrid_Satellite_Vector)
        proj_G_parameter = -np.dot(Earth_Goldstone_Vector, Goldstone_Satellite_Vector)/np.dot(Goldstone_Satellite_Vector, Goldstone_Satellite_Vector)
        
        proj_C = proj_C_parameter * Canberra_Satellite_Vector
        proj_M = proj_M_parameter * Madrid_Satellite_Vector
        proj_G = proj_G_parameter * Goldstone_Satellite_Vector
        
        #projection values should be positive --> if projection = negative, no Earth blockage
        if 0 <= proj_C_parameter <= 1:
            closest_point_C = Earth_Canberra_Vector + proj_C
            closest_p_dist_C = np.linalg.norm(closest_point_C)
        else:
            closest_p_dist_C = 6700 #random number definitely over limit
    
        if 0 <= proj_M_parameter <= 1:
            closest_point_M = Earth_Madrid_Vector + proj_M
            closest_p_dist_M = np.linalg.norm(closest_point_M)
        else:
            closest_p_dist_M = 6700
            
        if 0 <= proj_G_parameter <= 1:
            closest_point_G = Earth_Goldstone_Vector + proj_G
            closest_p_dist_G = np.linalg.norm(closest_point_G)
        else:
            closest_p_dist_G = 6700
            
        
        #determine elevation angle
        #https://www.satnow.com/calculators/coverage-angle-of-a-satellite - see image
        
        elevation_mask = np.deg2rad(5) #elevation angle must be greater than 10 deg
        #https://ipnpr.jpl.nasa.gov/2000-2009/progress_report/42-160/160A.pdf
        
        #angles between horizon and satellite to ground vector
        cos_C = np.dot(-Earth_Canberra_Vector, Canberra_Satellite_Vector) / (Canberra_Earth_mag * np.linalg.norm(Canberra_Satellite_Vector))
        cos_M = np.dot(-Earth_Madrid_Vector, Madrid_Satellite_Vector) / (Madrid_Earth_mag * np.linalg.norm(Madrid_Satellite_Vector))
        cos_G = np.dot(-Earth_Goldstone_Vector, Goldstone_Satellite_Vector) / (Goldstone_Earth_mag * np.linalg.norm(Goldstone_Satellite_Vector))
    
        elevation_C = np.pi/2 - np.arccos(cos_C) 
        elevation_M = np.pi/2 - np.arccos(cos_M) 
        elevation_G = np.pi/2 - np.arccos(cos_G) 

        Canberra_values = [Canberra_Earth_mag, elevation_C, closest_p_dist_C]
        Madrid_values = [Madrid_Earth_mag, elevation_M, closest_p_dist_M]
        Goldstone_values = [Goldstone_Earth_mag, elevation_G, closest_p_dist_G]
        
        visible_stations = []
        station_data = [(Canberra_values, closest_p_dist_C), (Madrid_values, closest_p_dist_M), (Goldstone_values, closest_p_dist_G)]
        
        for j, (values, line_of_sight_dist) in enumerate(station_data):
            if line_of_sight_dist > Earth_radius and values[1] > elevation_mask:
                visible_stations.append((j, values[2]))  # (station_index, station_to_satellite_distance)

        if not visible_stations:
            print("no visible stations")
            continue

        #select best station based on distance between spacecraft and each station
        visible_stations_magnitudes = [station[1] for station in visible_stations]
    
        best_station = min(visible_stations_magnitudes)
        best_station_index = [station[0] for station in visible_stations][visible_stations_magnitudes.index(best_station)]

        # Use the station values for further calculations
        ECI_states = [Canberra_ECI_state, Madrid_ECI_state, Goldstone_ECI_state]
        best_station_values = ECI_states[best_station_index]
        
        #generate range and range rate data --> !! convert to individual values
        
        DSN_sim_coord = np.zeros_like(truth_data) #initialise empty array
        
        DSN_sim_range = ECI_truth[i, 0:3] - best_station_values[i, 0:3]
        
        range_bias = np.random.normal(0, 7.5/1000/3, DSN_sim_range.shape) # m to km
        range_noise = np.random.normal(0, 3/1000/3, DSN_sim_range.shape) # m to km
        
        DSN_sim_range = DSN_sim_range + range_bias + range_noise
        DSN_sim_coord[:, 0:3] = DSN_sim_range  
        
        DSN_sim_range_rate = ECI_truth[i, 3:6] - best_station_values[i, 3:6]
        
        range_rate_bias = np.random.normal(0, 2.5/10000/3, DSN_sim_range_rate.shape) # mm/s to km/s
        range_rate_noise = np.random.normal(0, 1/10000/3, DSN_sim_range_rate.shape) # mm/s to km/s
        
        DSN_sim_range_rate = DSN_sim_range_rate + range_rate_bias + range_rate_noise
        
        # Store valid measurement
        DSN_measurement = np.concatenate([DSN_sim_range, DSN_sim_range_rate])
        DSN_sim_coord.append(DSN_measurement)
        visible_indices.append(i)

    # Convert to final reference trajectory (only visible states)
    DSN_sim_coord = np.array(DSN_sim_coord)
    
    if DSN_sim_coord == []:
        
        print("no visible stations for any states")
        return
    
    
    def ECI_to_BRF(coordinates, time):
        
        #first convert ECI back to BCI
        #note time is already in ephemeris time (sec)
        
        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  # Leap seconds kernel
        spice.furnsh("de430.bsp")     # Earth-Moon ephemeris kernel
       
        ECI_to_BRF_coord = np.zeros_like(coordinates) #initialise empty array
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('399', et, 'J2000', 'NONE', '3')[0]
            Moon_State = spice.spkezr('301', et, 'J2000', 'NONE', '3')[0]
            
            EarthPositionData = Earth_State[:3]
            EarthVelocityData = Earth_State[3:]
        
            MoonPositionData = Moon_State[:3]
            MoonVelocityData = Moon_State[3:]
            
            ECI_to_BCI_position = coordinates[i, 0:3] + EarthPositionData
            ECI_to_BCI_velocity = coordinates[i, 3:6] + EarthVelocityData
      
            #BCI to BRF --> see HALO paper 3.2.3
            Earth_Moon_Pos = MoonPositionData - EarthPositionData
        
            u_R = Earth_Moon_Pos / np.linalg.norm(Earth_Moon_Pos)
        
            h_Earth_Moon = np.cross(Earth_Moon_Pos, MoonVelocityData - EarthVelocityData) #angular momentum of system
            u_theta = h_Earth_Moon / np.linalg.norm(h_Earth_Moon)
        
            u_z = np.cross(u_theta, u_R)
        
            RotationMatrix = np.column_stack((u_R, u_theta, u_z))
        
            omega_scalar = np.linalg.norm(h_Earth_Moon) / np.linalg.norm(Earth_Moon_Pos)**2
            omega = omega_scalar * u_z
        
            BRF_position = RotationMatrix.T @ ECI_to_BCI_position
            BRF_velocity = RotationMatrix.T @ (ECI_to_BCI_velocity - np.cross(omega, BRF_position))
            
            ECI_to_BRF_coord[i, 0:3] = BRF_position
            ECI_to_BRF_coord[i, 3:6] = BRF_velocity
        
        return ECI_to_BRF_coord
    
    DSN_Sim_Data = ECI_to_BRF(DSN_sim_coord, tk)
    
    return DSN_Sim_Data  
 
    
