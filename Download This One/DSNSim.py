#Deep Space Network (DSN) Simulation Function

#Takes truth dataset and converts to DSN measurements
#Then converts back to measurements for use in CR3BP.py

# Convert DSN station coordinates from geodetic (latitude, longitude, altitude) to Earth Centred, Earth Fixed (ECEF) Frame -->
# --> Convert these coordinates from ECEF to Earth Centred Inertial (ECI) Frame

#'Truth' data from HALO (xyzuvw, __) -->
# --> Convert truth data from barycentric rotating frame (BRF) to ECI -->
# --> Generate range and range rate data by subtracting the position of the station from the data --> ***
# --> Apply noise to the measurements to generate 'reference data' --->
# --> Convert measurements back to MCI

def DSN_generation(truth_data, tk):
    
    import numpy as np
    
    def geodetic_to_ECEF(coordinates):
        #https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf - equations source
        
        #convert latitude and longitude from degrees to radians
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
        
        ECEF_coordinates = [x, y, z]#assume tk is already J2000 ephemeris time
        et = tk[0]  #use tk directly without conversion

        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  #leap seconds kernel
        spice.furnsh("de430.bsp")     # Earth-Moon ephemeris kernel
        
        utc_time = spice.et2utc(et, 'C', 0)
        print(f"Date: {utc_time}")
        
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
            theta = GMST
            
            #form RotationMatrix
            RotationMatrix = np.zeros((3, 3))
            RotationMatrix[0, :3] = [np.cos(theta), -1*np.sin(theta), 0]
            RotationMatrix[1, :3] = [np.sin(theta), np.cos(theta), 0]
            RotationMatrix[2, :3] = [0, 0, 1]
            
            rotated_coordinates = RotationMatrix.T @ coordinates[i, :3]
            all_rotated_coordinates[i, :3] = rotated_coordinates
            
            station_velocity = np.cross(np.array([0, 0, omega]), rotated_coordinates)
            all_rotated_velocities[i, :3] = station_velocity  
            
        state = np.zeros((len(time), 6))
        state[:, :3] = all_rotated_coordinates
        state[:, 3:6] = all_rotated_velocities
            
        return state
    
    
    def ECI_to_BCI(coordinates, time):
        
        #first convert ECI back to BCI
        #note time is already in ephemeris time (sec)
        
        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  #leap seconds kernel
        spice.furnsh("de430.bsp")     #Earth-Moon ephemeris kernel
       
        converted_coord = np.zeros_like(coordinates) #initialise empty array
        
        m_e = 5.9722e24  # kg
        m_m = 7.3477e22   # kg
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('399', et, 'J2000', 'NONE', '0')[0]
            Moon_State = spice.spkezr('301', et, 'J2000', 'NONE', '0')[0]
            
            barypos = (m_e * Earth_State[:3] + m_m * Moon_State[:3]) / (m_e + m_m)
            baryvel = (m_e * Earth_State[3:6] + m_m * Moon_State[3:6]) / (m_e + m_m)
            
            EarthPositionData = Earth_State[:3] - barypos
            EarthVelocityData = Earth_State[3:6] - baryvel
            
            converted_coord[i, 0:3] = coordinates[i, 0:3] + EarthPositionData
            converted_coord[i, 3:6] = coordinates[i, 3:6] + EarthVelocityData
            
        return converted_coord
    
    def ECI_to_MCI(coordinates, time):
        
        #first convert ECI back to MCI
        #note time is already in ephemeris time (sec)
        
        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  # Leap seconds kernel
        spice.furnsh("de430.bsp")     #Earth-Moon ephemeris kernel
       
        converted_coord = np.zeros_like(coordinates) #initialise empty array
        
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Moon_State = spice.spkezr('301', et, 'J2000', 'NONE', '0')[0]
            
            converted_coord[i, 0:3] = coordinates[i, 0:3] - Moon_State[:3]
            converted_coord[i, 3:6] = coordinates[i, 3:6] - Moon_State[3:6]    
            
        return converted_coord
    
    #Station Coordinates (latitude (deg), longitude (deg), height (km))
    #for NRHO L2 Southern orbit, can assume no lunar blockages (reason why orbit is so advantageous is constant LoS with Earth)
    #therefore only have to consider Earth blockages for the ground stations
    
    #Canberra, ACT, Australia
    #https://www.cdscc.nasa.gov/Pages/opening_hours.html#:~:text=Location,58'%2053%22%20.574%20East
    Canberra_coord = [-35.40, 149.0, 0.550] 
    
    Canberra_ECEF_coord = geodetic_to_ECEF(Canberra_coord)
    Canberra_ECEF_coord_array = np.full((len(tk), 3), Canberra_ECEF_coord)  
    Canberra_ECI_state = ECEF_to_ECI(Canberra_ECEF_coord_array, tk)
    Canberra_BCI_state = ECI_to_BCI(Canberra_ECI_state, tk)
    
    
    #Madrid, Spain
    #https://www.mdscc.nasa.gov/index.php/en/training-and-visitors-center/visitors-center/horarios-e-informacion-general/#:~:text=Directions,that%20are%20developed%20in%20MDSCC.
    Madrid_coord = [40.43, -4.249, 0.720] 
    
    Madrid_ECEF_coord = geodetic_to_ECEF(Madrid_coord)
    Madrid_ECEF_coord_array = np.full((len(tk), 3), Madrid_ECEF_coord)  
    Madrid_ECI_state = ECEF_to_ECI(Madrid_ECEF_coord_array, tk)
    Madrid_BCI_state = ECI_to_BCI(Madrid_ECI_state, tk)
    
  
    #Goldstone, California, USA
    #https://pds.nasa.gov/ds-view/pds/viewContext.jsp?identifier=urn%3Anasa%3Apds%3Acontext%3Atelescope%3Agoldstone.dss14_70m&version=1.0
    #elevation: https://ipnpr.jpl.nasa.gov/progress_report/42-196/196A.pdf
    Goldstone_coord = [35.43, -116.9, 1.001] 
    
    Goldstone_ECEF_coord = geodetic_to_ECEF(Goldstone_coord)
    Goldstone_ECEF_coord_array = np.full((len(tk), 3), Goldstone_ECEF_coord)  
    Goldstone_ECI_state = ECEF_to_ECI(Goldstone_ECEF_coord_array, tk)
    Goldstone_BCI_state = ECI_to_BCI(Goldstone_ECI_state, tk)
    
    DSN_Sim = []
    selected_stations = []
    #initialise array for ground stations
    
    for i in range(len(truth_data)):
        
        visible_stations = []
        
        #need to determine which ground station communicates with the spacecraft
        #so which ground stations have line of sight (-80 < elevation < 80)
        
        #elevation angle should be within bounds 0 --> 180
       #     #https://www.satnow.com/calculators/coverage-angle-of-a-satellite
        
        #determine satellite position relative to each ground station
        #i.e. ground station-to-satellite vector
        
        Canberra_Satellite_Vector = truth_data[i, 0:3] - Canberra_BCI_state[i, 0:3]
        Madrid_Satellite_Vector = truth_data[i, 0:3] - Madrid_BCI_state[i, 0:3]
        Goldstone_Satellite_Vector = truth_data[i, 0:3] - Goldstone_BCI_state[i, 0:3]
        
        #note Earth-to-ground-station vectors are = ECI_states
        Earth_Canberra_Vector = Canberra_BCI_state[i, 0:3]
        Earth_Madrid_Vector = Madrid_BCI_state[i, 0:3]
        Earth_Goldstone_Vector = Goldstone_BCI_state[i, 0:3]
        
        Canberra_Earth_mag = np.linalg.norm(Canberra_BCI_state[i, 0:3])
        Madrid_Earth_mag = np.linalg.norm(Madrid_BCI_state[i, 0:3])
        Goldstone_Earth_mag = np.linalg.norm(Goldstone_BCI_state[i, 0:3])
    
        Canberra_Satellite_distance = np.linalg.norm(Canberra_Satellite_Vector)    
        Madrid_Satellite_distance = np.linalg.norm(Madrid_Satellite_Vector)    
        Goldstone_Satellite_distance = np.linalg.norm(Goldstone_Satellite_Vector)    
    
        #determine elevation angle
        #https://www.satnow.com/calculators/coverage-angle-of-a-satellite - see image
        
        elevation_mask = np.deg2rad(10) #elevation angle must be greater than 10 deg
        #https://ipnpr.jpl.nasa.gov/2000-2009/progress_report/42-160/160A.pdf
        
        #angles between horizon and satellite to ground vector
        cos_C = np.dot(-Earth_Canberra_Vector, Canberra_Satellite_Vector) / (Canberra_Earth_mag * np.linalg.norm(Canberra_Satellite_Vector))
        cos_M = np.dot(-Earth_Madrid_Vector, Madrid_Satellite_Vector) / (Madrid_Earth_mag * np.linalg.norm(Madrid_Satellite_Vector))
        cos_G = np.dot(-Earth_Goldstone_Vector, Goldstone_Satellite_Vector) / (Goldstone_Earth_mag * np.linalg.norm(Goldstone_Satellite_Vector))
    
        elevation_C = np.pi/2 - np.arccos(cos_C) 
        elevation_M = np.pi/2 - np.arccos(cos_M) 
        elevation_G = np.pi/2 - np.arccos(cos_G) 

        #collect visible stations
        if elevation_C > elevation_mask:
            distance_C = np.linalg.norm(Canberra_Satellite_Vector)
            visible_stations.append((0, 'Canberra', Canberra_BCI_state[i], distance_C))
            
        if elevation_M > elevation_mask:
            distance_M = np.linalg.norm(Madrid_Satellite_Vector)
            visible_stations.append((1, 'Madrid', Madrid_BCI_state[i], distance_M))
            
        if elevation_G > elevation_mask:
            distance_G = np.linalg.norm(Goldstone_Satellite_Vector)
            visible_stations.append((2, 'Goldstone', Goldstone_BCI_state[i], distance_G))
        
        if not visible_stations:
            DSN_Sim.append(None)
            selected_stations.append(np.zeros(6))
            continue
        
        #select best station (closest)
        best_station = min(visible_stations, key=lambda x: x[3])
        station_index, station_name, best_station_state, distance = best_station
    
        position = truth_data[i, 0:3] - best_station_state[0:3]       
        velocity = truth_data[i, 3:6] - best_station_state[3:6] 

        #convert to scalars
        RANGE = np.linalg.norm(position)
        RANGE_RATE = np.dot(position, velocity) / np.linalg.norm(position) #radial, angle measurement
        
        #add noise and bias
        
        range_bias = np.random.normal(0, 7.5/3/1000) # m to km
        range_noise = np.random.normal(0, 3/3/1000) # m to km
        DSN_sim_range = RANGE + range_bias + range_noise
        
        range_rate_bias = np.random.normal(0, (2.5/3)*1e-6) # mm/s to km/s
        range_rate_noise = np.random.normal(0, (1/3)*1e-6) # mm/s to km/s
        DSN_sim_range_rate = RANGE_RATE + range_rate_bias + range_rate_noise
        
        selected_stations.append(best_station_state)
        
        DSN_Sim.append([DSN_sim_range, DSN_sim_range_rate])
    
    ground_station_state = np.array(selected_stations)
    
    return DSN_Sim, ground_station_state