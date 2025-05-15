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
        
        for i in range(len(time)-1):
            
            #calculate Greenwich Mean Sidereal Time
            tk = time[i] 
            #determine years, months, days
            sec_to_year = 60 * 60 * 24 * 7 * 52
            y = int(tk / sec_to_year) #years
            
            sec_to_month = 60 * 60 * 24 * 7 * 4
            m = int( (tk - y * sec_to_year) / sec_to_month) #months
            
            sec_to_day = 60 * 60 * 24
            d = int( (tk - y * sec_to_year - m * sec_to_month) / sec_to_day) #days
            
            UT_in_sec = tk - y * sec_to_year - m * sec_to_month - d*sec_to_day
            UT = UT_in_sec / (60 * 60) #in hours
            
            J0 = 367*y - int(7 * (y + int((m + 9)/12)) / 4) + int(275 * m / 9) + d + 1721013.5 #Julian date, days
            T0 = (J0 - 2451545) / 36525
            thetaG0 = 100.4606184 + 36000.77004 * T0 + 0.000387933 * T0**2 - 2.583e-8 * T0**3 #Greenwich Mean Sidereal Time, deg
            GMST = thetaG0 + 360.98564724 * UT / 24
            GMST = np.radians(GMST % 360) #normalise to be between 0 and 360 degrees and convert to radians
            
            # UT1 = #Universal Time 1
            # UTC = #Universal Time Coordinated
            #theta = omega * (GMST - (UT1 - UTC)) #angular rotation occurred since Epoch, note that UT1 == UTC therefore:
            omega = 0.26179939 #angular rotational velocity of Earth, rad/hr
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
    Goldstone_coord = [243.1, 35.43, 1.001] 
    
    Goldstone_ECEF_coord = geodetic_to_ECEF(Goldstone_coord)
    Goldstone_ECEF_coord_array = np.full((len(tk), 3), Goldstone_ECEF_coord)  
    Goldstone_ECI_coord, Goldstone_ECI_velocity = ECEF_to_ECI(Goldstone_ECEF_coord_array, tk)
    
    Goldstone_ECI_state = np.zeros_like(truth_data)
    Goldstone_ECI_state[:, 0:3] = Goldstone_ECI_coord  
    Goldstone_ECI_state[:, 3:6] = Goldstone_ECI_velocity  
    
    #Convert truth data BIF (barycentric inertial frame) --> ECI
    
    def BIF_to_ECI(coordinates, time):
        
        #note time is already in ephemeris time (sec)
        
        import spiceypy as spice
        
        spice.furnsh("visualisation processes/input/naif0012.tls")  # Leap seconds kernel
        spice.furnsh("de430.bsp")     # Earth-Moon ephemeris kernel
        
        converted_coord = np.zeros_like(coordinates) #initialise empty array
        Earth_State_Data = np.zeros((len(time), 6))  
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('399', et, 'J2000', 'NONE', '3')[0]
            Earth_State_Data[i] = Earth_State
            
        EarthPositionData = Earth_State_Data[:, 0:3]
        EarthVelocityData = Earth_State_Data[:, 3:6]
            
        converted_coord[:, 0:3] = coordinates[:, 0:3] - EarthPositionData
        converted_coord[:, 3:6] = coordinates[:, 3:6] - EarthVelocityData
        
        return converted_coord
    
    ECI_truth = BIF_to_ECI(truth_data, tk)
        
    #generate range and range rate data
    
    DSN_sim_coord = np.zeros_like(truth_data) #initialise empty array
    
    DSN_sim_range = ECI_truth[:, 0:3] - Canberra_ECI_state[:, 0:3]
    
    range_bias = np.random.normal(0, 7.5/1000/3, DSN_sim_range.shape) # m to km
    range_noise = np.random.normal(0, 3/1000/3, DSN_sim_range.shape) # m to km
    
    DSN_sim_range = DSN_sim_range + range_bias + range_noise
    DSN_sim_coord[:, 0:3] = DSN_sim_range  
    
    DSN_sim_range_rate = ECI_truth[:, 3:6] - Canberra_ECI_state[:, 3:6]
    
    range_rate_bias = np.random.normal(0, 2.5/10000/3, DSN_sim_range_rate.shape) # mm/s to km/s
    range_rate_noise = np.random.normal(0, 1/10000/3, DSN_sim_range_rate.shape) # mm/s to km/s
    
    DSN_sim_range_rate = DSN_sim_range_rate + range_rate_bias + range_rate_noise
    DSN_sim_coord[:, 3:6] = DSN_sim_range_rate
    
    
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
 
    
