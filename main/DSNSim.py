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
        
        phi, Lambda, h = coordinates
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
            
            rotated_coordinates = coordinates[i] * np.linalg.inv(RotationMatrix)
            all_rotated_coordinates[i] = rotated_coordinates #store
            
            omega_radpersec = omega/60/60
            station_velocity = np.cross(np.array([0, 0, omega_radpersec]), rotated_coordinates[i])
            all_rotated_velocities[i] = station_velocity
            
        return all_rotated_coordinates, all_rotated_velocities
            
            
    #Station Coordinates (latitude (deg), longitude (deg), height (km))
    Canberra_coord = [35.31, 149.12, 0.578] #Canberra, ACT, Australia
    
    Canberra_ECEF_coord = geodetic_to_ECEF(Canberra_coord)
    Canberra_ECEF_coord_array = np.full(np.shape(tk), Canberra_ECEF_coord)
    Canberra_ECI_coord, Canberra_ECI_velocity = ECEF_to_ECI(Canberra_ECEF_coord_array, tk)
    
    Canberra_ECI_state = np.zeros_like(truth_data)
    Canberra_ECI_state[:, 0:2] = Canberra_ECI_coord
    Canberra_ECI_state[:, 3:5] = Canberra_ECI_velocity
    
    #do the same for Madrid and Goldstone eventually
    
    #Convert truth data BIF (barycentric inertial frame) --> ECI
    
    def BIF_to_ECI(coordinates, time):
        
        #note time is already in ephemeris time (sec)
        
        import spicepy as spice
       
        converted_coord = np.zeros_like(coordinates) #initialise empty array
        Earth_State_Data = np.zeros_like(coordinates)
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('399', et, 'J2000', 'NONE', '3')[0]
            Earth_State_Data[i] = Earth_State
            
        EarthPositionData = Earth_State_Data[:3]
        EarthVelocityData = Earth_State_Data[3:]
            
        converted_coord[:, 0:2] = coordinates[:, 0:2] - EarthPositionData #converted position data
        converted_coord[:, 3:5] = coordinates[:, 3:5] - EarthVelocityData #converted velocity data
        
        return converted_coord
    
    ECI_truth = BIF_to_ECI(truth_data, tk)
        
    #generate range and range rate data
    
    DSN_sim_coord = np.zeros_like(truth_data) #initialise empty array
    
    range_data = ECI_truth[:, 0:2] - Canberra_ECI_state
    range_rate_data = ECI_truth[:, 3:5] - Canberra_ECI_velocity #figure this out later
    
    range_bias = np.random.normal(0, 7.5/1000/3, np.shape(range_data)) #m to km
    range_noise = np.random.normal(0, 3/1000/3, np.shape(range_data)) #m to km
    DSN_sim_range = range_data + range_bias + range_noise
    DSN_sim_coord[:, 0:2] = DSN_sim_range
    
    range_rate_bias = np.random.normal(0, 2.5/10000/3, np.shape(range_rate_data)) #mm/s to km/s
    range_rate_noise = np.random.normal(0, 1/10000/3, np.shape(range_rate_data)) #mm/s to km/s
    DSN_sim_range_rate = range_rate_data + range_rate_bias + range_rate_noise
    DSN_sim_coord[:, 3:5] = DSN_sim_range_rate
    
    
    def ECI_to_BRF(coordinates, time):
        
        #first convert ECI back to BCI
        #note time is already in ephemeris time (sec)
        
        import spicepy as spice
       
        ECI_to_BRF_coord = np.zeros_like(coordinates) #initialise empty array
        
        for i in range(len(coordinates)):
            
            et = time[i]
            Earth_State = spice.spkezr('399', et, 'J2000', 'NONE', '3')[0]
            Moon_State = spice.spkezr('301', et, 'J2000', 'NONE', '3')[0]
            
            EarthPositionData = Earth_State[:3]
            EarthVelocityData = Earth_State[3:]
        
            MoonPositionData = Moon_State[:3]
            MoonVelocityData = Moon_State[3:]
            
            ECI_to_BCI_position = coordinates[:, 0:2] + EarthPositionData #converted position data
            ECI_to_BCI_velocity = coordinates[:, 3:5] + EarthVelocityData #converted velocity data
      
            #BCI to BRF --> change to multiple states***
            Earth_Moon_Pos = MoonPositionData - EarthPositionData
        
            #defining coordinate axes for BRF
            x_BRF = Earth_Moon_Pos / np.linalg.norm(Earth_Moon_Pos)
        
            h_Earth_Moon = np.cross(Earth_Moon_Pos, MoonVelocityData - EarthVelocityData) #angular momentum of system
            z_BRF = h_Earth_Moon / np.linalg.norm(h_Earth_Moon)
        
            y_BRF = np.cross(z_BRF, x_BRF)
        
            RotationMatrix = np.column_stack((x_BRF, y_BRF, z_BRF))
        
            omega_scalar = np.linalg.norm(h_Earth_Moon) / np.linalg.norm(Earth_Moon_Pos)**2
            omega = omega_scalar * z_BRF
        
            BRF_position = RotationMatrix.T @ ECI_to_BCI_position
            BRF_velocity = RotationMatrix.T @ ECI_to_BCI_velocity - np.cross(omega, BRF_position)
            
            ECI_to_BRF_coord[i] = BRF_position, BRF_velocity
        
        return ECI_to_BRF_coord
    
    DSN_Sim_Data = ECI_to_BRF(DSN_sim_coord, tk)
    
    return DSN_Sim_Data
        
        
    
