#convert truth data to measurement data

import numpy as np

#from HALO propagator
postruth_data = np.load("NRHOTruthPoints.npy") #position data x, y, z

#set measurement bias (systematic), select arbitrarily
rangebias_std = 7.5 #metres

#set measurement noise (random), select arbitrarily 
rangenoise_std = 3 #metres

#generate measurement data by incorporating bias and noise
rangeinput_data = postruth_data+rangebias_std+np.random.normal(0, rangenoise_std, postruth_data.shape) #convert truth to measurement by adding measurement noise
