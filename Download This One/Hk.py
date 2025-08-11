#Hk computation function

def compute_Hk(position, velocity, Xk_range, Xk_range_rate):
      
    import numpy as np
    
    #measyrement mapping matrix
    #partial derivatives
    #dr/dx dr/dv
    #drr/dx drr/dv
    
    #partial derivatives
    drdx = position/ Xk_range
    drdv = np.zeros(3)
   
    drrdx= velocity / Xk_range - (np.dot(position, velocity) / Xk_range ** 3) * position
    drrdv = position / Xk_range
   
    #assemble H matrix 
    H = np.zeros((2, 6))
    H[0, :3] = drdx
    H[0, 3:6] = drdv
    H[1, :3] = drrdx
    H[1, 3:6] = drrdv
       
    return H