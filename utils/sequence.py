import numpy as np

def weight_select(loss, i, flag=True):
    
    i = i + 1
    accumu = 0
    accumu += i ** 2
    
    
    if flag:
    
        return i / np.sqrt(accumu) * loss
        
    else:
        return (np.sqrt(accumu) / i - 0.8) * loss
        
