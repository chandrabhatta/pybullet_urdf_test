import math
import numpy as np

#Function to determine the risk factor of the robot trajectory
#Inputs:
#   - obstacleDistances: an array of the distances to the closest obstacle
#   
#Outputs:
#   - riskFactor: the risk factor of the robot trajectory
def riskFactor(obstacleDistances): 
    #Initializing risk factor value and dimension size of the input
    riskFactor = 0
    dimensions = len(np.array(obstacleDistances).shape)
        
    #Check the dimensions of the input. If 2D, transform to 1D iwth minimum distance per step.
    if dimensions == 1:
        distances = obstacleDistances
    elif dimensions == 2:
        distances = [min(obstacleDistances[i]) for i in range(len(obstacleDistances))]
    else:
        raise ValueError("Error: the obstacleDistances array has an invalid number of dimensions")
    
    #Calculate the sum of the risk factor, with 1 over the closest distance
    for i in range(len(distances)):
        riskFactor += (1 / (distances[i]))
    
    #Determine and return the final value by dividing by the total number of steps
    riskFactor = riskFactor / np.array(distances).shape[0]
    return riskFactor